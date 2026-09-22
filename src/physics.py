"""6軸垂直多関節ロボットアーム(肘型+球状手首)の真値シミュレータ。

軸構成はEPSON C8A901S(6軸垂直多関節、肘型+球状手首)を参考にする。細かな
諸元(リンク長・質量・慣性など)は一致させておらず、軸の向き(DHパラメータの
alpha)のみ参考にした標準的な「肘型マニピュレータ+球状手首」構成(Spongの
教科書等で標準的な例)を用いる:

    i | a_i | alpha_i | d_i  | theta_i
    1 | 0   | 90°     | 0.3  | q1 (腰・ヨー)
    2 | 0.5 | 0       | 0    | q2 (肩)
    3 | 0.5 | 0       | 0    | q3 (肘)
    4 | 0   | 90°     | 0.5  | q4 (前腕ロール)
    5 | 0   | -90°    | 0    | q5 (手首ピッチ)
    6 | 0   | 0       | 0.1  | q6 (手首ロール)

q1-q3は姉妹プロジェクト scaraSurrogate の Phase3(ヨー+肩・肘)と同じ軸構成。
q4-q6が球状手首(3軸の回転軸が1点で交わる)を構成するため、Phase3で使った
ブロック対角の簡略化は使えず、一般的な空間剛体力学(DHパラメータ+再帰的
ニュートン・オイラー法, RNEA)で運動方程式を構築する。

RNEAで質量行列M(q)(単位加速度ごとの列を求める)とバイアス力
bias(q,q_dot) = C(q,q_dot)q_dot + G(q)(重力込みの1回のRNEA呼び出し)を計算し、
フォワードダイナミクス q_ddot = M(q)^-1 (tau - bias - friction) でシミュレートする。

リンクの質量・重心・慣性テンソルは、DHパラメータの物理的な意味とは独立に
(細かな諸元は今回のプロジェクトの主眼ではないため)、各リンク自身の局所座標系
において合理的な範囲で任意に設定する(一様棒近似、各軸等方の慣性)。
"""
import numpy as np

GRAVITY = 9.81
N_JOINTS = 6

# DHパラメータ(修正DH規約: Craig)。標準DH表(issue #1参照)
#   i | a_i | alpha_i | d_i  | theta_i
#   1 | 0   | 90°     | 0.3  | q1
#   2 | 0.5 | 0       | 0    | q2
#   3 | 0.5 | 0       | 0    | q3
#   4 | 0   | 90°     | 0.5  | q4
#   5 | 0   | -90°    | 0    | q5
#   6 | 0   | 0       | 0.1  | q6
# を修正DH規約に変換(frame iの構成にはa_{i-1},alpha_{i-1}(1つ前の行のリンク
# 諸元)とd_i,theta_i(自分の行)を使う)。d_iは値そのものは変わらない。
DH_A = np.array([0.0, 0.0, 0.5, 0.5, 0.0, 0.0])
DH_ALPHA = np.array([0.0, np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2])
DH_D = np.array([0.3, 0.0, 0.0, 0.5, 0.0, 0.1])

# 各リンクの質量・重心(自リンク座標系i内)・慣性テンソル(重心まわり、自リンク
# 座標系i内、等方近似)。修正DH規約ではframe iはjoint iの位置にあり、リンクiは
# frame iからframe i+1に向かって伸びるため、重心はframe (i+1)への変換オフセット
# (DH_A[i+1], -DH_D[i+1]*sin(DH_ALPHA[i+1]), DH_D[i+1]*cos(DH_ALPHA[i+1]))の
# 半分の位置に置く(一様棒近似、Phase1-3のlc=l/2規約の3D版)。最終リンク(6)は
# 次の関節がないため、手首先端に妥当な慣性を持たせるための任意値とする。
LINK_COM = np.array(
    [
        [0.0, 0.0, 0.15],  # link1: d1=0.3の半分
        [0.25, 0.0, 0.0],  # link2: a2=0.5の半分
        [0.25, 0.0, 0.0],  # link3: a3=0.5の半分
        [0.0, 0.0, 0.25],  # link4: d4=0.5の半分
        [0.0, 0.0, 0.15],  # link5: 球状手首(実際のオフセットは無いが、質量行列M(q)の
        [0.0, 0.0, 0.15],  # link6: 条件数が悪化しないよう先端リンクにも十分な慣性を与える)
    ]
)
LINK_MASS = np.array([2.0, 1.5, 1.0, 0.8, 0.6, 0.5])


def _isotropic_inertia() -> np.ndarray:
    """一様棒の横方向慣性モーメント(I=m*l^2/12, l=2*|com|)を等方的に適用した
    3x3対角慣性テンソルのリスト(重心まわり、自リンク座標系内)。"""
    lengths = 2.0 * np.linalg.norm(LINK_COM, axis=-1)
    lengths = np.maximum(lengths, 0.02)
    i_scalar = LINK_MASS * lengths**2 / 12.0
    return np.array([np.eye(3) * i for i in i_scalar])


LINK_INERTIA = _isotropic_inertia()  # shape (6, 3, 3)

C_VISCOUS = np.full(N_JOINTS, 1.0)
C_COULOMB = np.full(N_JOINTS, 0.4)
V_STRIBECK = 0.03


def dh_transform(a: float, alpha: float, d: float, theta: np.ndarray) -> np.ndarray:
    """修正DH変換行列(Craig規約: frame iの座標をframe i-1の座標に変換する4x4同次変換)。

    T = Rot_x(alpha) Trans_x(a) Rot_z(theta) Trans_z(d)。標準DH(Rot_z*Trans_z*Trans_x*Rot_x)
    と異なり、この並び順だと並進オフセット(a,d)がtheta(関節自身の回転)に依存しない
    ため、RNEAの再帰式(offsetは前段フレームの剛体に固定されているという前提)が
    そのまま成立する。

    theta: shape (...,) -> shape (..., 4, 4)
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    ones = np.ones_like(ct)
    zeros = np.zeros_like(ct)
    row0 = np.stack([ct, -st, zeros, a * ones], axis=-1)
    row1 = np.stack([st * ca, ct * ca, -sa * ones, -d * sa * ones], axis=-1)
    row2 = np.stack([st * sa, ct * sa, ca * ones, d * ca * ones], axis=-1)
    row3 = np.stack([zeros, zeros, zeros, ones], axis=-1)
    return np.stack([row0, row1, row2, row3], axis=-2)


def _link_transforms(q: np.ndarray) -> list[np.ndarray]:
    """各関節iのDH変換 T_i (frame iからframe i-1への変換、3x3回転+3並進)を返す。

    Returns: [(R_1,p_1), ..., (R_6,p_6)]  R: (...,3,3), p: (...,3)
    """
    transforms = []
    for i in range(N_JOINTS):
        T = dh_transform(DH_A[i], DH_ALPHA[i], DH_D[i], q[..., i])
        transforms.append((T[..., :3, :3], T[..., :3, 3]))
    return transforms


def rnea(q: np.ndarray, q_dot: np.ndarray, q_ddot: np.ndarray, gravity: float = GRAVITY) -> np.ndarray:
    """再帰的ニュートン・オイラー法(RNEA)で逆動力学 tau(q,q_dot,q_ddot) を計算する。

    重力は基準リンクの並進加速度を-gravityに設定することで組み込む(標準的な
    手法)。摩擦は含まない(呼び出し側で別途加える)。

    q, q_dot, q_ddot: shape (..., 6) -> Returns: shape (..., 6)
    """
    batch_shape = q.shape[:-1]
    transforms = _link_transforms(q)

    z_hat = np.zeros(batch_shape + (3,))
    z_hat[..., 2] = 1.0

    omega = np.zeros(batch_shape + (3,))
    omega_dot = np.zeros(batch_shape + (3,))
    v_dot = np.zeros(batch_shape + (3,))
    v_dot[..., 2] = gravity  # 基準リンクを+gravityで加速させ重力を模擬(標準的な手法)

    omegas, omega_dots, v_dot_coms, forces, torques = [], [], [], [], []

    for i in range(N_JOINTS):
        R, p = transforms[i]  # R: (i-1)R_i, p: (i-1)P_i
        R_T = np.swapaxes(R, -1, -2)  # (i)R_(i-1)

        qd_i = q_dot[..., i : i + 1]
        qdd_i = q_ddot[..., i : i + 1]

        omega_prev_in_i = np.einsum("...ij,...j->...i", R_T, omega)
        omega_i = omega_prev_in_i + qd_i * z_hat

        omega_dot_prev_in_i = np.einsum("...ij,...j->...i", R_T, omega_dot)
        omega_dot_i = (
            omega_dot_prev_in_i
            + np.cross(omega_prev_in_i, qd_i * z_hat)
            + qdd_i * z_hat
        )

        # v̇_i = (i)R_(i-1) ( ω̇_{i-1} × (i-1)P_i + ω_{i-1} × (ω_{i-1} × (i-1)P_i) + v̇_{i-1} )
        v_dot_i = np.einsum(
            "...ij,...j->...i",
            R_T,
            np.cross(omega_dot, p) + np.cross(omega, np.cross(omega, p)) + v_dot,
        )

        com = LINK_COM[i]
        v_dot_com_i = v_dot_i + np.cross(omega_dot_i, com) + np.cross(omega_i, np.cross(omega_i, com))

        omega = omega_i
        omega_dot = omega_dot_i
        v_dot = v_dot_i

        omegas.append(omega_i)
        omega_dots.append(omega_dot_i)
        v_dot_coms.append(v_dot_com_i)

    f_next = np.zeros(batch_shape + (3,))
    n_next = np.zeros(batch_shape + (3,))
    tau = np.zeros(batch_shape + (N_JOINTS,))

    for i in range(N_JOINTS - 1, -1, -1):
        m = LINK_MASS[i]
        I = LINK_INERTIA[i]
        omega_i = omegas[i]
        omega_dot_i = omega_dots[i]
        com = LINK_COM[i]

        F_i = m * v_dot_coms[i]
        I_omega_dot = np.einsum("ij,...j->...i", I, omega_dot_i)
        I_omega = np.einsum("ij,...j->...i", I, omega_i)
        N_i = I_omega_dot + np.cross(omega_i, I_omega)

        if i + 1 < N_JOINTS:
            R_next, p_next = transforms[i + 1]  # (i)R_(i+1), (i)P_(i+1)
            f_next_in_i = np.einsum("...ij,...j->...i", R_next, f_next)
            n_next_in_i = np.einsum("...ij,...j->...i", R_next, n_next)
        else:
            f_next_in_i = np.zeros_like(F_i)
            n_next_in_i = np.zeros_like(N_i)
            p_next = np.zeros(batch_shape + (3,))

        f_i = f_next_in_i + F_i
        n_i = (
            N_i
            + n_next_in_i
            + np.cross(com, F_i)
            + np.cross(p_next, f_next_in_i)
        )

        tau[..., i] = np.einsum("...i,i->...", n_i, np.array([0.0, 0.0, 1.0]))

        f_next = f_i
        n_next = n_i

    return tau


def mass_matrix(q: np.ndarray) -> np.ndarray:
    """質量行列M(q)。RNEAをq_ddot=単位ベクトル・q_dot=0・重力なしでN_JOINTS回
    呼び出し、各列を求める(合成剛体法)。shape (...,6) -> shape (...,6,6)"""
    batch_shape = q.shape[:-1]
    zero = np.zeros(batch_shape + (N_JOINTS,))
    M = np.zeros(batch_shape + (N_JOINTS, N_JOINTS))
    for j in range(N_JOINTS):
        qdd = np.zeros(batch_shape + (N_JOINTS,))
        qdd[..., j] = 1.0
        M[..., :, j] = rnea(q, zero, qdd, gravity=0.0)
    return M


def bias_forces(q: np.ndarray, q_dot: np.ndarray, g: float = GRAVITY) -> np.ndarray:
    """バイアス力 C(q,q_dot)q_dot + G(q)。RNEAをq_ddot=0で1回呼び出す。"""
    zero = np.zeros_like(q)
    return rnea(q, q_dot, zero, gravity=g)


def friction_torque(
    q_dot: np.ndarray, c_viscous: np.ndarray = C_VISCOUS, c_coulomb: np.ndarray = C_COULOMB
) -> np.ndarray:
    """関節摩擦トルク(粘性+クーロン、Stribeck風の滑らかな近似)。"""
    return c_viscous * q_dot + c_coulomb * np.tanh(q_dot / V_STRIBECK)


def dynamics(
    state: np.ndarray,
    tau: np.ndarray,
    c_viscous: np.ndarray = C_VISCOUS,
    c_coulomb: np.ndarray = C_COULOMB,
    g: float = GRAVITY,
) -> np.ndarray:
    """状態の時間微分dx/dtを返す。state: shape (...,12) = [q(6), q_dot(6)]"""
    q = state[..., :N_JOINTS]
    q_dot = state[..., N_JOINTS:]

    M = mass_matrix(q)
    bias = bias_forces(q, q_dot, g)
    friction = friction_torque(q_dot, c_viscous, c_coulomb)

    rhs = tau - bias - friction
    q_ddot = np.linalg.solve(M, rhs[..., None])[..., 0]

    return np.concatenate([q_dot, q_ddot], axis=-1)


def rk4_step(
    state: np.ndarray,
    dt: float,
    tau: np.ndarray,
    c_viscous: np.ndarray = C_VISCOUS,
    c_coulomb: np.ndarray = C_COULOMB,
    g: float = GRAVITY,
) -> np.ndarray:
    k1 = dynamics(state, tau, c_viscous, c_coulomb, g)
    k2 = dynamics(state + 0.5 * dt * k1, tau, c_viscous, c_coulomb, g)
    k3 = dynamics(state + 0.5 * dt * k2, tau, c_viscous, c_coulomb, g)
    k4 = dynamics(state + dt * k3, tau, c_viscous, c_coulomb, g)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(
    initial_state: np.ndarray,
    dt: float,
    n_steps: int,
    tau: np.ndarray | None = None,
    c_viscous: np.ndarray = C_VISCOUS,
    c_coulomb: np.ndarray = C_COULOMB,
    g: float = GRAVITY,
) -> np.ndarray:
    """初期状態からn_steps+1点の軌道を生成する。Returns: shape (n_steps+1, 12)"""
    if tau is None:
        tau_seq = np.zeros((n_steps, N_JOINTS))
    elif tau.ndim == 1:
        tau_seq = np.tile(tau, (n_steps, 1))
    else:
        tau_seq = tau

    traj = [initial_state]
    state = initial_state
    for t in range(n_steps):
        state = rk4_step(state, dt, tau_seq[t], c_viscous, c_coulomb, g)
        traj.append(state)
    return np.stack(traj, axis=0)


def forward_kinematics(q: np.ndarray) -> np.ndarray:
    """各リンク重心のワールド座標での位置を返す(検証・可視化用)。

    Returns: shape (..., 6, 3)(リンク1〜6の重心位置)
    """
    transforms = _link_transforms(q)
    batch_shape = q.shape[:-1]
    R_world = np.broadcast_to(np.eye(3), batch_shape + (3, 3)).copy()
    p_world = np.zeros(batch_shape + (3,))
    coms = []
    for i in range(N_JOINTS):
        R_i, p_i = transforms[i]  # (i-1)R_i, (i-1)P_i
        # frame iの原点のワールド座標 = frame i-1の原点 + (frame i-1の姿勢で回転した(i-1)P_i)
        p_world = p_world + np.einsum("...ij,...j->...i", R_world, p_i)
        R_world = np.einsum("...ij,...jk->...ik", R_world, R_i)
        com_world = p_world + np.einsum("...ij,...j->...i", R_world, LINK_COM[i])
        coms.append(com_world)
    return np.stack(coms, axis=-2)


def potential_energy(state: np.ndarray, g: float = GRAVITY) -> np.ndarray:
    """位置エネルギー(各リンク重心の高さ×質量×重力)。"""
    q = state[..., :N_JOINTS]
    coms_world = forward_kinematics(q)  # (..., 6, 3)
    heights = coms_world[..., 2]  # z成分
    return g * np.einsum("...i,i->...", heights, LINK_MASS)


def total_energy(state: np.ndarray, g: float = GRAVITY) -> np.ndarray:
    """力学的全エネルギー(運動エネルギー+位置エネルギー)。"""
    return kinetic_energy(state) + potential_energy(state, g)


def kinetic_energy(state: np.ndarray) -> np.ndarray:
    """全運動エネルギー 0.5 * q_dot^T M(q) q_dot。"""
    q = state[..., :N_JOINTS]
    q_dot = state[..., N_JOINTS:]
    M = mass_matrix(q)
    Mq = np.einsum("...ij,...j->...i", M, q_dot)
    return 0.5 * np.einsum("...i,...i->...", q_dot, Mq)
