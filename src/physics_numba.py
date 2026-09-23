"""単一状態のRNEA計算をnumbaでJITコンパイルした高速版(推論・制御専用)。

`src/physics.py`のnumpy実装と全く同じ式・同じDHパラメータ/リンク諸元を使う。
学習パラメータを持たない「摩擦なしの参照物理」(質量行列M(q)・バイアス力
bias(q,q_dot))はここでは自動微分が不要なため、numbaでネイティブコードに
コンパイルして高速化する(学習(勾配計算)には`src/model.py`のtorch版を
引き続き使う。numba版は逆伝播をサポートしないため推論・制御専用)。

バッチ処理を前提としたnumpy/torch版と異なり、単一状態を明示的なループ・
スカラー演算で処理する(PTP制御など1軌道をリアルタイムに近い速度で
回す用途を想定)。
"""
import numpy as np
from numba import njit

from src.physics import DH_A, DH_ALPHA, DH_D, GRAVITY, LINK_COM, LINK_INERTIA, LINK_MASS, N_JOINTS


@njit(cache=True)
def _dh_transform_numba(a: float, alpha: float, d: float, theta: float):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    R = np.array(
        [
            [ct, -st, 0.0],
            [st * ca, ct * ca, -sa],
            [st * sa, ct * sa, ca],
        ]
    )
    p = np.array([a, -d * sa, d * ca])
    return R, p


@njit(cache=True)
def rnea_numba(
    q: np.ndarray,
    q_dot: np.ndarray,
    q_ddot: np.ndarray,
    dh_a: np.ndarray,
    dh_alpha: np.ndarray,
    dh_d: np.ndarray,
    link_mass: np.ndarray,
    link_com: np.ndarray,
    link_inertia: np.ndarray,
    gravity: float,
) -> np.ndarray:
    """単一状態のRNEA(physics.rneaと同じ式)。q,q_dot,q_ddot: shape (6,) -> shape (6,)"""
    n = q.shape[0]
    z_hat = np.array([0.0, 0.0, 1.0])

    Rs = np.zeros((n, 3, 3))
    ps = np.zeros((n, 3))
    omegas = np.zeros((n, 3))
    omega_dots = np.zeros((n, 3))
    v_dot_coms = np.zeros((n, 3))

    omega = np.zeros(3)
    omega_dot = np.zeros(3)
    v_dot = np.array([0.0, 0.0, gravity])

    for i in range(n):
        R, p = _dh_transform_numba(dh_a[i], dh_alpha[i], dh_d[i], q[i])
        Rs[i] = R
        ps[i] = p
        R_T = R.T

        qd_i = q_dot[i]
        qdd_i = q_ddot[i]

        omega_prev_in_i = R_T @ omega
        omega_i = omega_prev_in_i + qd_i * z_hat

        omega_dot_prev_in_i = R_T @ omega_dot
        omega_dot_i = omega_dot_prev_in_i + np.cross(omega_prev_in_i, qd_i * z_hat) + qdd_i * z_hat

        v_dot_i = R_T @ (np.cross(omega_dot, p) + np.cross(omega, np.cross(omega, p)) + v_dot)

        com = link_com[i]
        v_dot_com_i = v_dot_i + np.cross(omega_dot_i, com) + np.cross(omega_i, np.cross(omega_i, com))

        omega = omega_i
        omega_dot = omega_dot_i
        v_dot = v_dot_i

        omegas[i] = omega_i
        omega_dots[i] = omega_dot_i
        v_dot_coms[i] = v_dot_com_i

    f_next = np.zeros(3)
    n_next = np.zeros(3)
    tau = np.zeros(n)

    for i in range(n - 1, -1, -1):
        m = link_mass[i]
        I = link_inertia[i]
        omega_i = omegas[i]
        omega_dot_i = omega_dots[i]
        com = link_com[i]

        F_i = m * v_dot_coms[i]
        I_omega_dot = I @ omega_dot_i
        I_omega = I @ omega_i
        N_i = I_omega_dot + np.cross(omega_i, I_omega)

        if i + 1 < n:
            R_next = Rs[i + 1]
            p_next = ps[i + 1]
            f_next_in_i = R_next @ f_next
            n_next_in_i = R_next @ n_next
        else:
            f_next_in_i = np.zeros(3)
            n_next_in_i = np.zeros(3)
            p_next = np.zeros(3)

        f_i = f_next_in_i + F_i
        n_i = N_i + n_next_in_i + np.cross(com, F_i) + np.cross(p_next, f_next_in_i)

        tau[i] = n_i[2]

        f_next = f_i
        n_next = n_i

    return tau


@njit(cache=True)
def mass_matrix_numba(
    q: np.ndarray,
    dh_a: np.ndarray,
    dh_alpha: np.ndarray,
    dh_d: np.ndarray,
    link_mass: np.ndarray,
    link_com: np.ndarray,
    link_inertia: np.ndarray,
) -> np.ndarray:
    """質量行列M(q)。RNEAをq_ddot=単位ベクトルごとにN_JOINTS回呼ぶ合成剛体法。"""
    n = q.shape[0]
    zero = np.zeros(n)
    M = np.zeros((n, n))
    for j in range(n):
        qdd = np.zeros(n)
        qdd[j] = 1.0
        M[:, j] = rnea_numba(q, zero, qdd, dh_a, dh_alpha, dh_d, link_mass, link_com, link_inertia, 0.0)
    return M


@njit(cache=True)
def bias_forces_numba(
    q: np.ndarray,
    q_dot: np.ndarray,
    dh_a: np.ndarray,
    dh_alpha: np.ndarray,
    dh_d: np.ndarray,
    link_mass: np.ndarray,
    link_com: np.ndarray,
    link_inertia: np.ndarray,
    gravity: float,
) -> np.ndarray:
    """バイアス力C(q,q_dot)q_dot+G(q)。RNEAをq_ddot=0で1回呼ぶ。"""
    zero = np.zeros(q.shape[0])
    return rnea_numba(q, q_dot, zero, dh_a, dh_alpha, dh_d, link_mass, link_com, link_inertia, gravity)


@njit(cache=True)
def dynamics_numba(
    state: np.ndarray,
    tau: np.ndarray,
    dh_a: np.ndarray,
    dh_alpha: np.ndarray,
    dh_d: np.ndarray,
    link_mass: np.ndarray,
    link_com: np.ndarray,
    link_inertia: np.ndarray,
    gravity: float,
    c_viscous: np.ndarray,
    c_coulomb: np.ndarray,
    v_stribeck: float,
) -> np.ndarray:
    n = c_viscous.shape[0]
    q = state[:n]
    q_dot = state[n:]

    M = mass_matrix_numba(q, dh_a, dh_alpha, dh_d, link_mass, link_com, link_inertia)
    bias = bias_forces_numba(q, q_dot, dh_a, dh_alpha, dh_d, link_mass, link_com, link_inertia, gravity)
    friction = c_viscous * q_dot + c_coulomb * np.tanh(q_dot / v_stribeck)

    rhs = tau - bias - friction
    q_ddot = np.linalg.solve(M, rhs)

    out = np.zeros(2 * n)
    out[:n] = q_dot
    out[n:] = q_ddot
    return out


@njit(cache=True)
def rk4_step_numba(
    state: np.ndarray,
    dt: float,
    tau: np.ndarray,
    dh_a: np.ndarray,
    dh_alpha: np.ndarray,
    dh_d: np.ndarray,
    link_mass: np.ndarray,
    link_com: np.ndarray,
    link_inertia: np.ndarray,
    gravity: float,
    c_viscous: np.ndarray,
    c_coulomb: np.ndarray,
    v_stribeck: float,
) -> np.ndarray:
    k1 = dynamics_numba(state, tau, dh_a, dh_alpha, dh_d, link_mass, link_com, link_inertia, gravity, c_viscous, c_coulomb, v_stribeck)
    k2 = dynamics_numba(state + 0.5 * dt * k1, tau, dh_a, dh_alpha, dh_d, link_mass, link_com, link_inertia, gravity, c_viscous, c_coulomb, v_stribeck)
    k3 = dynamics_numba(state + 0.5 * dt * k2, tau, dh_a, dh_alpha, dh_d, link_mass, link_com, link_inertia, gravity, c_viscous, c_coulomb, v_stribeck)
    k4 = dynamics_numba(state + dt * k3, tau, dh_a, dh_alpha, dh_d, link_mass, link_com, link_inertia, gravity, c_viscous, c_coulomb, v_stribeck)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@njit(cache=True)
def rollout_numba(
    initial_state: np.ndarray,
    dt: float,
    tau_seq: np.ndarray,
    dh_a: np.ndarray,
    dh_alpha: np.ndarray,
    dh_d: np.ndarray,
    link_mass: np.ndarray,
    link_com: np.ndarray,
    link_inertia: np.ndarray,
    gravity: float,
    c_viscous: np.ndarray,
    c_coulomb: np.ndarray,
    v_stribeck: float,
) -> np.ndarray:
    """単一軌道をRK4でロールアウトする(学習済み摩擦係数込み、推論・制御専用)。

    tau_seq: shape (n_steps, 6) -> Returns: shape (n_steps+1, 12)
    """
    n_steps = tau_seq.shape[0]
    n = initial_state.shape[0]
    traj = np.zeros((n_steps + 1, n))
    traj[0] = initial_state
    state = initial_state.copy()
    for t in range(n_steps):
        state = rk4_step_numba(
            state, dt, tau_seq[t], dh_a, dh_alpha, dh_d, link_mass, link_com, link_inertia,
            gravity, c_viscous, c_coulomb, v_stribeck,
        )
        traj[t + 1] = state
    return traj


@njit(cache=True)
def run_ptp_closed_loop_numba(
    initial_state: np.ndarray,
    target: np.ndarray,
    n_steps: int,
    dt: float,
    dh_a: np.ndarray,
    dh_alpha: np.ndarray,
    dh_d: np.ndarray,
    link_mass: np.ndarray,
    link_com: np.ndarray,
    link_inertia: np.ndarray,
    gravity: float,
    c_viscous: np.ndarray,
    c_coulomb: np.ndarray,
    v_stribeck: float,
    kp: float,
    ki: float,
    kd: float,
    tau_max: np.ndarray,
    integral_clip: float,
) -> np.ndarray:
    """計算トルク法PTP制御の閉ループ全体(コントローラ+プラント)を1回のJIT
    呼び出しにまとめる(#26: ステップごとのPython呼び出しオーバーヘッドを
    完全に排除し、単一軌道のリアルタイム制御を最大限高速化する)。

    `c_viscous`/`c_coulomb`に真値(physics.C_VISCOUS/C_COULOMB)を渡せば
    真値モデルの閉ループシミュレーション、学習済みサロゲートの係数を渡せば
    サロゲートモデルの閉ループシミュレーションになる(コントローラ自身の
    質量行列・バイアス力は既知の物理として両者で共通)。

    Returns: traj shape (n_steps+1, 12)
    """
    n = c_viscous.shape[0]
    traj = np.zeros((n_steps + 1, 2 * n))
    traj[0] = initial_state
    state = initial_state.copy()
    integral = np.zeros(n)

    for t in range(n_steps):
        q = state[:n]
        q_dot = state[n:]
        e = np.arctan2(np.sin(q - target), np.cos(q - target))
        integral = integral + e * dt
        for i in range(n):
            if integral[i] > integral_clip:
                integral[i] = integral_clip
            elif integral[i] < -integral_clip:
                integral[i] = -integral_clip

        q_ddot_desired = -kp * e - ki * integral - kd * q_dot

        M = mass_matrix_numba(q, dh_a, dh_alpha, dh_d, link_mass, link_com, link_inertia)
        bias = bias_forces_numba(q, q_dot, dh_a, dh_alpha, dh_d, link_mass, link_com, link_inertia, gravity)
        tau = M @ q_ddot_desired + bias
        for i in range(n):
            if tau[i] > tau_max[i]:
                tau[i] = tau_max[i]
            elif tau[i] < -tau_max[i]:
                tau[i] = -tau_max[i]

        state = rk4_step_numba(
            state, dt, tau, dh_a, dh_alpha, dh_d, link_mass, link_com, link_inertia,
            gravity, c_viscous, c_coulomb, v_stribeck,
        )
        traj[t + 1] = state

    return traj


def mass_matrix(q: np.ndarray) -> np.ndarray:
    """physics.mass_matrixと同じシグネチャの単一状態版(呼び出し側でDH/リンク
    諸元を渡す手間を省くラッパー)。"""
    return mass_matrix_numba(q, DH_A, DH_ALPHA, DH_D, LINK_MASS, LINK_COM, LINK_INERTIA)


def bias_forces(q: np.ndarray, q_dot: np.ndarray, g: float = GRAVITY) -> np.ndarray:
    """physics.bias_forcesと同じシグネチャの単一状態版。"""
    return bias_forces_numba(q, q_dot, DH_A, DH_ALPHA, DH_D, LINK_MASS, LINK_COM, LINK_INERTIA, g)


def run_ptp_closed_loop(
    initial_state: np.ndarray,
    target: np.ndarray,
    n_steps: int,
    dt: float,
    c_viscous: np.ndarray,
    c_coulomb: np.ndarray,
    kp: float,
    ki: float,
    kd: float,
    tau_max: np.ndarray,
    integral_clip: float = 4.0,
    v_stribeck: float = 0.03,
    g: float = GRAVITY,
) -> np.ndarray:
    """`run_ptp_closed_loop_numba`のDH/リンク諸元を省略できるラッパー。"""
    return run_ptp_closed_loop_numba(
        initial_state, target, n_steps, dt,
        DH_A, DH_ALPHA, DH_D, LINK_MASS, LINK_COM, LINK_INERTIA, g,
        c_viscous, c_coulomb, v_stribeck,
        kp, ki, kd, tau_max, integral_clip,
    )


def run_ptp_surrogate_from_model(
    model,
    initial_state: np.ndarray,
    target: np.ndarray,
    n_steps: int,
    dt: float,
    kp: float,
    ki: float,
    kd: float,
    tau_max: np.ndarray,
    integral_clip: float = 4.0,
) -> np.ndarray:
    """学習済み`StructuredFrictionGrayBoxModel`から摩擦係数(定数)を取り出し、
    numbaでコンパイルした閉ループPTP制御を実行する(#26)。

    摩擦係数以外(質量行列・バイアス力=既知の物理)は真値と共通のため、
    学習パラメータである摩擦係数だけをtorchモデルから抽出すればよい。
    """
    import torch

    with torch.no_grad():
        c_viscous = torch.exp(model.log_c_viscous).cpu().numpy()
        c_coulomb = torch.exp(model.log_c_coulomb).cpu().numpy()
    return run_ptp_closed_loop(
        initial_state, target, n_steps, dt, c_viscous, c_coulomb, kp, ki, kd, tau_max, integral_clip, model.v_stribeck
    )
