"""6軸アーム用のNSS(Neural State Space)サロゲートモデル。

grayboxモデルは、質量行列M(q)・バイアス力(コリオリ+重力)を「既知の物理」
として扱う(RNEAをtorchで微分可能な形に移植したもの。`src/physics.py`の
numpy実装と全く同じ式・同じ変数名で書いており、DHパラメータ・リンク質量/
重心/慣性テンソルも同じ値を定数として使う)。学習対象は摩擦係数のみ
(粘性+クーロン摩擦の関数形をハードコードし、係数(関節ごと6個)だけを
学習パラメータにする、姉妹プロジェクトsc araSurrogateのPhase2-M8/M12/M13/M14
の教訓を最初から適用)。

状態は12次元 x=[q(6), q_dot(6)]。周期角度6個を含むため、
(sin q_i, cos q_i)×6 + q_dot(6) の18次元にエンコードしてからネットワークに
渡す(ブラックボックス版でのみ使用。グレーボックス版は物理量をそのまま使う)。
入力は6次元 u=[tau1,...,tau6]。
"""
import numpy as np
import torch
import torch.nn as nn

from src.physics import (
    DH_A,
    DH_ALPHA,
    DH_D,
    GRAVITY,
    LINK_COM,
    LINK_INERTIA,
    LINK_MASS,
    N_JOINTS,
    V_STRIBECK,
)

STATE_DIM = 2 * N_JOINTS  # 12
STATE_ENC_DIM = 3 * N_JOINTS  # 18: (sin,cos)x6 + q_dot x6
CONTROL_DIM = N_JOINTS  # 6
DT = 0.002


def encode_state(state: torch.Tensor) -> torch.Tensor:
    """(..., 12) [q(6),q_dot(6)] -> (..., 18) [sin,cos の6組, q_dot(6)]"""
    q = state[..., :N_JOINTS]
    q_dot = state[..., N_JOINTS:]
    return torch.cat([torch.sin(q), torch.cos(q), q_dot], dim=-1)


def decode_state(enc: torch.Tensor) -> torch.Tensor:
    """(..., 18) -> (..., 12) [q(6),q_dot(6)]"""
    sin_q = enc[..., :N_JOINTS]
    cos_q = enc[..., N_JOINTS : 2 * N_JOINTS]
    q_dot = enc[..., 2 * N_JOINTS :]
    q = torch.atan2(sin_q, cos_q)
    return torch.cat([q, q_dot], dim=-1)


def _make_mlp(in_dim: int, out_dim: int, hidden_dim: int, n_hidden_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
    for _ in range(n_hidden_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


def _dh_transform_torch(a: float, alpha: float, d: float, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """修正DH変換(physics.dh_transformのtorch版)。回転行列Rと並進pのみ返す
    (RNEAでは4x4同次変換ではなくR,pの組で十分なため)。

    theta: shape (...,) -> R: shape (..., 3, 3), p: shape (..., 3)
    """
    ct, st = torch.cos(theta), torch.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    ones = torch.ones_like(ct)
    zeros = torch.zeros_like(ct)
    row0 = torch.stack([ct, -st, zeros], dim=-1)
    row1 = torch.stack([st * ca, ct * ca, -sa * ones], dim=-1)
    row2 = torch.stack([st * sa, ct * sa, ca * ones], dim=-1)
    R = torch.stack([row0, row1, row2], dim=-2)
    p = torch.stack([a * ones, -d * sa * ones, d * ca * ones], dim=-1)
    return R, p


def _link_transforms_torch(q: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """各関節iのDH変換(R,p)のリスト(physics._link_transformsのtorch版)。"""
    return [
        _dh_transform_torch(float(DH_A[i]), float(DH_ALPHA[i]), float(DH_D[i]), q[..., i])
        for i in range(N_JOINTS)
    ]


def rnea_torch(q: torch.Tensor, q_dot: torch.Tensor, q_ddot: torch.Tensor, gravity: float = GRAVITY) -> torch.Tensor:
    """RNEAのtorch版(physics.rneaと全く同じ式)。q,q_dot,q_ddot: shape (...,6) -> shape (...,6)"""
    device, dtype = q.device, q.dtype
    batch_shape = q.shape[:-1]
    transforms = _link_transforms_torch(q)

    link_com = torch.as_tensor(LINK_COM, device=device, dtype=dtype)
    link_mass = torch.as_tensor(LINK_MASS, device=device, dtype=dtype)
    link_inertia = torch.as_tensor(LINK_INERTIA, device=device, dtype=dtype)

    z_hat = torch.zeros(*batch_shape, 3, device=device, dtype=dtype)
    z_hat[..., 2] = 1.0

    omega = torch.zeros(*batch_shape, 3, device=device, dtype=dtype)
    omega_dot = torch.zeros(*batch_shape, 3, device=device, dtype=dtype)
    v_dot = torch.zeros(*batch_shape, 3, device=device, dtype=dtype)
    v_dot = v_dot.clone()
    v_dot[..., 2] = gravity

    omegas, omega_dots, v_dot_coms = [], [], []

    for i in range(N_JOINTS):
        R, p = transforms[i]
        R_T = R.transpose(-1, -2)

        qd_i = q_dot[..., i : i + 1]
        qdd_i = q_ddot[..., i : i + 1]

        omega_prev_in_i = torch.einsum("...ij,...j->...i", R_T, omega)
        omega_i = omega_prev_in_i + qd_i * z_hat

        omega_dot_prev_in_i = torch.einsum("...ij,...j->...i", R_T, omega_dot)
        omega_dot_i = (
            omega_dot_prev_in_i
            + torch.cross(omega_prev_in_i, qd_i * z_hat, dim=-1)
            + qdd_i * z_hat
        )

        v_dot_i = torch.einsum(
            "...ij,...j->...i",
            R_T,
            torch.cross(omega_dot, p, dim=-1) + torch.cross(omega, torch.cross(omega, p, dim=-1), dim=-1) + v_dot,
        )

        com = link_com[i].expand_as(omega_dot_i)
        v_dot_com_i = v_dot_i + torch.cross(omega_dot_i, com, dim=-1) + torch.cross(
            omega_i, torch.cross(omega_i, com, dim=-1), dim=-1
        )

        omega = omega_i
        omega_dot = omega_dot_i
        v_dot = v_dot_i

        omegas.append(omega_i)
        omega_dots.append(omega_dot_i)
        v_dot_coms.append(v_dot_com_i)

    f_next = torch.zeros(*batch_shape, 3, device=device, dtype=dtype)
    n_next = torch.zeros(*batch_shape, 3, device=device, dtype=dtype)
    tau_list = [None] * N_JOINTS

    for i in range(N_JOINTS - 1, -1, -1):
        m = link_mass[i]
        I = link_inertia[i]
        omega_i = omegas[i]
        omega_dot_i = omega_dots[i]
        com = link_com[i]

        F_i = m * v_dot_coms[i]
        I_omega_dot = torch.einsum("ij,...j->...i", I, omega_dot_i)
        I_omega = torch.einsum("ij,...j->...i", I, omega_i)
        N_i = I_omega_dot + torch.cross(omega_i, I_omega, dim=-1)

        if i + 1 < N_JOINTS:
            R_next, p_next = transforms[i + 1]
            f_next_in_i = torch.einsum("...ij,...j->...i", R_next, f_next)
            n_next_in_i = torch.einsum("...ij,...j->...i", R_next, n_next)
        else:
            f_next_in_i = torch.zeros_like(F_i)
            n_next_in_i = torch.zeros_like(N_i)
            p_next = torch.zeros(*batch_shape, 3, device=device, dtype=dtype)

        f_i = f_next_in_i + F_i
        n_i = (
            N_i
            + n_next_in_i
            + torch.cross(com.expand_as(F_i), F_i, dim=-1)
            + torch.cross(p_next, f_next_in_i, dim=-1)
        )

        tau_list[i] = n_i[..., 2]

        f_next = f_i
        n_next = n_i

    return torch.stack(tau_list, dim=-1)


def mass_matrix_torch(q: torch.Tensor) -> torch.Tensor:
    """質量行列M(q)のtorch版(physics.mass_matrixと同じくRNEAをN_JOINTS回呼ぶ)。
    -> shape (...,6,6)"""
    batch_shape = q.shape[:-1]
    # N_JOINTS本のrnea呼び出しをPythonループで直列に行うと学習時に大きな
    # オーバーヘッドになるため、6本の単位ベクトルをすべて1回のバッチ次元に
    # 詰め込み、rnea_torchを1回だけ呼ぶ(rnea_torchの内部処理は"..."で任意の
    # バッチ次元に対応しているため、そのまま利用できる)。
    eye = torch.eye(N_JOINTS, device=q.device, dtype=q.dtype)
    extra_dims = (1,) * len(batch_shape)
    q_rep = q.unsqueeze(0).expand(N_JOINTS, *batch_shape, N_JOINTS)
    zero_rep = torch.zeros_like(q_rep)
    qdd_rep = eye.reshape(N_JOINTS, *extra_dims, N_JOINTS).expand(N_JOINTS, *batch_shape, N_JOINTS)
    tau_rep = rnea_torch(q_rep, zero_rep, qdd_rep, gravity=0.0)  # (6, *batch_shape, 6)
    return tau_rep.movedim(0, -1)  # M[..., i, j] = tau_rep[j, ..., i]


def bias_forces_torch(q: torch.Tensor, q_dot: torch.Tensor, g: float = GRAVITY) -> torch.Tensor:
    """バイアス力C(q,q_dot)q_dot+G(q)のtorch版。"""
    zero = torch.zeros_like(q)
    return rnea_torch(q, q_dot, zero, gravity=g)


class AutoregressiveModel(nn.Module):
    """自己回帰ロールアウト基底クラス(CONTROL_DIM=6)。"""

    def step(self, state: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _default_u(state: torch.Tensor, u: torch.Tensor | None) -> torch.Tensor:
        if u is not None:
            return u
        return torch.zeros(*state.shape[:-1], CONTROL_DIM, device=state.device)

    @torch.no_grad()
    def rollout(self, initial_state: np.ndarray, n_steps: int, tau_seq: np.ndarray | None = None) -> np.ndarray:
        device = next(self.parameters()).device
        state = torch.as_tensor(initial_state, dtype=torch.float32, device=device)
        traj = [state]
        for t in range(n_steps):
            u = None if tau_seq is None else torch.as_tensor(
                tau_seq[t], dtype=torch.float32, device=device
            ).expand(*state.shape[:-1], CONTROL_DIM)
            state = self.step(state, u)
            traj.append(state)
        return torch.stack(traj, dim=0).cpu().numpy()

    def rollout_diff(self, state: torch.Tensor, tau_seq: torch.Tensor, n_steps: int) -> torch.Tensor:
        traj = [state]
        for t in range(n_steps):
            state = self.step(state, tau_seq[t])
            traj.append(state)
        return torch.stack(traj, dim=0)


class NSSModel(AutoregressiveModel):
    """ブラックボックス版: 状態遷移そのものをMLPで学習する(比較用ベースライン)。"""

    def __init__(self, hidden_dim: int = 128, n_hidden_layers: int = 3):
        super().__init__()
        self.net = _make_mlp(STATE_ENC_DIM + CONTROL_DIM, STATE_ENC_DIM, hidden_dim, n_hidden_layers)

    def forward(self, state: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
        enc = encode_state(state)
        net_in = torch.cat([enc, self._default_u(state, u)], dim=-1)
        return self.net(net_in)

    def step(self, state: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
        return decode_state(self.forward(state, u))


class StructuredFrictionGrayBoxModel(AutoregressiveModel):
    """グレーボックス版: M(q)・バイアス力(コリオリ+重力)はRNEA(既知の物理)で
    計算し、摩擦の関数形(粘性+クーロン、Stribeck風の滑らかな近似)もハード
    コードして、係数(関節ごと6個ずつ)だけを学習パラメータにする。
    """

    def __init__(self, dt: float = DT, g: float = GRAVITY, v_stribeck: float = V_STRIBECK):
        super().__init__()
        self.log_c_viscous = nn.Parameter(torch.zeros(N_JOINTS))
        self.log_c_coulomb = nn.Parameter(torch.zeros(N_JOINTS))
        self.dt = dt
        self.g = g
        self.v_stribeck = v_stribeck

    def residual_torque(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        q_dot = state[..., N_JOINTS:]
        c_viscous = torch.exp(self.log_c_viscous)
        c_coulomb = torch.exp(self.log_c_coulomb)
        return c_viscous * q_dot + c_coulomb * torch.tanh(q_dot / self.v_stribeck)

    def dynamics(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        q, q_dot = state[..., :N_JOINTS], state[..., N_JOINTS:]

        M = mass_matrix_torch(q)
        bias = bias_forces_torch(q, q_dot, self.g)
        friction = self.residual_torque(state, u)

        rhs = (u - bias - friction).unsqueeze(-1)
        q_ddot = torch.linalg.solve(M, rhs).squeeze(-1)
        return torch.cat([q_dot, q_ddot], dim=-1)

    def step(self, state: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
        u = self._default_u(state, u)
        dt = self.dt
        k1 = self.dynamics(state, u)
        k2 = self.dynamics(state + 0.5 * dt * k1, u)
        k3 = self.dynamics(state + 0.5 * dt * k2, u)
        k4 = self.dynamics(state + dt * k3, u)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
