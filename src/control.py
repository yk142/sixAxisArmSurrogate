"""6軸アーム用の重力補償込み計算トルク法PTP制御。

    tau = M(q) @ (-Kp*e - Ki*integral(e) - Kd*q_dot) + bias(q,q_dot)

bias(q,q_dot) = C(q,q_dot)q_dot + G(q) はRNEA1回で計算する(physics.bias_forces)。
"""
import numpy as np

from src.physics import N_JOINTS, bias_forces, mass_matrix, rk4_step

# scaraSurrogateの最終値(KP=30,KI=3.5,KD=22)を出発点にしたところ、腕関節の
# トルク要求(重力補償だけで~25Nm)がscaraSurrogateの系より大きく、飽和して
# 減結合が崩れ収束が非常に遅くなることを確認した。ゲインを上げて収束を速め、
# トルク上限も腕・手首それぞれの慣性スケールに応じて実測(飽和診断)から
# 十分な余裕を持たせた値にする。
KP = 150.0
KI = 15.0
KD = 24.0
TAU_MAX = np.array([70.0, 60.0, 25.0, 5.0, 5.0, 1.0])
INTEGRAL_CLIP = 4.0


def angular_error(q: np.ndarray, target: np.ndarray) -> np.ndarray:
    """q と target の符号付き誤差(各関節、wrap済みで(-pi,pi]に収まる)。"""
    return np.arctan2(np.sin(q - target), np.cos(q - target))


class PTPController:
    """重力補償込みの計算トルク法によるPTPコントローラ(積分項つき、6自由度)。"""

    def __init__(self, kp: float = KP, ki: float = KI, kd: float = KD, tau_max: np.ndarray = TAU_MAX):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.tau_max = tau_max
        self.integral = np.zeros(N_JOINTS)

    def reset(self) -> None:
        self.integral = np.zeros(N_JOINTS)

    def compute(self, state: np.ndarray, target: np.ndarray, dt: float) -> np.ndarray:
        q, q_dot = state[:N_JOINTS], state[N_JOINTS:]
        e = angular_error(q, target)
        self.integral = np.clip(self.integral + e * dt, -INTEGRAL_CLIP, INTEGRAL_CLIP)

        q_ddot_desired = -self.kp * e - self.ki * self.integral - self.kd * q_dot

        M = mass_matrix(q[None, :])[0]
        bias = bias_forces(q[None, :], q_dot[None, :])[0]
        tau = M @ q_ddot_desired + bias
        return np.clip(tau, -self.tau_max, self.tau_max)

    def reset_batch(self, batch_size: int) -> None:
        self.integral = np.zeros((batch_size, N_JOINTS))

    def compute_batch(self, state: np.ndarray, target: np.ndarray, dt: float) -> np.ndarray:
        """`compute`のバッチ版。state: shape (batch,12), target: shape (batch,6) -> tau: shape (batch,6)

        physics.mass_matrix/bias_forcesが元々バッチ次元(`...`)に対応している
        ことを利用し、多数のシナリオを1回の呼び出しでまとめて計算する
        (#12: 数十シナリオのPTP検証を高速化するため)。
        """
        q, q_dot = state[..., :N_JOINTS], state[..., N_JOINTS:]
        e = angular_error(q, target)
        self.integral = np.clip(self.integral + e * dt, -INTEGRAL_CLIP, INTEGRAL_CLIP)

        q_ddot_desired = -self.kp * e - self.ki * self.integral - self.kd * q_dot

        M = mass_matrix(q)
        bias = bias_forces(q, q_dot)
        tau = np.einsum("...ij,...j->...i", M, q_ddot_desired) + bias
        return np.clip(tau, -self.tau_max, self.tau_max)


def run_ptp_true(
    initial_state: np.ndarray, target: np.ndarray, n_steps: int, dt: float, controller: PTPController | None = None
) -> np.ndarray:
    """真の物理モデルを閉ループでPTP制御する。Returns: traj shape (n_steps+1, 12)"""
    controller = controller or PTPController()
    controller.reset()
    state = initial_state.copy()
    traj = [state.copy()]
    for _ in range(n_steps):
        tau = controller.compute(state, target, dt)
        state = rk4_step(state, dt, tau)
        traj.append(state.copy())
    return np.stack(traj, axis=0)


def run_ptp_true_batch(
    initial_states: np.ndarray,
    targets: np.ndarray,
    n_steps: int,
    dt: float,
    controller: PTPController | None = None,
) -> np.ndarray:
    """真の物理モデルで多数のシナリオを同時に閉ループPTP制御する(#12)。

    initial_states: shape (batch,12), targets: shape (batch,6)
    Returns: traj shape (n_steps+1, batch, 12)
    """
    controller = controller or PTPController()
    batch = initial_states.shape[0]
    controller.reset_batch(batch)
    state = initial_states.copy()
    traj = [state.copy()]
    for _ in range(n_steps):
        tau = controller.compute_batch(state, targets, dt)
        state = rk4_step(state, dt, tau)
        traj.append(state.copy())
    return np.stack(traj, axis=0)


def run_ptp_surrogate_batch(
    model,
    initial_states: np.ndarray,
    targets: np.ndarray,
    n_steps: int,
    dt: float,
    controller: PTPController | None = None,
) -> np.ndarray:
    """NSSサロゲートモデルで多数のシナリオを同時に閉ループPTP制御する(#12)。

    model.step(state, u)はtorch側で"..."バッチ次元に対応しているため、
    そのまま複数シナリオをまとめて処理できる。

    initial_states: shape (batch,12), targets: shape (batch,6)
    Returns: traj shape (n_steps+1, batch, 12)
    """
    import torch

    controller = controller or PTPController()
    batch = initial_states.shape[0]
    controller.reset_batch(batch)
    device = next(model.parameters()).device
    state_t = torch.as_tensor(initial_states, dtype=torch.float32, device=device)
    state = initial_states.copy()
    traj = [state_t]
    for _ in range(n_steps):
        tau = controller.compute_batch(state, targets, dt)
        tau_t = torch.as_tensor(tau, dtype=torch.float32, device=device)
        with torch.no_grad():
            state_t = model.step(state_t, tau_t)
        state = state_t.cpu().numpy()
        traj.append(state_t)
    return torch.stack(traj, dim=0).cpu().numpy()


def run_ptp_surrogate(
    model,
    initial_state: np.ndarray,
    target: np.ndarray,
    n_steps: int,
    dt: float,
    controller: PTPController | None = None,
) -> np.ndarray:
    """NSSサロゲートモデル(model.rollout互換)を閉ループでPTP制御する。Returns: traj shape (n_steps+1, 12)"""
    controller = controller or PTPController()
    controller.reset()
    state = initial_state.copy()
    traj = [state.copy()]
    for _ in range(n_steps):
        tau = controller.compute(state, target, dt)
        state = model.rollout(state, 1, tau_seq=tau[None, :])[-1]
        traj.append(state.copy())
    return np.stack(traj, axis=0)
