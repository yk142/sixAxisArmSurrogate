"""自己回帰ロールアウトと誤差指標の計算。"""
import numpy as np

from src.physics import N_JOINTS, simulate_batch


def true_rollout(initial_states: np.ndarray, dt: float, n_steps: int, tau_seq: np.ndarray | None = None) -> np.ndarray:
    """shape (n_ic, n_steps+1, 12)

    tau_seq: shape (n_steps, 6)。全ICに共通のトルク列として適用する。
    """
    n_ic = initial_states.shape[0]
    if tau_seq is None:
        tau_seqs = np.zeros((n_ic, n_steps, N_JOINTS))
    else:
        tau_seqs = np.tile(tau_seq[None, :, :], (n_ic, 1, 1))
    return simulate_batch(initial_states, dt, n_steps, tau_seqs)


def rmse_curve(true_traj: np.ndarray, pred_traj: np.ndarray) -> np.ndarray:
    """horizon stepごとのRMSEを返す(角度6次元はwrap済み差、角速度6次元は単純差)。

    true_traj, pred_traj: shape (n_steps+1, n_ic, 12)

    Returns: shape (n_steps+1,)
    """
    angle_err = np.arctan2(
        np.sin(true_traj[..., :N_JOINTS] - pred_traj[..., :N_JOINTS]),
        np.cos(true_traj[..., :N_JOINTS] - pred_traj[..., :N_JOINTS]),
    )
    vel_err = true_traj[..., N_JOINTS:] - pred_traj[..., N_JOINTS:]
    sq_err = (angle_err**2).sum(axis=-1) + (vel_err**2).sum(axis=-1)
    return np.sqrt(sq_err.mean(axis=1))
