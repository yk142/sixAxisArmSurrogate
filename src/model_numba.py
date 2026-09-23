"""#28: 軽量NN(LightweightGrayBoxModel、#19)の単一軌道向け推論をnumbaで
高速化する。

`src/model.py`のLightweightGrayBoxModelは質量行列M(q)・バイアス力
bias(q,q_dot)を小さなMLP(隠れ層あり、本当のNN)で近似する。#26のRNEA版と
異なり、ここで高速化するのは本物のニューラルネットワークのforward pass。
MLPのforward passは単純な行列積+tanh活性化のみなので、RNEAより単純に
numbaでJITコンパイルできる。

学習済みモデルの重み・バイアス(`nn.Linear`の`.weight`/`.bias`)をnumpy配列
として取り出し、以降は推論・制御専用のnumba実装で完結させる(numbaは
自動微分をサポートしないため、学習には引き続きtorch版を使う)。
"""
import numpy as np
import torch
from numba import njit
from numba.typed import List

from src.model import N_JOINTS, V_STRIBECK
from src.physics_numba import bias_forces_numba, mass_matrix_numba
from src.physics import DH_A, DH_ALPHA, DH_D, GRAVITY, LINK_COM, LINK_INERTIA, LINK_MASS


@njit(cache=True)
def _softplus(x: float) -> float:
    # log(1+exp(x))の数値的に安定な計算(#19のLightweightGrayBoxModelの
    # softplus(diag)+1e-3と同じ)。
    if x > 0.0:
        return x + np.log1p(np.exp(-x))
    return np.log1p(np.exp(x))


@njit(cache=True)
def mlp_forward_numba(x: np.ndarray, weights: list, biases: list) -> np.ndarray:
    """torch.nn.Sequential([Linear,Tanh,...,Linear])と同じforward pass。

    weights[i]: shape (out_i, in_i)(nn.Linear.weightと同じ形状)
    最終層以外にtanhを適用する(_make_mlpと同じ構成)。
    """
    h = x
    n_layers = len(weights)
    for i in range(n_layers):
        h = weights[i] @ h + biases[i]
        if i < n_layers - 1:
            h = np.tanh(h)
    return h


@njit(cache=True)
def _build_L_numba(raw: np.ndarray, n: int) -> np.ndarray:
    """コレスキー分解Lの下三角要素(torch.tril_indicesと同じ順序)を埋める。
    対角はsoftplus+1e-3で正にする(LightweightGrayBoxModel.mass_matrixと同じ)。
    """
    L = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                L[i, j] = _softplus(raw[idx]) + 1e-3
            else:
                L[i, j] = raw[idx]
            idx += 1
    return L


@njit(cache=True)
def mass_matrix_nn_numba(q: np.ndarray, mass_weights: list, mass_biases: list) -> np.ndarray:
    n = q.shape[0]
    enc_q = np.concatenate((np.sin(q), np.cos(q)))
    raw = mlp_forward_numba(enc_q, mass_weights, mass_biases)
    L = _build_L_numba(raw, n)
    return L @ L.T


@njit(cache=True)
def bias_forces_nn_numba(state: np.ndarray, bias_weights: list, bias_biases: list) -> np.ndarray:
    n = state.shape[0] // 2
    q = state[:n]
    q_dot = state[n:]
    enc = np.concatenate((np.sin(q), np.cos(q), q_dot))
    return mlp_forward_numba(enc, bias_weights, bias_biases)


@njit(cache=True)
def dynamics_nn_numba(
    state: np.ndarray,
    tau: np.ndarray,
    mass_weights: list,
    mass_biases: list,
    bias_weights: list,
    bias_biases: list,
    c_viscous: np.ndarray,
    c_coulomb: np.ndarray,
    v_stribeck: float,
) -> np.ndarray:
    n = c_viscous.shape[0]
    q = state[:n]
    q_dot = state[n:]

    M = mass_matrix_nn_numba(q, mass_weights, mass_biases)
    bias = bias_forces_nn_numba(state, bias_weights, bias_biases)
    friction = c_viscous * q_dot + c_coulomb * np.tanh(q_dot / v_stribeck)

    rhs = tau - bias - friction
    q_ddot = np.linalg.solve(M, rhs)

    out = np.zeros(2 * n)
    out[:n] = q_dot
    out[n:] = q_ddot
    return out


@njit(cache=True)
def rk4_step_nn_numba(
    state: np.ndarray,
    dt: float,
    tau: np.ndarray,
    mass_weights: list,
    mass_biases: list,
    bias_weights: list,
    bias_biases: list,
    c_viscous: np.ndarray,
    c_coulomb: np.ndarray,
    v_stribeck: float,
) -> np.ndarray:
    k1 = dynamics_nn_numba(state, tau, mass_weights, mass_biases, bias_weights, bias_biases, c_viscous, c_coulomb, v_stribeck)
    k2 = dynamics_nn_numba(state + 0.5 * dt * k1, tau, mass_weights, mass_biases, bias_weights, bias_biases, c_viscous, c_coulomb, v_stribeck)
    k3 = dynamics_nn_numba(state + 0.5 * dt * k2, tau, mass_weights, mass_biases, bias_weights, bias_biases, c_viscous, c_coulomb, v_stribeck)
    k4 = dynamics_nn_numba(state + dt * k3, tau, mass_weights, mass_biases, bias_weights, bias_biases, c_viscous, c_coulomb, v_stribeck)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@njit(cache=True)
def rollout_nn_numba(
    initial_state: np.ndarray,
    dt: float,
    tau_seq: np.ndarray,
    mass_weights: list,
    mass_biases: list,
    bias_weights: list,
    bias_biases: list,
    c_viscous: np.ndarray,
    c_coulomb: np.ndarray,
    v_stribeck: float,
) -> np.ndarray:
    """単一軌道をRK4でロールアウトする(#26のrollout_numbaのNN版)。

    tau_seq: shape (n_steps, 6) -> Returns: shape (n_steps+1, 12)
    """
    n_steps = tau_seq.shape[0]
    n = initial_state.shape[0]
    traj = np.zeros((n_steps + 1, n))
    traj[0] = initial_state
    state = initial_state.copy()
    for t in range(n_steps):
        state = rk4_step_nn_numba(
            state, dt, tau_seq[t], mass_weights, mass_biases, bias_weights, bias_biases,
            c_viscous, c_coulomb, v_stribeck,
        )
        traj[t + 1] = state
    return traj


@njit(cache=True)
def run_ptp_surrogate_nn_closed_loop_numba(
    initial_state: np.ndarray,
    target: np.ndarray,
    n_steps: int,
    dt: float,
    mass_weights: list,
    mass_biases: list,
    bias_weights: list,
    bias_biases: list,
    c_viscous: np.ndarray,
    c_coulomb: np.ndarray,
    v_stribeck: float,
    kp: float,
    ki: float,
    kd: float,
    tau_max: np.ndarray,
    integral_clip: float,
) -> np.ndarray:
    """計算トルク法PTP制御の閉ループ(#26のrun_ptp_closed_loop_numbaと同じ
    構成)。コントローラ自身のM(q)・bias(q,q_dot)は既知の真値物理(RNEA)を
    使い、プラント(状態遷移)は軽量NN(#19)のforward passで計算する。

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

        M_ctrl = mass_matrix_numba(q, DH_A, DH_ALPHA, DH_D, LINK_MASS, LINK_COM, LINK_INERTIA)
        bias_ctrl = bias_forces_numba(q, q_dot, DH_A, DH_ALPHA, DH_D, LINK_MASS, LINK_COM, LINK_INERTIA, GRAVITY)
        tau = M_ctrl @ q_ddot_desired + bias_ctrl
        for i in range(n):
            if tau[i] > tau_max[i]:
                tau[i] = tau_max[i]
            elif tau[i] < -tau_max[i]:
                tau[i] = -tau_max[i]

        state = rk4_step_nn_numba(
            state, dt, tau, mass_weights, mass_biases, bias_weights, bias_biases,
            c_viscous, c_coulomb, v_stribeck,
        )
        traj[t + 1] = state

    return traj


def run_ptp_surrogate_nn_from_model(
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
    """学習済み`LightweightGrayBoxModel`の重み・摩擦係数を取り出し、numbaで
    コンパイルした閉ループPTP制御(コントローラ=真値物理、プラント=軽量NN)を
    実行する(#28)。"""
    mass_weights, mass_biases = _extract_mlp_weights(model.mass_net)
    bias_weights, bias_biases = _extract_mlp_weights(model.bias_net)
    with torch.no_grad():
        c_viscous = torch.exp(model.log_c_viscous).cpu().numpy().astype(np.float64)
        c_coulomb = torch.exp(model.log_c_coulomb).cpu().numpy().astype(np.float64)
    return run_ptp_surrogate_nn_closed_loop_numba(
        initial_state.astype(np.float64), target.astype(np.float64), n_steps, dt,
        mass_weights, mass_biases, bias_weights, bias_biases,
        c_viscous, c_coulomb, model.v_stribeck,
        kp, ki, kd, tau_max.astype(np.float64), integral_clip,
    )


def _extract_mlp_weights(net: torch.nn.Sequential) -> tuple[list, list]:
    """nn.Sequential([Linear,Tanh,...,Linear])からnumba用の重み・バイアス
    リスト(numba.typed.List、float64のnumpy配列)を取り出す。"""
    weights = List()
    biases = List()
    with torch.no_grad():
        for layer in net:
            if isinstance(layer, torch.nn.Linear):
                weights.append(layer.weight.detach().cpu().numpy().astype(np.float64))
                biases.append(layer.bias.detach().cpu().numpy().astype(np.float64))
    return weights, biases


def rollout_from_lightweight_model(
    model, initial_state: np.ndarray, dt: float, tau_seq: np.ndarray
) -> np.ndarray:
    """学習済み`LightweightGrayBoxModel`から重み・摩擦係数を取り出し、numbaで
    コンパイルしたロールアウトを実行する(#28)。"""
    mass_weights, mass_biases = _extract_mlp_weights(model.mass_net)
    bias_weights, bias_biases = _extract_mlp_weights(model.bias_net)
    with torch.no_grad():
        c_viscous = torch.exp(model.log_c_viscous).cpu().numpy().astype(np.float64)
        c_coulomb = torch.exp(model.log_c_coulomb).cpu().numpy().astype(np.float64)
    return rollout_nn_numba(
        initial_state.astype(np.float64), dt, tau_seq.astype(np.float64),
        mass_weights, mass_biases, bias_weights, bias_biases,
        c_viscous, c_coulomb, model.v_stribeck,
    )
