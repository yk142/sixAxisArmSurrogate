"""学習・評価用の軌道データセット生成。

M1(#1)実装時に判明した通り、手首側の小慣性関節はdt=0.02では数値的に不安定に
なるため、M1のエネルギー保存検証で確認済みのdt=0.002を用いる。またトルク
範囲は全関節一律ではなく、各関節の慣性スケールに応じて個別に設定する
(腕リンク(1-3)は慣性が大きく大きなトルクに耐えるが、手首リンク(4-6)は
慣性が小さく同じトルクでは暴走してしまうため)。

`simulate_batch`(src/physics.py)を使い、軌道をまとめて生成することで
高速化する(680軌道×100ステップが逐次実行なら数十分かかるところ、
バッチ化により数十秒で完了する)。
"""
import numpy as np

from src.physics import N_JOINTS, simulate_batch

THETA_RANGE = (-np.pi, np.pi)
THETA_DOT_RANGE = (-1.0, 1.0)

DT = 0.002
TORQUE_HOLD_STEPS = 5

# 各関節の慣性スケールに応じたトルク範囲(M1実装時の診断: 腕関節(1-3)は
# 慣性が大きく±35-40Nm程度まで安定、手首関節(4-6)は慣性が小さく
# それぞれ±1.5, 1.5, 0.2Nm程度が目安)。
TAU_MAX = np.array([40.0, 35.0, 10.0, 1.5, 1.5, 0.2])

# 角速度がこれを超える軌道はカオス的とみなし棄却して引き直す(Phase2-M2の
# 教訓と同じ棄却サンプリング)。
VELOCITY_LIMIT = 15.0
MAX_RESAMPLE_ATTEMPTS = 5


def sample_initial_states(n: int, rng: np.random.Generator) -> np.ndarray:
    """ICを一様サンプリングする。shape (n, 12) = [q(6), q_dot(6)]"""
    theta = rng.uniform(*THETA_RANGE, size=(n, N_JOINTS))
    theta_dot = rng.uniform(*THETA_DOT_RANGE, size=(n, N_JOINTS))
    return np.concatenate([theta, theta_dot], axis=-1)


def sample_torque_sequences(
    n: int,
    n_steps: int,
    rng: np.random.Generator,
    tau_max: np.ndarray = TAU_MAX,
    hold_steps: int = TORQUE_HOLD_STEPS,
) -> np.ndarray:
    """n本分のランダムな区分定数トルク列を生成する。shape (n, n_steps, 6)"""
    n_holds = int(np.ceil(n_steps / hold_steps))
    hold_values = rng.uniform(-1.0, 1.0, size=(n, n_holds, N_JOINTS)) * tau_max
    return np.repeat(hold_values, hold_steps, axis=1)[:, :n_steps]


def generate_controlled_trajectories(
    n_trajectories: int, dt: float, n_steps: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """ランダムトルク列で駆動した軌道データセットを`simulate_batch`でまとめて生成する。

    角速度が`VELOCITY_LIMIT`を超える軌道は棄却して引き直す。

    Returns:
        trajectories: shape (n_traj, n_steps + 1, 12)
        tau_seqs:     shape (n_traj, n_steps, 6)
    """
    rng = np.random.default_rng(seed)
    initial_states = sample_initial_states(n_trajectories, rng)
    tau_seqs = sample_torque_sequences(n_trajectories, n_steps, rng)
    trajectories = simulate_batch(initial_states, dt, n_steps, tau_seqs)

    for _ in range(MAX_RESAMPLE_ATTEMPTS):
        bad = ~np.all(np.isfinite(trajectories), axis=(1, 2)) | (
            np.max(np.abs(trajectories[..., N_JOINTS:]), axis=(1, 2)) > VELOCITY_LIMIT
        )
        n_bad = int(bad.sum())
        if n_bad == 0:
            break
        initial_states[bad] = sample_initial_states(n_bad, rng)
        tau_seqs[bad] = sample_torque_sequences(n_bad, n_steps, rng)
        trajectories[bad] = simulate_batch(initial_states[bad], dt, n_steps, tau_seqs[bad])

    return trajectories, tau_seqs


def make_rollout_windows(
    trajectories: np.ndarray, tau_seqs: np.ndarray, k: int, stride: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """軌道群+トルク列からKステップの学習ウィンドウ (x0, u_seq, targets) を作る。

    trajectories: shape (n_traj, n_steps + 1, 12), tau_seqs: shape (n_traj, n_steps, 6)

    Returns:
        x0:      shape (n_windows, 12)      各ウィンドウの初期状態
        u_seq:   shape (k, n_windows, 6)    各ウィンドウのKステップ分トルク列
        targets: shape (k, n_windows, 12)   各ウィンドウのKステップ分正解状態
    """
    n_traj, n_steps_plus_1, _ = trajectories.shape
    n_steps = n_steps_plus_1 - 1

    x0_parts, u_parts, target_parts = [], [], []
    for start in range(0, n_steps - k + 1, stride):
        x0_parts.append(trajectories[:, start, :])
        u_parts.append(tau_seqs[:, start : start + k, :])
        target_parts.append(trajectories[:, start + 1 : start + k + 1, :])

    x0 = np.concatenate(x0_parts, axis=0)
    u_seq = np.concatenate(u_parts, axis=0).transpose(1, 0, 2)
    targets = np.concatenate(target_parts, axis=0).transpose(1, 0, 2)
    return x0, u_seq, targets
