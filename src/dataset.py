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

from src.physics import N_JOINTS, bias_forces, simulate_batch

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



# #6: 全軌道を一律TAU_MAXでランダム駆動すると、V_STRIBECK(0.03)付近の低速域を
# ほとんど探索せず(実測: 全関節で|q_dot|<2*V_STRIBECKとなる時間は3-6%のみ、
# 中央値は0.6-1.1rad/sとV_STRIBECKよりずっと大きい)、tanh(q_dot/V_STRIBECK)が
# ほぼ常に飽和(±1)してしまう。これでは粘性摩擦(速度に線形)とクーロン摩擦
# (速度の符号のみに依存する定数)の寄与が観測上ほぼ同じ形になり、係数を分離
# して識別できない。軌道ごとにトルクの強さをランダムに変える(弱いトルクの
# 軌道は低速域に留まりやすい)ことで、低速域・高速域の両方を学習データに
# 含める。
TORQUE_SCALE_RANGE = (0.05, 1.0)


def sample_torque_sequences(
    n: int,
    n_steps: int,
    rng: np.random.Generator,
    tau_max: np.ndarray = TAU_MAX,
    hold_steps: int = TORQUE_HOLD_STEPS,
) -> np.ndarray:
    """n本分のランダムな区分定数トルク列を生成する。shape (n, n_steps, 6)

    軌道ごとに独立したスケール係数(TORQUE_SCALE_RANGE)をtau_maxに掛けることで、
    低速域(弱いトルク)から高速域(強いトルク)まで幅広い速度域を学習データに
    含める。
    """
    n_holds = int(np.ceil(n_steps / hold_steps))
    scale = rng.uniform(*TORQUE_SCALE_RANGE, size=(n, 1, 1))
    hold_values = rng.uniform(-1.0, 1.0, size=(n, n_holds, N_JOINTS)) * tau_max * scale
    return np.repeat(hold_values, hold_steps, axis=1)[:, :n_steps]


# #6: 腕リンク(1-3)は無トルクだと重力だけで素早く加速してしまうため、
# トルクのスケールを下げるだけでは低速域を十分に探索できない(実測で確認済み)。
# データの一部は初期姿勢の重力補償トルク+小さなランダム摂動で駆動し、
# 関節を低速(V_STRIBECK付近)に留まらせることで、粘性摩擦とクーロン摩擦を
# 分離しやすくする。
GRAVITY_COMPENSATED_FRACTION = 0.25
GRAVITY_COMPENSATED_NOISE_SCALE = 0.05


def sample_gravity_compensated_torque_sequences(
    initial_states: np.ndarray,
    n_steps: int,
    rng: np.random.Generator,
    tau_max: np.ndarray = TAU_MAX,
    hold_steps: int = TORQUE_HOLD_STEPS,
) -> np.ndarray:
    """初期姿勢の重力補償トルク+小さなランダム摂動の区分定数トルク列を生成する。

    関節をほぼ静止状態(低速域)に留まらせ、V_STRIBECK付近を学習データに
    含めるためのもの。shape (n, n_steps, 6)
    """
    n = initial_states.shape[0]
    q0 = initial_states[:, :N_JOINTS]
    gravity_comp = bias_forces(q0, np.zeros_like(q0))  # (n, 6)

    n_holds = int(np.ceil(n_steps / hold_steps))
    noise = rng.uniform(-1.0, 1.0, size=(n, n_holds, N_JOINTS)) * tau_max * GRAVITY_COMPENSATED_NOISE_SCALE
    hold_values = gravity_comp[:, None, :] + noise
    return np.repeat(hold_values, hold_steps, axis=1)[:, :n_steps]


def generate_controlled_trajectories(
    n_trajectories: int, dt: float, n_steps: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """トルク列で駆動した軌道データセットを`simulate_batch`でまとめて生成する。

    半数はランダムトルク列(高速域を含む幅広い速度域を探索)、残り半数は
    重力補償+小摂動トルク列(V_STRIBECK付近の低速域を探索)で駆動する
    (#6: 粘性摩擦とクーロン摩擦を分離して識別するため)。

    角速度が`VELOCITY_LIMIT`を超える軌道は棄却して引き直す。

    Returns:
        trajectories: shape (n_traj, n_steps + 1, 12)
        tau_seqs:     shape (n_traj, n_steps, 6)
    """
    rng = np.random.default_rng(seed)
    initial_states = sample_initial_states(n_trajectories, rng)

    n_gc = int(n_trajectories * GRAVITY_COMPENSATED_FRACTION)
    is_gc = np.zeros(n_trajectories, dtype=bool)
    is_gc[:n_gc] = True

    tau_seqs = np.empty((n_trajectories, n_steps, N_JOINTS))
    tau_seqs[is_gc] = sample_gravity_compensated_torque_sequences(initial_states[is_gc], n_steps, rng)
    tau_seqs[~is_gc] = sample_torque_sequences((~is_gc).sum(), n_steps, rng)

    trajectories = simulate_batch(initial_states, dt, n_steps, tau_seqs)

    for _ in range(MAX_RESAMPLE_ATTEMPTS):
        bad = ~np.all(np.isfinite(trajectories), axis=(1, 2)) | (
            np.max(np.abs(trajectories[..., N_JOINTS:]), axis=(1, 2)) > VELOCITY_LIMIT
        )
        n_bad = int(bad.sum())
        if n_bad == 0:
            break
        initial_states[bad] = sample_initial_states(n_bad, rng)
        bad_gc = bad & is_gc
        bad_random = bad & ~is_gc
        if bad_gc.any():
            tau_seqs[bad_gc] = sample_gravity_compensated_torque_sequences(initial_states[bad_gc], n_steps, rng)
        if bad_random.any():
            tau_seqs[bad_random] = sample_torque_sequences(int(bad_random.sum()), n_steps, rng)
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
