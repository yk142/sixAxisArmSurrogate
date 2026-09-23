"""#26追加検証: numba版の推論構成(単一軌道向け閉ループ)で、#12と同じ
50シナリオ(始点・目標点をともに全域からランダムサンプリング)のPTP制御を
検証する。

#12はtorch版をバッチ化して50シナリオを1回のシミュレーションでまとめて
処理したが、numba版はバッチ化による恩恵がない(#26で確認済み)ため、ここでは
単純にPythonループで50シナリオを逐次実行する。1軌道あたりが非常に高速な
ため、逐次実行でも実用的な時間で完了する。

`evaluate_ptp_scenarios.py`と全く同じサンプリング(同じSEED)・しきい値・
ステップ数を使うことで、torch版(#12)との結果比較ができるようにする。
"""
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import src.physics_numba as phn
from src.evaluate_ptp_scenarios import N_SCENARIOS, N_STEPS, SEED, SUCCESS_THRESHOLD, WINDOW, sample_scenarios
from src.control import KD, KI, KP, TAU_MAX, angular_error
from src.model import StructuredFrictionGrayBoxModel
from src.physics import C_COULOMB, C_VISCOUS, N_JOINTS
from src.train import DT

OUT_DIR = "outputs"


def final_error(traj: np.ndarray, target: np.ndarray) -> float:
    """1シナリオの最終窓平均追従誤差。traj: shape (n_steps+1, 12) -> float"""
    q_final_window = traj[-WINDOW:, :N_JOINTS]
    return float(np.abs(angular_error(q_final_window, target)).sum(axis=-1).mean())


def run_all_true(initial_states: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = initial_states.shape[0]
    errs = np.zeros(n)
    trajs = []
    for i in range(n):
        traj = phn.run_ptp_closed_loop(
            initial_states[i], targets[i], N_STEPS, DT, C_VISCOUS, C_COULOMB, KP, KI, KD, TAU_MAX, 4.0
        )
        errs[i] = final_error(traj, targets[i])
        trajs.append(traj)
    return np.stack(trajs, axis=1), errs


def run_all_surrogate(model, initial_states: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = initial_states.shape[0]
    errs = np.zeros(n)
    trajs = []
    for i in range(n):
        traj = phn.run_ptp_surrogate_from_model(model, initial_states[i], targets[i], N_STEPS, DT, KP, KI, KD, TAU_MAX, 4.0)
        errs[i] = final_error(traj, targets[i])
        trajs.append(traj)
    return np.stack(trajs, axis=1), errs


def main() -> None:
    rng = np.random.default_rng(SEED)
    initial_states, targets = sample_scenarios(N_SCENARIOS, rng)

    # warmup (JITコンパイル、1回目の呼び出し分は計測から除く)
    phn.run_ptp_closed_loop(initial_states[0], targets[0], 2, DT, C_VISCOUS, C_COULOMB, KP, KI, KD, TAU_MAX, 4.0)

    print(f"running {N_SCENARIOS} scenarios x {N_STEPS} steps for true physics (numba, sequential) ...")
    t0 = time.time()
    true_traj, true_err = run_all_true(initial_states, targets)
    t_true = time.time() - t0
    true_success = true_err < SUCCESS_THRESHOLD
    print(f"true physics: {true_success.sum()}/{N_SCENARIOS} success, mean err={true_err.mean():.4f}  ({t_true:.1f}s)")

    graybox = StructuredFrictionGrayBoxModel()
    graybox.load_state_dict(torch.load(f"{OUT_DIR}/model_graybox.pt", map_location="cpu"))
    graybox.eval()
    phn.run_ptp_surrogate_from_model(graybox, initial_states[0], targets[0], 2, DT, KP, KI, KD, TAU_MAX, 4.0)

    print(f"running {N_SCENARIOS} scenarios x {N_STEPS} steps for graybox surrogate (numba, sequential) ...")
    t0 = time.time()
    gray_traj, gray_err = run_all_surrogate(graybox, initial_states, targets)
    t_gray = time.time() - t0
    gray_success = gray_err < SUCCESS_THRESHOLD
    print(f"graybox: {gray_success.sum()}/{N_SCENARIOS} success, mean err={gray_err.mean():.4f}  ({t_gray:.1f}s)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(true_err, bins=20, alpha=0.7, label="true physics")
    axes[0].hist(gray_err, bins=20, alpha=0.7, label="graybox")
    axes[0].axvline(SUCCESS_THRESHOLD, color="red", linestyle=":", label="success threshold")
    axes[0].set_xlabel("final-window mean tracking error [rad]")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"n={N_SCENARIOS} random (start, target) scenarios (numba)")
    axes[0].legend(fontsize=8)

    axes[1].scatter(true_err, gray_err, s=12, alpha=0.7)
    max_val = max(true_err.max(), gray_err.max(), SUCCESS_THRESHOLD) * 1.1
    axes[1].plot([0, max_val], [0, max_val], color="gray", linestyle="--", linewidth=1)
    axes[1].axvline(SUCCESS_THRESHOLD, color="red", linestyle=":", linewidth=1)
    axes[1].axhline(SUCCESS_THRESHOLD, color="red", linestyle=":", linewidth=1)
    axes[1].set_xlabel("true physics final error [rad]")
    axes[1].set_ylabel("graybox final error [rad]")
    axes[1].set_title("per-scenario comparison (numba)")

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/m2_ptp_scenarios_numba.png", dpi=120)
    print(f"saved {OUT_DIR}/m2_ptp_scenarios_numba.png")


if __name__ == "__main__":
    main()
