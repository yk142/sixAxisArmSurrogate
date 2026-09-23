"""M2追加検証(#12): 任意の始点から任意の目標点へのPTP制御を、多数のランダム
シナリオでまとめて検証する。

これまでの`evaluate_ptp.py`は固定した目標姿勢1つ+4通りのオフセットのみを
検証していたが、実用性を確認するには始点・目標点の両方をランダムに
サンプリングした、より多数のシナリオで検証する必要がある。

`run_ptp_true_batch`/`run_ptp_surrogate_batch`(src/control.py)でN_SCENARIOS
本を1回のシミュレーションでまとめて処理することで高速化する。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.control import PTPController, angular_error, run_ptp_surrogate_batch, run_ptp_true_batch
from src.model import StructuredFrictionGrayBoxModel
from src.physics import N_JOINTS
from src.train import DT

N_SCENARIOS = 50
N_STEPS = 6000  # 12秒分(始点・目標点を全域からランダムサンプリングすると
# 複数関節が同時に大きくずれるケースがあり、6秒では収束しきらない=発散では
# なく単に遅い、ことを診断で確認したため、その分の余裕を見て延ばす)
WINDOW = 250  # 最終0.5秒
SUCCESS_THRESHOLD = 0.2
SEED = 2000

OUT_DIR = "outputs"


def sample_scenarios(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """始点・目標点をそれぞれ独立に(-pi,pi)全域から一様サンプリングする。

    Returns: initial_states shape (n,12)(角速度0スタート), targets shape (n,6)
    """
    q0 = rng.uniform(-np.pi, np.pi, size=(n, N_JOINTS))
    targets = rng.uniform(-np.pi, np.pi, size=(n, N_JOINTS))
    initial_states = np.concatenate([q0, np.zeros((n, N_JOINTS))], axis=1)
    return initial_states, targets


def final_errors(traj: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """各シナリオの最終窓平均追従誤差。traj: shape (n_steps+1, n, 12) -> shape (n,)"""
    q_final_window = traj[-WINDOW:, :, :N_JOINTS]
    err = np.abs(angular_error(q_final_window, targets[None, :, :])).sum(axis=-1)  # (WINDOW, n)
    return err.mean(axis=0)


def main() -> None:
    rng = np.random.default_rng(SEED)
    initial_states, targets = sample_scenarios(N_SCENARIOS, rng)

    print(f"running {N_SCENARIOS} scenarios x {N_STEPS} steps for true physics ...")
    true_traj = run_ptp_true_batch(initial_states, targets, N_STEPS, DT, PTPController())
    true_err = final_errors(true_traj, targets)
    true_success = true_err < SUCCESS_THRESHOLD
    print(f"true physics: {true_success.sum()}/{N_SCENARIOS} success, mean err={true_err.mean():.4f}")

    graybox = StructuredFrictionGrayBoxModel()
    graybox.load_state_dict(torch.load(f"{OUT_DIR}/model_graybox.pt", map_location="cpu"))
    graybox.eval()

    print(f"running {N_SCENARIOS} scenarios x {N_STEPS} steps for graybox surrogate ...")
    gray_traj = run_ptp_surrogate_batch(graybox, initial_states, targets, N_STEPS, DT, PTPController())
    gray_err = final_errors(gray_traj, targets)
    gray_success = gray_err < SUCCESS_THRESHOLD
    print(f"graybox: {gray_success.sum()}/{N_SCENARIOS} success, mean err={gray_err.mean():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(true_err, bins=20, alpha=0.7, label="true physics")
    axes[0].hist(gray_err, bins=20, alpha=0.7, label="graybox")
    axes[0].axvline(SUCCESS_THRESHOLD, color="red", linestyle=":", label="success threshold")
    axes[0].set_xlabel("final-window mean tracking error [rad]")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"n={N_SCENARIOS} random (start, target) scenarios")
    axes[0].legend(fontsize=8)

    axes[1].scatter(true_err, gray_err, s=12, alpha=0.7)
    max_val = max(true_err.max(), gray_err.max(), SUCCESS_THRESHOLD) * 1.1
    axes[1].plot([0, max_val], [0, max_val], color="gray", linestyle="--", linewidth=1)
    axes[1].axvline(SUCCESS_THRESHOLD, color="red", linestyle=":", linewidth=1)
    axes[1].axhline(SUCCESS_THRESHOLD, color="red", linestyle=":", linewidth=1)
    axes[1].set_xlabel("true physics final error [rad]")
    axes[1].set_ylabel("graybox final error [rad]")
    axes[1].set_title("per-scenario comparison")

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/m2_ptp_scenarios.png", dpi=120)
    print(f"saved {OUT_DIR}/m2_ptp_scenarios.png")


if __name__ == "__main__":
    main()
