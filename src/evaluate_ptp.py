"""M2: 6軸アームで計算トルク法PTP制御を、true physics(オラクル)と
grayboxサロゲートの2条件で比較する。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.control import PTPController, angular_error, run_ptp_surrogate, run_ptp_true
from src.model import StructuredFrictionGrayBoxModel
from src.physics import N_JOINTS
from src.train import DT

TARGET = np.array([0.5, 0.3, -0.4, 0.2, -0.3, 0.4])
N_STEPS = 3000  # 6秒分
WINDOW = 250  # 最終0.5秒
# 6関節合計の誤差しきい値。scaraSurrogate Phase3(3関節、しきい値0.1)と
# 同程度の関節あたり厳しさになるよう関節数に比例させる。
SUCCESS_THRESHOLD = 0.2

OFFSETS = [
    np.array([0.6, -0.5, 0.4, -0.3, 0.5, -0.4]),
    np.array([-0.6, 0.5, -0.4, 0.3, -0.5, 0.4]),
]

OUT_DIR = "outputs"


def make_test_ics() -> list[np.ndarray]:
    return [np.concatenate([TARGET + off, np.zeros(N_JOINTS)]) for off in OFFSETS]


def final_error(traj: np.ndarray, target: np.ndarray) -> float:
    return float(np.abs(angular_error(traj[-WINDOW:, :N_JOINTS], target)).sum(axis=-1).mean())


def evaluate_all(graybox: StructuredFrictionGrayBoxModel) -> dict[str, list[float]]:
    ics = make_test_ics()
    results = {"true physics": [], "graybox": []}
    for ic in ics:
        true_traj = run_ptp_true(ic, TARGET, N_STEPS, DT, PTPController())
        gray_traj = run_ptp_surrogate(graybox, ic, TARGET, N_STEPS, DT, PTPController())

        results["true physics"].append(final_error(true_traj, TARGET))
        results["graybox"].append(final_error(gray_traj, TARGET))
        print(f"IC offset norm={np.linalg.norm(ic[:N_JOINTS] - TARGET):.2f}: true={results['true physics'][-1]:.4f} gray={results['graybox'][-1]:.4f}")
    return results


def plot_representative_trajectory(graybox: StructuredFrictionGrayBoxModel) -> None:
    ic = make_test_ics()[0]
    t = np.arange(N_STEPS + 1) * DT

    true_traj = run_ptp_true(ic, TARGET, N_STEPS, DT, PTPController())
    gray_traj = run_ptp_surrogate(graybox, ic, TARGET, N_STEPS, DT, PTPController())

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for i in range(N_JOINTS):
        ax = axes[i // 3, i % 3]
        ax.plot(t, true_traj[:, i], label="true", linewidth=1.5)
        ax.plot(t, gray_traj[:, i], "--", label="graybox", linewidth=1.2)
        ax.axhline(TARGET[i], color="gray", linestyle=":", linewidth=1, label="target")
        ax.set_xlabel("time [s]")
        ax.set_ylabel(f"q{i+1} [rad]")
        ax.legend(fontsize=8)

    fig.suptitle("M2: PTP control representative trajectory")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/m2_ptp_trajectory.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/m2_ptp_trajectory.png")


def plot_error_summary(results: dict[str, list[float]]) -> None:
    plt.figure(figsize=(6, 4.5))
    plt.boxplot(list(results.values()), tick_labels=list(results.keys()))
    plt.axhline(SUCCESS_THRESHOLD, color="red", linestyle=":", linewidth=1, label="success threshold")
    plt.ylabel("final-window mean tracking error [rad]")
    plt.title(f"M2: PTP final tracking error (n={len(next(iter(results.values())))} IC)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/m2_ptp_error_summary.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/m2_ptp_error_summary.png")


def main() -> None:
    graybox = StructuredFrictionGrayBoxModel()
    graybox.load_state_dict(torch.load(f"{OUT_DIR}/model_graybox.pt", map_location="cpu"))
    graybox.eval()

    results = evaluate_all(graybox)
    n_success = sum(1 for e in results["graybox"] if e < SUCCESS_THRESHOLD)
    print(f"graybox success: {n_success}/{len(results['graybox'])}")
    plot_error_summary(results)
    plot_representative_trajectory(graybox)


if __name__ == "__main__":
    main()
