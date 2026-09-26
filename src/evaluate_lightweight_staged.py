"""#32: 段階的カリキュラム(直接教師ありmass_net/bias_net + 摩擦のみロールアウト
学習)モデルの開ループRMSE・PTP制御成功率を評価する。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.control import PTPController, run_ptp_surrogate, run_ptp_true
from src.dataset import sample_initial_states
from src.evaluate_graybox import N_STEPS_EVAL, make_control_conditions
from src.evaluate_ptp import SUCCESS_THRESHOLD, TARGET, final_error, make_test_ics
from src.model import LightweightGrayBoxModel
from src.physics import N_JOINTS
from src.rollout import rmse_curve, true_rollout
from src.train import DT

OUT_DIR = "outputs"
CKPT = f"{OUT_DIR}/model_lightweight_staged.pt"
SEED_TEST = 1000
N_STEPS_6S = 3000
N_STEPS_12S = 6000
WINDOW = 250


def main() -> None:
    model = LightweightGrayBoxModel()
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()

    conditions = make_control_conditions()
    rng = np.random.default_rng(SEED_TEST)
    ics = sample_initial_states(50, rng)
    curves = {}
    for label, tau_seq in conditions.items():
        true_traj = np.transpose(true_rollout(ics, DT, N_STEPS_EVAL, tau_seq=tau_seq), (1, 0, 2))
        pred_traj = model.rollout(ics, N_STEPS_EVAL, tau_seq=tau_seq)
        curves[label] = rmse_curve(true_traj, pred_traj)
        print(f"[open-loop RMSE] {label}: final={curves[label][-1]:.4f}  max={curves[label].max():.4f}")

    t = np.arange(N_STEPS_EVAL + 1) * DT
    fig, axes = plt.subplots(1, len(curves), figsize=(6 * len(curves), 4.5))
    for ax, label in zip(axes, curves):
        ax.plot(t, curves[label])
        ax.set_xlabel("time [s]")
        ax.set_ylabel("RMSE")
        ax.set_title(label, fontsize=10)
    fig.suptitle("#32: LightweightGrayBoxModel (staged: supervised M/bias + friction) open-loop RMSE")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/m32_rmse_horizon.png", dpi=120)
    print(f"saved {OUT_DIR}/m32_rmse_horizon.png")

    with torch.no_grad():
        for label, n_steps in [("6秒", N_STEPS_6S), ("12秒", N_STEPS_12S)]:
            results = []
            for ic in make_test_ics():
                traj = run_ptp_surrogate(model, ic, TARGET, n_steps, DT, PTPController())
                err = final_error(traj, TARGET)
                results.append(err)
                print(f"[PTP {label}] IC offset norm={np.linalg.norm(ic[:N_JOINTS] - TARGET):.2f}: final_err={err:.4f}")
            n_success = sum(1 for e in results if e < SUCCESS_THRESHOLD)
            print(f"PTP success ({label}): {n_success}/{len(results)}")

    ic = make_test_ics()[0]
    t_ptp = np.arange(N_STEPS_12S + 1) * DT
    with torch.no_grad():
        true_traj = run_ptp_true(ic, TARGET, N_STEPS_12S, DT, PTPController())
        pred_traj = run_ptp_surrogate(model, ic, TARGET, N_STEPS_12S, DT, PTPController())
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for i in range(N_JOINTS):
        ax = axes[i // 3, i % 3]
        ax.plot(t_ptp, true_traj[:, i], label="true", linewidth=1.5)
        ax.plot(t_ptp, pred_traj[:, i], "--", label="staged", linewidth=1.2)
        ax.axhline(TARGET[i], color="gray", linestyle=":", linewidth=1, label="target")
        ax.set_xlabel("time [s]")
        ax.set_ylabel(f"q{i + 1} [rad]")
        ax.legend(fontsize=8)
    fig.suptitle("#32: PTP control, true vs staged (supervised M/bias + friction)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/m32_ptp_trajectory.png", dpi=150)
    print(f"saved {OUT_DIR}/m32_ptp_trajectory.png")


if __name__ == "__main__":
    main()
