"""M2: グレーボックス版(構造化摩擦)の開ループロールアウト誤差・摩擦係数の
学習確認を行う。姉妹プロジェクトscaraSurrogateのevaluate_graybox_3d.pyと
同じ検証方針。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataset import TAU_MAX, sample_initial_states
from src.model import StructuredFrictionGrayBoxModel
from src.physics import C_COULOMB, C_VISCOUS, N_JOINTS
from src.rollout import rmse_curve, true_rollout
from src.train import DT, K_MAX

N_STEPS_EVAL = 100  # 0.2秒分
N_TEST_TRAJ = 50
SEED_TEST = 1000
OUT_DIR = "outputs"


def make_control_conditions() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED_TEST + 1)
    n_holds = int(np.ceil(N_STEPS_EVAL / 5))
    hold_values = rng.uniform(-1.0, 1.0, size=(n_holds, N_JOINTS)) * TAU_MAX
    random_tau = np.repeat(hold_values, 5, axis=0)[:N_STEPS_EVAL]
    return {
        "zero torque": np.zeros((N_STEPS_EVAL, N_JOINTS)),
        "random torque": random_tau,
    }


def rmse_at_horizon(model: StructuredFrictionGrayBoxModel, conditions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED_TEST)
    ics = sample_initial_states(N_TEST_TRAJ, rng)

    curves = {}
    for label, tau_seq in conditions.items():
        true_traj = np.transpose(true_rollout(ics, DT, N_STEPS_EVAL, tau_seq=tau_seq), (1, 0, 2))
        pred_traj = model.rollout(ics, N_STEPS_EVAL, tau_seq=tau_seq)
        curves[label] = rmse_curve(true_traj, pred_traj)
    return curves


def plot_rmse(curves: dict[str, np.ndarray]) -> None:
    t = np.arange(N_STEPS_EVAL + 1) * DT
    fig, axes = plt.subplots(1, len(curves), figsize=(6 * len(curves), 4.5))
    if len(curves) == 1:
        axes = [axes]
    for ax, label in zip(axes, curves):
        ax.plot(t, curves[label])
        ax.axvline(K_MAX * DT, color="gray", linestyle="--", linewidth=1, label="training horizon")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("RMSE")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/m2_rmse_horizon.png", dpi=120)
    print(f"saved {OUT_DIR}/m2_rmse_horizon.png")


def plot_timeseries(model: StructuredFrictionGrayBoxModel, tau_seq: np.ndarray) -> None:
    rng = np.random.default_rng(SEED_TEST + 2)
    ic = sample_initial_states(1, rng)[0]
    true_traj = true_rollout(ic[None, :], DT, N_STEPS_EVAL, tau_seq=tau_seq)[0]
    pred_traj = model.rollout(ic[None, :], N_STEPS_EVAL, tau_seq=tau_seq)[:, 0, :]
    t = np.arange(N_STEPS_EVAL + 1) * DT

    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    for i in range(N_JOINTS):
        axes[0].plot(t, true_traj[:, i], color=f"C{i}", label=f"q{i+1} true")
        axes[0].plot(t, pred_traj[:, i], color=f"C{i}", linestyle="--", label=f"q{i+1} pred")
    axes[0].set_ylabel("joint angle [rad]")
    axes[0].axvline(K_MAX * DT, color="gray", linestyle="--", linewidth=1)
    axes[0].legend(ncol=3, fontsize=7)
    axes[0].set_title("random torque, true (solid) vs graybox (dashed)")

    for i in range(N_JOINTS):
        axes[1].plot(t, true_traj[:, N_JOINTS + i], color=f"C{i}")
        axes[1].plot(t, pred_traj[:, N_JOINTS + i], color=f"C{i}", linestyle="--")
    axes[1].set_ylabel("joint velocity [rad/s]")
    axes[1].set_xlabel("time [s]")
    axes[1].axvline(K_MAX * DT, color="gray", linestyle="--", linewidth=1)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/m2_timeseries.png", dpi=120)
    print(f"saved {OUT_DIR}/m2_timeseries.png")


def main() -> None:
    model = StructuredFrictionGrayBoxModel()
    model.load_state_dict(torch.load(f"{OUT_DIR}/model_graybox.pt", map_location="cpu"))
    model.eval()

    learned_viscous = torch.exp(model.log_c_viscous).detach().numpy()
    learned_coulomb = torch.exp(model.log_c_coulomb).detach().numpy()
    print("true  c_viscous:", C_VISCOUS, " c_coulomb:", C_COULOMB)
    print("learned c_viscous:", learned_viscous)
    print("learned c_coulomb:", learned_coulomb)

    conditions = make_control_conditions()
    curves = rmse_at_horizon(model, conditions)
    plot_rmse(curves)
    plot_timeseries(model, conditions["random torque"])


if __name__ == "__main__":
    main()
