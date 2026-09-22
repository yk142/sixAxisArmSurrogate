"""M1: 真値シミュレータの検証用スクリプト。

無トルク・無摩擦の自由運動でエネルギー保存則を確認し、各関節角度の時系列を
プロットする(scaraSurrogateのverify_physics_3d.pyと同じ検証方針)。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.physics import N_JOINTS, simulate, total_energy

ZERO_FRICTION = np.zeros(N_JOINTS)


def main() -> None:
    initial_state = np.array([0.2, 0.3, -0.5, 0.4, -0.2, 0.1, 0.3, -0.2, 0.5, -0.1, 0.2, -0.3])
    dt = 0.005
    n_steps = 400  # 2秒分

    traj = simulate(initial_state, dt, n_steps, tau=None, c_viscous=ZERO_FRICTION, c_coulomb=ZERO_FRICTION)
    t = np.arange(n_steps + 1) * dt
    e = total_energy(traj)
    e0 = e[0]
    max_rel_drift = np.max(np.abs(e - e0)) / np.abs(e0)
    print(f"max relative energy drift over {t[-1]:.1f}s: {max_rel_drift:.3e}")

    fig, axes = plt.subplots(3, 1, figsize=(8, 10))

    for i in range(N_JOINTS):
        axes[0].plot(t, traj[:, i], label=f"q{i+1}")
    axes[0].set_ylabel("joint angle [rad]")
    axes[0].set_title("Free motion (no torque, no friction)")
    axes[0].legend(ncol=3, fontsize=8)

    for i in range(N_JOINTS):
        axes[1].plot(t, traj[:, N_JOINTS + i], label=f"q{i+1}_dot")
    axes[1].set_ylabel("joint velocity [rad/s]")
    axes[1].legend(ncol=3, fontsize=8)

    axes[2].plot(t, e, label="total energy")
    axes[2].axhline(e0, color="gray", linestyle="--", linewidth=0.8, label="E(0)")
    axes[2].set_ylabel("energy [J]")
    axes[2].set_xlabel("time [s]")
    axes[2].set_title(f"max relative drift = {max_rel_drift:.2e}")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig("outputs/m1_energy_conservation.png", dpi=120)
    print("saved outputs/m1_energy_conservation.png")


if __name__ == "__main__":
    main()
