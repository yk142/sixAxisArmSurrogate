import numpy as np

from src.control import PTPController, angular_error, run_ptp_true
from src.physics import N_JOINTS


def test_ptp_true_converges_to_target():
    target = np.array([0.5, 0.3, -0.4, 0.2, -0.3, 0.4])
    offset = np.array([0.6, -0.5, 0.4, -0.3, 0.5, -0.4])
    ic = np.concatenate([target + offset, np.zeros(N_JOINTS)])

    traj = run_ptp_true(ic, target, n_steps=1500, dt=0.002, controller=PTPController())
    final_err = np.abs(angular_error(traj[-250:, :N_JOINTS], target)).sum(axis=-1).mean()

    assert final_err < 0.2
