import numpy as np

from src.control import PTPController, angular_error, run_ptp_true, run_ptp_true_batch
from src.physics import N_JOINTS


def test_ptp_true_converges_to_target():
    target = np.array([0.5, 0.3, -0.4, 0.2, -0.3, 0.4])
    offset = np.array([0.6, -0.5, 0.4, -0.3, 0.5, -0.4])
    ic = np.concatenate([target + offset, np.zeros(N_JOINTS)])

    traj = run_ptp_true(ic, target, n_steps=1500, dt=0.002, controller=PTPController())
    final_err = np.abs(angular_error(traj[-250:, :N_JOINTS], target)).sum(axis=-1).mean()

    assert final_err < 0.2


def test_ptp_true_batch_matches_sequential():
    """#12: バッチ版PTP制御が1本ずつ実行した場合と一致することを確認する。"""
    rng = np.random.default_rng(0)
    batch = 3
    q0 = rng.uniform(-np.pi, np.pi, size=(batch, N_JOINTS))
    targets = rng.uniform(-np.pi, np.pi, size=(batch, N_JOINTS))
    ics = np.concatenate([q0, np.zeros((batch, N_JOINTS))], axis=1)

    traj_batch = run_ptp_true_batch(ics, targets, n_steps=200, dt=0.002, controller=PTPController())

    for i in range(batch):
        traj_single = run_ptp_true(ics[i], targets[i], n_steps=200, dt=0.002, controller=PTPController())
        assert np.allclose(traj_batch[:, i, :], traj_single, atol=1e-8)
