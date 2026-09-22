import numpy as np

from src.dataset import generate_controlled_trajectories, make_rollout_windows
from src.physics import N_JOINTS


def test_generate_controlled_trajectories_are_finite_and_bounded():
    traj, tau = generate_controlled_trajectories(20, dt=0.002, n_steps=50, seed=0)
    assert traj.shape == (20, 51, 2 * N_JOINTS)
    assert tau.shape == (20, 50, N_JOINTS)
    assert np.all(np.isfinite(traj))
    assert np.max(np.abs(traj[..., N_JOINTS:])) < 20.0


def test_make_rollout_windows_shapes():
    traj, tau = generate_controlled_trajectories(5, dt=0.002, n_steps=20, seed=1)
    k = 4
    x0, u_seq, targets = make_rollout_windows(traj, tau, k, stride=2)
    n_windows_per_traj = (20 - k) // 2 + 1
    assert x0.shape == (5 * n_windows_per_traj, 2 * N_JOINTS)
    assert u_seq.shape == (k, 5 * n_windows_per_traj, N_JOINTS)
    assert targets.shape == (k, 5 * n_windows_per_traj, 2 * N_JOINTS)
