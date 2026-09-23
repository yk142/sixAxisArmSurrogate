import numpy as np

import src.physics as ph
import src.physics_numba as phn
from src.control import PTPController, TAU_MAX, run_ptp_true


def test_rnea_matches_numpy():
    rng = np.random.default_rng(1)
    q = rng.uniform(-2, 2, size=6)
    q_dot = rng.uniform(-2, 2, size=6)
    q_ddot = rng.uniform(-2, 2, size=6)

    tau_np = ph.rnea(q, q_dot, q_ddot)
    tau_nb = phn.rnea_numba(
        q, q_dot, q_ddot, ph.DH_A, ph.DH_ALPHA, ph.DH_D, ph.LINK_MASS, ph.LINK_COM, ph.LINK_INERTIA, ph.GRAVITY
    )
    assert np.allclose(tau_np, tau_nb, atol=1e-10)


def test_mass_matrix_and_bias_match_numpy():
    rng = np.random.default_rng(2)
    q = rng.uniform(-2, 2, size=6)
    q_dot = rng.uniform(-2, 2, size=6)

    M_np = ph.mass_matrix(q[None, :])[0]
    M_nb = phn.mass_matrix(q)
    assert np.allclose(M_np, M_nb, atol=1e-10)

    bias_np = ph.bias_forces(q[None, :], q_dot[None, :])[0]
    bias_nb = phn.bias_forces(q, q_dot)
    assert np.allclose(bias_np, bias_nb, atol=1e-10)


def test_rollout_matches_numpy_simulate():
    rng = np.random.default_rng(3)
    ic = rng.uniform(-1, 1, size=12)
    n_steps = 20
    tau = rng.uniform(-1, 1, size=(n_steps, 6))
    c_viscous = np.ones(6)
    c_coulomb = np.full(6, 0.4)

    traj_np = ph.simulate(ic, 0.002, n_steps, tau=tau, c_viscous=c_viscous, c_coulomb=c_coulomb)
    traj_nb = phn.rollout_numba(
        ic, 0.002, tau, ph.DH_A, ph.DH_ALPHA, ph.DH_D, ph.LINK_MASS, ph.LINK_COM, ph.LINK_INERTIA,
        ph.GRAVITY, c_viscous, c_coulomb, ph.V_STRIBECK,
    )
    assert np.allclose(traj_np, traj_nb, atol=1e-10)


def test_closed_loop_ptp_matches_python_controller():
    """#26: numbaにまとめた閉ループPTPが、Python版コントローラ+rk4_stepの
    逐次実行(run_ptp_true)と一致することを確認する。"""
    target = np.array([0.5, 0.3, -0.4, 0.2, -0.3, 0.4])
    ic = np.concatenate([target + np.array([0.6, -0.5, 0.4, -0.3, 0.5, -0.4]), np.zeros(6)])
    n_steps = 100
    dt = 0.002

    traj_ref = run_ptp_true(ic, target, n_steps, dt, PTPController())
    traj_nb = phn.run_ptp_closed_loop(
        ic, target, n_steps, dt, ph.C_VISCOUS, ph.C_COULOMB, 150.0, 15.0, 24.0, TAU_MAX, 4.0
    )
    assert np.allclose(traj_ref, traj_nb, atol=1e-8)
