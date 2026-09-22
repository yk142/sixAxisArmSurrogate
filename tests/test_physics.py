import numpy as np

from src.physics import N_JOINTS, simulate, total_energy

ZERO_FRICTION = np.zeros(N_JOINTS)


def test_energy_conservation_no_torque_no_friction():
    """無トルク・無摩擦では、6軸全体で力学的全エネルギーが保存されること。"""
    initial_state = np.array([0.2, 0.3, -0.5, 0.4, -0.2, 0.1, 0.3, -0.2, 0.5, -0.1, 0.2, -0.3])
    dt = 0.002
    n_steps = 2000  # 4秒分

    traj = simulate(initial_state, dt, n_steps, tau=None, c_viscous=ZERO_FRICTION, c_coulomb=ZERO_FRICTION)

    e = total_energy(traj)
    e0 = e[0]
    max_rel_drift = np.max(np.abs(e - e0)) / np.abs(e0)

    assert max_rel_drift < 1e-2


def test_rest_state_is_fixed_point_when_gravity_compensated():
    """全関節角速度0・トルク=重力補償トルクなら不動点であること(基本的な整合性チェック)。"""
    from src.physics import bias_forces

    q = np.array([0.3, 0.2, -0.4, 0.1, 0.5, -0.2])
    q_dot = np.zeros(N_JOINTS)
    state = np.concatenate([q, q_dot])
    gravity_compensation = bias_forces(q[None, :], q_dot[None, :])[0]

    traj = simulate(
        state, dt=0.005, n_steps=200, tau=gravity_compensation, c_viscous=ZERO_FRICTION, c_coulomb=ZERO_FRICTION
    )
    assert np.allclose(traj, state, atol=1e-6)


def test_mass_matrix_is_symmetric_positive_definite():
    from src.physics import mass_matrix

    q = np.array([0.4, -0.6, 0.2, 0.8, -0.3, 0.5])
    M = mass_matrix(q)
    assert np.allclose(M, M.T, atol=1e-8)
    eigvals = np.linalg.eigvalsh(M)
    assert np.all(eigvals > 0)
