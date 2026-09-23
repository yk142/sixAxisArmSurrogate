import numpy as np
import torch

import src.physics as ph
from src.model import (
    N_JOINTS,
    StructuredFrictionGrayBoxModel,
    bias_forces_torch,
    mass_matrix_torch,
    rnea_torch,
)


def test_rnea_matches_numpy():
    rng = np.random.default_rng(0)
    q = rng.uniform(-2, 2, size=N_JOINTS)
    q_dot = rng.uniform(-2, 2, size=N_JOINTS)
    q_ddot = rng.uniform(-2, 2, size=N_JOINTS)

    tau_np = ph.rnea(q, q_dot, q_ddot)
    tau_torch = rnea_torch(torch.tensor(q), torch.tensor(q_dot), torch.tensor(q_ddot)).numpy()
    assert np.allclose(tau_np, tau_torch, atol=1e-8)


def test_mass_matrix_matches_numpy():
    rng = np.random.default_rng(1)
    q = rng.uniform(-2, 2, size=(3, N_JOINTS))
    M_np = ph.mass_matrix(q)
    M_torch = mass_matrix_torch(torch.tensor(q)).numpy()
    assert np.allclose(M_np, M_torch, atol=1e-8)


def test_bias_forces_matches_numpy():
    rng = np.random.default_rng(2)
    q = rng.uniform(-2, 2, size=N_JOINTS)
    q_dot = rng.uniform(-2, 2, size=N_JOINTS)
    bias_np = ph.bias_forces(q[None, :], q_dot[None, :])[0]
    bias_torch = bias_forces_torch(torch.tensor(q)[None, :], torch.tensor(q_dot)[None, :])[0].numpy()
    assert np.allclose(bias_np, bias_torch, atol=1e-8)


def _zero_residual_graybox() -> StructuredFrictionGrayBoxModel:
    model = StructuredFrictionGrayBoxModel()
    with torch.no_grad():
        model.log_c_viscous.fill_(-30.0)  # exp(-30) ~= 0
        model.log_c_coulomb.fill_(-30.0)
    return model


def test_zero_residual_matches_known_physics_without_friction():
    """残差がほぼ0なら、グレーボックスの1ステップは無摩擦の既知物理RK4と一致すること。"""
    model = _zero_residual_graybox()
    dt = model.dt
    zero_friction = np.zeros(N_JOINTS)

    rng = np.random.default_rng(3)
    for _ in range(3):
        state_np = np.concatenate([rng.uniform(-1, 1, N_JOINTS), rng.uniform(-1, 1, N_JOINTS)])
        tau_np = rng.uniform(-1, 1, N_JOINTS)

        state = torch.tensor(state_np, dtype=torch.float32)
        u = torch.tensor(tau_np, dtype=torch.float32)

        with torch.no_grad():
            predicted = model.step(state, u).numpy()
        expected = ph.rk4_step(state_np, dt, tau_np, c_viscous=zero_friction, c_coulomb=zero_friction)
        assert np.allclose(predicted, expected, atol=1e-4)


def test_compiled_rollout_matches_uncompiled():
    """#16: torch.compile版のrolloutが非コンパイル版と一致すること。"""
    model = StructuredFrictionGrayBoxModel()
    rng = np.random.default_rng(4)
    ic = np.concatenate([rng.uniform(-1, 1, N_JOINTS), rng.uniform(-1, 1, N_JOINTS)])
    tau = rng.uniform(-1, 1, size=(3, N_JOINTS))

    plain = model.rollout(ic, 3, tau_seq=tau, use_compile=False)
    compiled = model.rollout(ic, 3, tau_seq=tau, use_compile=True)
    assert np.allclose(plain, compiled, atol=1e-4)
