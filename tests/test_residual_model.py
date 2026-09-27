import torch

from src.base_physics import bias_forces_base, mass_matrix_base
from src.model import N_JOINTS, ResidualGrayBoxModel


def test_base_physics_output_shapes():
    q = torch.randn(2, 3, N_JOINTS)
    q_dot = torch.randn(2, 3, N_JOINTS)
    m = mass_matrix_base(q)
    bias = bias_forces_base(q, q_dot)
    assert m.shape == (2, 3, N_JOINTS, N_JOINTS)
    assert bias.shape == q.shape


def test_mass_matrix_is_symmetric_positive_definite():
    torch.manual_seed(0)
    model = ResidualGrayBoxModel()
    q = torch.randn(4, N_JOINTS)
    m = model.mass_matrix(q)
    assert torch.allclose(m, m.transpose(-1, -2), atol=1e-6)
    eigvals = torch.linalg.eigvalsh(m)
    assert torch.all(eigvals > 0)


def test_step_shape_and_gradient_flow_to_residual_mlps():
    torch.manual_seed(0)
    model = ResidualGrayBoxModel()
    state = torch.randn(4, 2 * N_JOINTS)
    u = torch.randn(4, N_JOINTS)
    out = model.step(state, u)
    assert out.shape == state.shape

    loss = out.pow(2).sum()
    loss.backward()
    assert model.mass_net[0].weight.grad is not None
    assert torch.isfinite(model.mass_net[0].weight.grad).all()
    assert model.bias_net[0].weight.grad is not None
    assert torch.isfinite(model.bias_net[0].weight.grad).all()
