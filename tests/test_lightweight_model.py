import torch

from src.model import N_JOINTS, LightweightGrayBoxModel


def test_mass_matrix_is_symmetric_positive_definite():
    torch.manual_seed(0)
    model = LightweightGrayBoxModel()
    q = torch.randn(4, N_JOINTS)
    M = model.mass_matrix(q)
    assert torch.allclose(M, M.transpose(-1, -2), atol=1e-6)
    eigvals = torch.linalg.eigvalsh(M)
    assert torch.all(eigvals > 0)


def test_step_shape_and_gradient_flow():
    torch.manual_seed(0)
    model = LightweightGrayBoxModel()
    state = torch.randn(4, 2 * N_JOINTS)
    u = torch.randn(4, N_JOINTS)
    out = model.step(state, u)
    assert out.shape == state.shape

    loss = out.pow(2).sum()
    loss.backward()
    assert model.mass_net[0].weight.grad is not None
    assert model.bias_net.gravity_net[0].weight.grad is not None
    assert model.bias_net.coriolis_net[0].weight.grad is not None
