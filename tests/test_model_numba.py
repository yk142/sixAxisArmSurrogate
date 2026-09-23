import numpy as np
import torch

from src.control import KD, KI, KP, TAU_MAX
from src.model import N_JOINTS, LightweightGrayBoxModel
import src.model_numba as mn


def _make_model() -> LightweightGrayBoxModel:
    torch.manual_seed(0)
    model = LightweightGrayBoxModel()
    model.eval()
    return model


def test_mass_matrix_and_bias_match_torch():
    model = _make_model()
    rng = np.random.default_rng(1)
    q = rng.uniform(-2, 2, size=N_JOINTS)
    q_dot = rng.uniform(-2, 2, size=N_JOINTS)
    state = np.concatenate([q, q_dot])

    with torch.no_grad():
        M_torch = model.mass_matrix(torch.tensor(q, dtype=torch.float32)).numpy()
        bias_torch = model.bias_net(
            torch.tensor(np.concatenate([np.sin(q), np.cos(q), q_dot]), dtype=torch.float32)
        ).numpy()

    mass_w, mass_b = mn._extract_mlp_weights(model.mass_net)
    bias_w, bias_b = mn._extract_mlp_weights(model.bias_net)
    M_nb = mn.mass_matrix_nn_numba(q, mass_w, mass_b)
    bias_nb = mn.bias_forces_nn_numba(state, bias_w, bias_b)

    assert np.allclose(M_torch, M_nb, atol=1e-5)
    assert np.allclose(bias_torch, bias_nb, atol=1e-5)


def test_rollout_matches_torch():
    model = _make_model()
    rng = np.random.default_rng(2)
    ic = rng.uniform(-1, 1, size=2 * N_JOINTS)
    n_steps = 20
    tau = rng.uniform(-1, 1, size=(n_steps, N_JOINTS))

    traj_torch = model.rollout(ic, n_steps, tau_seq=tau)
    traj_nb = mn.rollout_from_lightweight_model(model, ic, 0.002, tau)
    assert np.allclose(traj_torch, traj_nb, atol=1e-4)


def test_closed_loop_matches_python_reference():
    """#28: numba版PTP閉ループ(コントローラ=真値物理、プラント=軽量NN)が、
    python版コントローラ+torch版モデルのrolloutの逐次実行と一致することを
    確認する。"""
    from src.control import PTPController

    model = _make_model()
    target = np.array([0.5, 0.3, -0.4, 0.2, -0.3, 0.4])
    ic = np.concatenate([target + np.array([0.6, -0.5, 0.4, -0.3, 0.5, -0.4]), np.zeros(N_JOINTS)])
    n_steps = 50
    dt = 0.002

    ctrl = PTPController()
    ctrl.reset()
    state = ic.copy()
    traj_ref = [state.copy()]
    for _ in range(n_steps):
        tau = ctrl.compute(state, target, dt)
        state = model.rollout(state, 1, tau_seq=tau[None, :])[-1]
        traj_ref.append(state.copy())
    traj_ref = np.stack(traj_ref)

    traj_nb = mn.run_ptp_surrogate_nn_from_model(model, ic, target, n_steps, dt, KP, KI, KD, TAU_MAX, 4.0)
    assert np.allclose(traj_ref, traj_nb, atol=1e-4)
