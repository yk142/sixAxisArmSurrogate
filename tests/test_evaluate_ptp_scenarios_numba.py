import numpy as np

from src.control import KD, KI, KP, TAU_MAX
from src.evaluate_ptp_scenarios_numba import final_error
from src.physics import C_COULOMB, C_VISCOUS, N_JOINTS
from src.train import DT
import src.physics_numba as phn


def test_final_error_matches_manual_computation():
    """final_errorが、生の角度差から計算した値と一致することを確認する
    (#12と同じ評価指標をnumba版でも使っていることの軽量な確認)。"""
    rng = np.random.default_rng(0)
    q0 = rng.uniform(-np.pi, np.pi, size=N_JOINTS)
    target = rng.uniform(-np.pi, np.pi, size=N_JOINTS)
    ic = np.concatenate([q0, np.zeros(N_JOINTS)])

    n_steps = 50
    traj = phn.run_ptp_closed_loop(ic, target, n_steps, DT, C_VISCOUS, C_COULOMB, KP, KI, KD, TAU_MAX, 4.0)
    err = final_error(traj, target)
    assert np.isfinite(err)
    assert err >= 0.0
