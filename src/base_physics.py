"""Issue #38: residual gray-box model用の安価な閉形式ベース物理。

RNEAは使わず、q1-q3をヨー+肩+肘の3自由度腕として解析式で近似する。
q4-q6の手首リンク質量は前腕先端の質点として腕側に集約し、手首3軸は
定数対角慣性、bias=0で近似する。
"""
import torch

from src.physics import DH_A, GRAVITY, LINK_COM, LINK_INERTIA, LINK_MASS, N_JOINTS

_L1 = float(DH_A[2])
_L2 = float(DH_A[3])
_LC1 = float(LINK_COM[1, 0])
_LC2 = float(LINK_COM[2, 0])

_M1 = float(LINK_MASS[1])
_M2 = float(LINK_MASS[2])
_MW = float(LINK_MASS[3:].sum())

_I1_PITCH = float(LINK_INERTIA[1, 1, 1])
_I2_PITCH = float(LINK_INERTIA[2, 1, 1])
_I1_YAW = float(LINK_INERTIA[1, 2, 2])
_I2_YAW = float(LINK_INERTIA[2, 2, 2])
_BASE_YAW_INERTIA = float(LINK_INERTIA[0, 2, 2])

_A_YAW = _M1 * _LC1**2 + _I1_YAW
_B_PITCH = _M2 * _L1 * _LC2 + _MW * _L1 * _L2
_G1_COEFF = (_M1 * _LC1 + _M2 * _L1 + _MW * _L1) * GRAVITY
_G2_COEFF = (_M2 * _LC2 + _MW * _L2) * GRAVITY

_M11_CONST = _I1_PITCH + _I2_PITCH + _M1 * _LC1**2 + _M2 * (_L1**2 + _LC2**2) + _MW * (
    _L1**2 + _L2**2
)
_M12_CONST = _I2_PITCH + _M2 * _LC2**2 + _MW * _L2**2
_M22_CONST = _I2_PITCH + _M2 * _LC2**2 + _MW * _L2**2

_WRIST_DIAG = [
    float(LINK_INERTIA[3, 2, 2] + LINK_INERTIA[4, 2, 2] + LINK_INERTIA[5, 2, 2]),
    float(LINK_INERTIA[4, 1, 1] + LINK_INERTIA[5, 1, 1]),
    float(LINK_INERTIA[5, 2, 2]),
]
_WRIST_DIAG = [max(v, 1e-3) for v in _WRIST_DIAG]
_M00_EPSILON = 2e-2
_SPD_EPSILON = 1e-5


def _arm_radius(q2: torch.Tensor, q3: torch.Tensor) -> torch.Tensor:
    return _L1 * torch.cos(q2) + _LC2 * torch.cos(q2 + q3)


def _tip_radius(q2: torch.Tensor, q3: torch.Tensor) -> torch.Tensor:
    return _L1 * torch.cos(q2) + _L2 * torch.cos(q2 + q3)


def _d_m00(q2: torch.Tensor, q3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b = _arm_radius(q2, q3)
    tip = _tip_radius(q2, q3)
    db_dq2 = -_L1 * torch.sin(q2) - _LC2 * torch.sin(q2 + q3)
    db_dq3 = -_LC2 * torch.sin(q2 + q3)
    dt_dq2 = -_L1 * torch.sin(q2) - _L2 * torch.sin(q2 + q3)
    dt_dq3 = -_L2 * torch.sin(q2 + q3)

    d_q2 = (
        -_A_YAW * torch.sin(2.0 * q2)
        + 2.0 * _M2 * b * db_dq2
        - _I2_YAW * torch.sin(2.0 * (q2 + q3))
        + 2.0 * _MW * tip * dt_dq2
    )
    d_q3 = (
        2.0 * _M2 * b * db_dq3
        - _I2_YAW * torch.sin(2.0 * (q2 + q3))
        + 2.0 * _MW * tip * dt_dq3
    )
    return d_q2, d_q3


def arm_mass_matrix_base(q_arm: torch.Tensor) -> torch.Tensor:
    """q1-q3の閉形式近似質量行列。q_arm: (...,3) -> (...,3,3)。"""
    q2, q3 = q_arm[..., 1], q_arm[..., 2]
    b = _arm_radius(q2, q3)
    tip = _tip_radius(q2, q3)

    m00 = (
        _BASE_YAW_INERTIA
        + _A_YAW * torch.cos(q2) ** 2
        + _M2 * b**2
        + _I2_YAW * torch.cos(q2 + q3) ** 2
        + _MW * tip**2
        + _M00_EPSILON
    )
    c3 = torch.cos(q3)
    m11 = _M11_CONST + 2.0 * _B_PITCH * c3
    m12 = _M12_CONST + _B_PITCH * c3
    m22 = torch.full_like(m00, _M22_CONST)

    batch_shape = q_arm.shape[:-1]
    m = torch.zeros(*batch_shape, 3, 3, dtype=q_arm.dtype, device=q_arm.device)
    m[..., 0, 0] = m00
    m[..., 1, 1] = m11
    m[..., 1, 2] = m12
    m[..., 2, 1] = m12
    m[..., 2, 2] = m22
    eye = torch.eye(3, dtype=q_arm.dtype, device=q_arm.device)
    return m + _SPD_EPSILON * eye


def arm_bias_base(q_arm: torch.Tensor, qd_arm: torch.Tensor, g: float = GRAVITY) -> torch.Tensor:
    """q1-q3の閉形式bias=C(q,qd)qd+G(q)。shape (...,3)。"""
    q2, q3 = q_arm[..., 1], q_arm[..., 2]
    q1d, q2d, q3d = qd_arm[..., 0], qd_arm[..., 1], qd_arm[..., 2]

    d_m00_q2, d_m00_q3 = _d_m00(q2, q3)
    h = -_B_PITCH * torch.sin(q3)
    g_scale = g / GRAVITY
    g2 = g_scale * (_G1_COEFF * torch.cos(q2) + _G2_COEFF * torch.cos(q2 + q3))
    g3 = g_scale * (_G2_COEFF * torch.cos(q2 + q3))

    bias = torch.zeros_like(q_arm)
    bias[..., 0] = d_m00_q2 * q1d * q2d + d_m00_q3 * q1d * q3d
    bias[..., 1] = h * (2.0 * q2d * q3d + q3d**2) - 0.5 * d_m00_q2 * q1d**2 + g2
    bias[..., 2] = -h * q2d**2 - 0.5 * d_m00_q3 * q1d**2 + g3
    return bias


def mass_matrix_base(q: torch.Tensor) -> torch.Tensor:
    """6軸ベース質量行列。腕3軸と手首3軸の対角ブロックのみ。"""
    batch_shape = q.shape[:-1]
    m = torch.zeros(*batch_shape, N_JOINTS, N_JOINTS, dtype=q.dtype, device=q.device)
    m[..., :3, :3] = arm_mass_matrix_base(q[..., :3])
    wrist_diag = torch.as_tensor(_WRIST_DIAG, dtype=q.dtype, device=q.device)
    m[..., 3, 3] = wrist_diag[0]
    m[..., 4, 4] = wrist_diag[1]
    m[..., 5, 5] = wrist_diag[2]
    eye = torch.eye(N_JOINTS, dtype=q.dtype, device=q.device)
    return m + _SPD_EPSILON * eye


def bias_forces_base(q: torch.Tensor, q_dot: torch.Tensor, g: float = GRAVITY) -> torch.Tensor:
    """6軸ベースbias。手首3軸は0近似。"""
    bias = torch.zeros_like(q)
    bias[..., :3] = arm_bias_base(q[..., :3], q_dot[..., :3], g)
    return bias
