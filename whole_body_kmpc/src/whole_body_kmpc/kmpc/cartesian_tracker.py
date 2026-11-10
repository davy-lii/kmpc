"""Frozen-Jacobian Cartesian tracker (position + optional flange orientation).

Decision y_k = q_{k+1}-q_0; execute Δq_0 = y_0. Stage cost follows the
artificial terminal pose q_a = q_0 + y_{N-1}. The ball only enters the
terminal offset V_o. Tracking uses the full J. N>1 QP is the pair
(y_0, y_{N-1}); yaml N only sizes the terminal box. V_o waist is a
joint-space pull to DLS ŷ_w (deep drop: J1/J2, ŷ3 from the pitch
residual on the lower wall); J4 sits with the arm block on a
height-gated residual. Flange orientation (relative-quat axis-angle) is
an extra arm-block term; waist DLS stays 3D position. Identity
quaternion means ori-off. R = λ diag(M(q_0)). V_com pulls horizontal
CoM toward the support center. Deep drop when z* is below HOME-d and
still descending; pitch stays in ±θ_max. Ascent uses θ_des(e_z).
Idle-arm W_o springs to q_pref. J4 has a heading cost and yaw box.
ARRIVED stops on flange origin (and ori when tracking) or stalled progress.
"""

from __future__ import annotations

import math
from collections import deque
from typing import List, Optional, Sequence, Tuple

import numpy as np

from kmpc.constraint_builder import ConstraintBuilder
from kmpc.cost_builder import CostBuilder, DiagCost
from kmpc.horizon_qp import (
    assemble_endpoints,
    shift_warm_endpoints,
    y_bounds,
)
from kmpc.qp_solver import solve_dense_box_active, solve_dense_box_ineq
from kmpc.types import (
    EE_FRAME_LEFT,
    EE_FRAME_RIGHT,
    N_DOF,
    PALM_FRAME_RIGHT,
    SLICE_LEFT,
    SLICE_RIGHT,
    SLICE_WAIST,
)
from kmpc.upper_command import (
    DEFAULT_UPPER_COMMAND,
    TrackingState,
    UpperCommand,
)
from kmpc.urdf_kinematics import UrdfKinematics

_TARGET_JUMP_M = 0.02
_TARGET_JUMP_ORI_RAD = 0.17
_IDENTITY_QUAT_EPS = 1e-6
_IDX_YAW = 3
_SLICE_PITCH = slice(0, 3)
_N_QP = 2
_VO_ARM_FLOOR_M = 0.03
_TORSO_HOLD_DEG = 2.0
_TORSO_STEP_DEG = 0.08
_TORSO_DROP_M = 0.40
_DEEP_EXIT_M = 0.02
_PITCH_WALL_EPS_DEG = 0.5
_STALL_WALL_EZ_M = 0.05
_STALL_WALL_E0_M = 0.20
_STALL_J2_SLACK_DEG = 8.0


def clamp_mag(w: np.ndarray, d: float) -> np.ndarray:
    """Buss ClampMag: cap ||w|| at d, independent of ||e_0||."""
    vec = np.asarray(w, dtype=float).reshape(-1)
    cap = float(d)
    if cap <= 0.0:
        return np.zeros_like(vec)
    nrm = float(np.linalg.norm(vec))
    if nrm <= cap:
        return vec.copy()
    return (cap / nrm) * vec


def clamp_axis(val: float, cap: float) -> float:
    """Clip one axis to ±cap. cap<=0 leaves the value unchanged."""
    v = float(val)
    bound = float(cap)
    if bound <= 0.0:
        return v
    return float(np.clip(v, -bound, bound))


def clamp_arm_residual(raw: np.ndarray, d_xz: float, d_y: float) -> np.ndarray:
    """Per-axis arm residual: xz at D_xz, y at D_y (0 = no y cap)."""
    out = np.asarray(raw, dtype=float).reshape(3).copy()
    out[0] = clamp_axis(out[0], d_xz)
    out[1] = clamp_axis(out[1], d_y)
    out[2] = clamp_axis(out[2], d_xz)
    return out


def jparse_scale_task(jac: np.ndarray, err: np.ndarray, gamma: float) -> np.ndarray:
    """Shrink task error along weak singular directions (J-PARSE)."""
    vec = np.asarray(err, dtype=float).reshape(3)
    j = np.asarray(jac, dtype=float)
    g = float(gamma)
    if g <= 0.0 or j.size == 0:
        return vec.copy()
    u, s, _vt = np.linalg.svd(j, full_matrices=False)
    if s.size == 0 or float(s[0]) <= 1e-12:
        return vec.copy()
    scale = np.ones_like(s)
    tiny = s < (g * float(s[0]))
    scale[tiny] = s[tiny] / (g * float(s[0]))
    return (u * scale) @ (u.T @ vec)


def pad_tracking_jacobian(jac: np.ndarray, gamma: float) -> np.ndarray:
    """Floor singular values at γ σ_max (J-PARSE safety Jacobian).

    Tracking uses this on the full J. V_o pads the arm block (plus J4)
    and the J1–J3 pitch block separately (same floor, not mute).
    """
    j = np.asarray(jac, dtype=float)
    g = float(gamma)
    if g <= 0.0 or j.size == 0:
        return j
    u, s, vt = np.linalg.svd(j, full_matrices=False)
    if s.size == 0 or float(s[0]) <= 1e-12:
        return j
    s_pad = np.maximum(s, g * float(s[0]))
    return (u * s_pad) @ vt


def pad_waist_columns(jac: np.ndarray, gamma: float) -> np.ndarray:
    """J-PARSE pad only the waist 4 columns of a task Jacobian."""
    j = np.asarray(jac, dtype=float).copy()
    j[:, SLICE_WAIST] = pad_tracking_jacobian(j[:, SLICE_WAIST], gamma)
    return j


def damped_joint_ls(
    jac: np.ndarray,
    err: np.ndarray,
    lam: float = 0.05,
) -> np.ndarray:
    """Damped LS: ŷ = (JᵀJ + λ²I)⁻¹ Jᵀ e for a 3×n task Jacobian."""
    j = np.asarray(jac, dtype=float)
    if j.ndim != 2:
        j = j.reshape(3, -1)
    n_col = int(j.shape[1])
    e = np.asarray(err, dtype=float).reshape(3)
    damp = float(lam)
    if damp < 0.0:
        damp = 0.0
    gram = j.T @ j
    gram.flat[:: n_col + 1] += damp * damp
    rhs = j.T @ e
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.zeros(n_col, dtype=float)


def quat_is_identity_xyzw(quat: Optional[np.ndarray]) -> bool:
    """ROS (x,y,z,w): missing / zero / w=±1 with tiny xyz → ori-off."""
    if quat is None:
        return True
    arr = np.asarray(quat, dtype=float).reshape(-1)
    if arr.size < 4:
        return True
    x, y, z, w = (float(v) for v in arr[:4])
    nrm = math.sqrt(w * w + x * x + y * y + z * z)
    if nrm < 1e-12:
        return True
    return math.sqrt(x * x + y * y + z * z) / nrm < _IDENTITY_QUAT_EPS


def rot_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
    """Rotation matrix → ROS (x, y, z, w)."""
    r = np.asarray(rot, dtype=float).reshape(3, 3)
    t = float(r[0, 0] + r[1, 1] + r[2, 2])
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    nrm = math.sqrt(w * w + x * x + y * y + z * z)
    if nrm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return np.array([x / nrm, y / nrm, z / nrm, w / nrm])


def _wxyz_from_xyzw(quat: np.ndarray) -> Tuple[float, float, float, float]:
    x, y, z, w = (float(v) for v in np.asarray(quat, dtype=float).reshape(4))
    nrm = math.sqrt(w * w + x * x + y * y + z * z)
    if nrm < 1e-12:
        return 1.0, 0.0, 0.0, 0.0
    return w / nrm, x / nrm, y / nrm, z / nrm


def ori_error_xyzw(quat_des: np.ndarray, quat_curr: np.ndarray) -> np.ndarray:
    """e_ori = 2 sgn(w) v from Δq = q_des ⊗ q_curr^{-1} (ROS xyzw in)."""
    wd, xd, yd, zd = _wxyz_from_xyzw(quat_des)
    wc, xc, yc, zc = _wxyz_from_xyzw(quat_curr)
    w = wd * wc + xd * xc + yd * yc + zd * zc
    x = xd * wc - wd * xc - yd * zc + zd * yc
    y = yd * wc + xd * zc - wd * yc - zd * xc
    z = zd * wc - xd * yc + yd * xc - wd * zc
    sgn = 1.0 if w >= 0.0 else -1.0
    return 2.0 * sgn * np.array([x, y, z], dtype=float)


def damped_pitch_ls(
    j_pitch: np.ndarray,
    err: np.ndarray,
    lam: float = 0.05,
) -> np.ndarray:
    """Damped 3×3 LS: ŷ = (JᵀJ + λ²I)⁻¹ Jᵀ e on J1–J3 columns.

    HOME / fold-Z pitch axes are nearly collinear. Undamped J_w⁺ blows
    up ŷ_w and then the arm residual. λ=0.05 is m/rad, same units as σ(J).
    """
    return damped_joint_ls(
        np.asarray(j_pitch, dtype=float).reshape(3, 3), err, lam)


class CartesianTracker:
    def __init__(
        self,
        kin: UrdfKinematics,
        cost_builder: CostBuilder,
        constraints: ConstraintBuilder,
        q_pos: float = 400.0,
        q_offset: Optional[float] = None,
        w_idle_arm: float = 40.0,
        w_o_arm: float = 1.0,
        track_gamma: float = 0.1,
        vo_clamp_m: float = 0.12,
        vo_clamp_y_m: float = 0.40,
        vo_arm_gamma: float = 0.3,
        waist_gamma: float = 0.3,
        lambda_dls: float = 0.0,
        n_horizon: int = 15,
        q_limit_margin: float = 0.035,
        track_alpha: float = 0.20,
        arrived_err: float = 0.01,
        arrived_exit: Optional[float] = None,
        q_com: float = 40.0,
        com_margin: float = 0.025,
        com_xmin: float = -0.322,
        com_xmax: float = 0.138,
        com_ymin: float = -0.235,
        com_ymax: float = 0.235,
        q_torso: float = 250.0,
        q_syn: float = 80.0,
        q_syn_abs: float = 80.0,
        alpha_syn: float = 1.75,
        torso_pitch_max_deg: float = 32.0,
        torso_hold_deg: float = _TORSO_HOLD_DEG,
        torso_step_deg: float = _TORSO_STEP_DEG,
        torso_drop_m: float = _TORSO_DROP_M,
        q_yaw: float = 80.0,
        yaw_max_deg: float = 35.0,
        q_offset_y: float = 150.0,
        torso_k_per_m: Optional[float] = None,
        torso_reach_pitch_follow: bool = True,
        barrier_joints: Sequence[int] = (8, 15),
        barrier_warn_deg: float = 20.0,
        barrier_floor_deg: float = 5.0,
        barrier_alpha: float = 2.0,
        barrier_w_max_ratio: float = 50.0,
        arrived_stall_mm: float = 0.5,
        arrived_stall_ticks: int = 25,
        waist_pred_lambda: float = 0.05,
        vo_arm_floor_m: float = _VO_ARM_FLOOR_M,
        q_calib: Optional[np.ndarray] = None,
        q_ori: float = 20.0,
        vo_clamp_ori_rad: float = 0.35,
        arrived_ori_rad: float = 0.087,
        waist_lock_eps: float = 1e-5,
    ) -> None:
        self.kin = kin
        self._constraints = constraints
        self._cost: DiagCost = cost_builder.expand()
        self.q_pos = float(q_pos)
        self.q_offset = float(q_pos if q_offset is None else q_offset)
        self.q_offset_y = float(q_offset_y)
        self.w_idle_arm = float(w_idle_arm)
        self.w_o_arm = float(w_o_arm)
        self.track_gamma = float(track_gamma)
        self.vo_clamp_m = float(vo_clamp_m)
        self.vo_clamp_y_m = float(vo_clamp_y_m)
        self.vo_arm_gamma = float(vo_arm_gamma)
        self.waist_gamma = float(waist_gamma)
        self.lambda_dls = float(lambda_dls)
        self.n_horizon = int(n_horizon)
        self.q_limit_margin = float(q_limit_margin)
        if self.n_horizon < 1:
            raise ValueError('n_horizon must be >= 1')
        if self.q_limit_margin < 0.0:
            raise ValueError('limit margins must be >= 0')
        if min(self.q_offset, self.q_offset_y, self.w_o_arm, q_com, q_torso,
               q_syn, q_syn_abs, q_yaw) < 0.0:
            raise ValueError('weights must be >= 0')
        if self.track_gamma < 0.0 or self.vo_arm_gamma < 0.0 or self.waist_gamma < 0.0:
            raise ValueError('track_gamma / vo_arm_gamma / waist_gamma must be >= 0')
        if self.vo_clamp_m < 0.0 or self.vo_clamp_y_m < 0.0:
            raise ValueError('vo_clamp_m / vo_clamp_y_m must be >= 0')
        self.q_ori = float(q_ori)
        self.vo_clamp_ori_rad = float(vo_clamp_ori_rad)
        self.arrived_ori_rad = float(arrived_ori_rad)
        if self.q_ori < 0.0 or self.vo_clamp_ori_rad < 0.0:
            raise ValueError('q_ori / vo_clamp_ori_rad must be >= 0')
        if self.arrived_ori_rad < 0.0:
            raise ValueError('arrived_ori_rad must be >= 0')
        self.vo_arm_floor_m = float(vo_arm_floor_m)
        if self.vo_arm_floor_m < 0.0:
            raise ValueError('vo_arm_floor_m must be >= 0')
        self.track_alpha = float(track_alpha)
        if not (0.0 < self.track_alpha <= 1.0):
            raise ValueError('need 0 < track_alpha <= 1')
        self.arrived_err = float(arrived_err)
        self.arrived_exit = float(
            0.02 if arrived_exit is None else arrived_exit)
        if self.arrived_err < 0.0 or self.arrived_exit < self.arrived_err:
            raise ValueError('need 0 <= arrived_err <= arrived_exit')
        self.q_com = float(q_com)
        self.com_margin = float(com_margin)
        if self.com_margin < 0.0:
            raise ValueError('com_margin must be >= 0')
        self.com_xmin = float(com_xmin)
        self.com_xmax = float(com_xmax)
        self.com_ymin = float(com_ymin)
        self.com_ymax = float(com_ymax)
        self.c_sup = np.array(
            [0.5 * (self.com_xmin + self.com_xmax),
             0.5 * (self.com_ymin + self.com_ymax)],
            dtype=float)
        self.q_torso = float(q_torso)
        self.q_syn = float(q_syn)
        self.q_syn_abs = float(q_syn_abs)
        self.alpha_syn = float(alpha_syn)
        self.q_yaw = float(q_yaw)
        self.torso_pitch_max = float(np.deg2rad(torso_pitch_max_deg))
        hold_deg = float(torso_hold_deg)
        step_deg = float(torso_step_deg)
        if hold_deg < 0.0 or step_deg < 0.0:
            raise ValueError('torso_hold_deg / torso_step_deg must be >= 0')
        self.torso_hold = float(np.deg2rad(hold_deg))
        self.torso_step = float(np.deg2rad(step_deg))
        self.torso_drop_m = float(torso_drop_m)
        if self.torso_drop_m < 0.0:
            raise ValueError('torso_drop_m must be >= 0')
        self._deep_exit_m = _DEEP_EXIT_M
        self.yaw_max = float(np.deg2rad(yaw_max_deg))
        if torso_k_per_m is None or float(torso_k_per_m) <= 0.0:
            self.torso_k_per_m = (
                self.torso_pitch_max / 0.80 if self.torso_pitch_max > 0.0 else 0.0)
        else:
            self.torso_k_per_m = float(torso_k_per_m)
        # 20261015 第二刀：下降方向 θ_des 跟 κ·e_z 走（深下降分支不动）。
        self.torso_reach_pitch_follow = bool(torso_reach_pitch_follow)
        # 20261015 第一刀：腕状态屏障（镜像 C++ apply_joint_barrier）。
        self.barrier_joints = [int(j) for j in barrier_joints]
        self.barrier_warn_rad = float(np.deg2rad(barrier_warn_deg))
        self.barrier_floor_rad = float(np.deg2rad(barrier_floor_deg))
        self.barrier_alpha = max(float(barrier_alpha), 0.0)
        self.barrier_w_max_ratio = float(barrier_w_max_ratio)
        self.arrived_stall_m = max(float(arrived_stall_mm), 0.0) * 1e-3
        self.arrived_stall_ticks = max(int(arrived_stall_ticks), 1)
        self.waist_lock_eps = float(waist_lock_eps)
        if self.waist_lock_eps <= 0.0:
            raise ValueError('waist_lock_eps must be > 0')
        self.waist_pred_lambda = max(float(waist_pred_lambda), 0.0)
        self._e0_hist: deque = deque(maxlen=self.arrived_stall_ticks)
        self._e0_prev = 0.0
        self._j_torso = np.zeros((1, N_DOF), dtype=float)
        self._j_torso[0, 0:3] = 1.0
        self._h_syn = np.zeros((1, N_DOF), dtype=float)
        self._h_syn[0, 0] = self.alpha_syn
        self._h_syn[0, 1] = 1.0
        q_cal = np.zeros(N_DOF) if q_calib is None else np.asarray(
            q_calib, dtype=float).reshape(N_DOF)
        self._q_yaw_pref = float(q_cal[_IDX_YAW])
        self._qo = np.diag(
            [self.q_offset, self.q_offset_y, self.q_offset]).astype(float)
        self.r_lambda = float(self.kin.inertia_lambda(q_cal, self._cost.R[0]))
        self._z_home = float(self.kin.fk_pos(q_cal, EE_FRAME_LEFT)[2])
        self._z_star = self._z_home
        self._deep_on = False
        self._dq_prev = np.zeros(N_DOF, dtype=float)
        self._n_qp = 1 if self.n_horizon <= 1 else _N_QP
        nv = N_DOF * self._n_qp
        self._p_work = np.zeros((nv, nv))
        self._g_work = np.zeros(nv)
        self._y_prev = np.zeros(nv)
        self._y_last = np.zeros((self._n_qp, N_DOF))
        self._y_w_pred = np.zeros(4, dtype=float)
        self._pitch_pin: Optional[float] = None
        self.last_e0 = 0.0
        self.last_e_ori = np.zeros(6, dtype=float)
        self.last_ori_on = False
        self.last_arrived = False
        self.last_com = np.zeros(3, dtype=float)
        self.last_pitch = 0.0
        self._prev_left: Optional[np.ndarray] = None
        self._prev_right: Optional[np.ndarray] = None
        self._e_ori_prev_left = np.zeros(3, dtype=float)
        self._e_ori_prev_right = np.zeros(3, dtype=float)
        self._upper = DEFAULT_UPPER_COMMAND.copy()
        self._tracking_state_prev = TrackingState.TRACKING

    def set_upper_commands(self, cmd: Optional[UpperCommand] = None) -> None:
        self._upper = (
            DEFAULT_UPPER_COMMAND.copy()
            if cmd is None
            else cmd.copy())

    def reset_warm(self) -> None:
        """Clear hot-start after a large cost / target discontinuity."""
        self._clear_hold_warm()
        self._e0_hist.clear()
        self._e0_prev = 0.0

    def _effective_dq_max(self) -> np.ndarray:
        dq_max = np.asarray(
            self._constraints.limits.dq_max, dtype=float).copy()
        if self._upper.waist_locked:
            dq_max[SLICE_WAIST] = self.waist_lock_eps
        return dq_max

    def _on_tracking_state_changed(self) -> None:
        self.reset_warm()

    def reset(self) -> None:
        self._dq_prev[:] = 0.0
        self._y_prev[:] = 0.0
        self.last_e0 = 0.0
        self.last_e_ori[:] = 0.0
        self.last_ori_on = False
        self.last_arrived = False
        self.last_com[:] = 0.0
        self.last_pitch = 0.0
        self._y_last[:] = 0.0
        self._y_w_pred[:] = 0.0
        self._pitch_pin = None
        self._prev_left = None
        self._prev_right = None
        self._e_ori_prev_left[:] = 0.0
        self._e_ori_prev_right[:] = 0.0
        self._e0_hist.clear()
        self._e0_prev = 0.0
        self._z_star = self._z_home
        self._deep_on = False
        self._upper = DEFAULT_UPPER_COMMAND.copy()
        self._tracking_state_prev = TrackingState.TRACKING

    def inertia_r(self, q0: np.ndarray) -> np.ndarray:
        mdiag = self.kin.mass_matrix_diag(np.asarray(q0, dtype=float).reshape(N_DOF))
        return np.maximum(self.r_lambda * mdiag, 1e-8)

    def compute(
        self,
        q0: np.ndarray,
        q_pref: np.ndarray,
        target_left: Optional[np.ndarray] = None,
        target_right: Optional[np.ndarray] = None,
        frame_left: str = EE_FRAME_LEFT,
        frame_right: str = EE_FRAME_RIGHT,
        mode: str = '',
        quat_left: Optional[np.ndarray] = None,
        quat_right: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (Δq_0, ee_error [eLxyz, eRxyz]). Missing arm error is 0.

        quat_* is ROS (x,y,z,w). Identity / None is ori-off. A length-7
        target is (xyz, quat).
        """
        del mode
        q0 = np.asarray(q0, dtype=float).reshape(N_DOF)
        q_pref = np.asarray(q_pref, dtype=float).reshape(N_DOF)
        state = self._upper.tracking_state
        if state != self._tracking_state_prev:
            self._on_tracking_state_changed()
            self._tracking_state_prev = state
        calibrating = state == TrackingState.CALIBRATING
        xyz_left, quat_left = self._split_target(target_left, quat_left)
        xyz_right, quat_right = self._split_target(target_right, quat_right)
        if calibrating:
            xyz_left = None
            xyz_right = None
            quat_left = None
            quat_right = None
        w = self._cost.W.copy()
        err = np.zeros(6, dtype=float)
        e_ori6 = np.zeros(6, dtype=float)
        p_com, j_com, mdiag = self.kin.inertial_quantities(q0)
        r = np.maximum(self.r_lambda * mdiag, 1e-8)
        self.last_com = p_com
        self.last_pitch = float(q0[0] + q0[1] + q0[2])

        tasks: Sequence[Tuple[Optional[np.ndarray], Optional[np.ndarray], str, slice]] = (
            (xyz_left, quat_left, frame_left, SLICE_LEFT),
            (xyz_right, quat_right, frame_right, SLICE_RIGHT),
        )
        active_slices: List[slice] = []
        jac_terms: List[Tuple] = []
        best_n = -1.0
        z_star = self._z_star
        any_ori = False
        e_ori_left = np.zeros(3, dtype=float)
        e_ori_right = np.zeros(3, dtype=float)
        for xyz, quat, frame, slc in tasks:
            if xyz is None:
                continue
            active_slices.append(slc)
            xyz = np.asarray(xyz, dtype=float).reshape(3)
            if frame in (EE_FRAME_RIGHT, PALM_FRAME_RIGHT, 'RightArm_Z_Link7'):
                link = EE_FRAME_RIGHT
            else:
                link = EE_FRAME_LEFT
            t_now = self.kin.fk(q0, link)
            p_now = t_now[:3, 3]
            e = xyz - p_now
            side = 'left' if slc == SLICE_LEFT else 'right'
            if side == 'left':
                err[0:3] = e
            else:
                err[3:6] = e
            jac6 = self.kin.jacobian_spatial(q0, link)
            jac_v = jac6[:3]
            jac_w = jac6[3:]
            jac_v_pad = pad_tracking_jacobian(jac_v, self.track_gamma)
            ori_on = self.q_ori > 0.0 and not quat_is_identity_xyzw(quat)
            e_ori = np.zeros(3, dtype=float)
            jac_w_pad = jac_w
            if ori_on:
                any_ori = True
                q_curr = rot_to_quat_xyzw(t_now[:3, :3])
                e_ori = ori_error_xyzw(np.asarray(quat, dtype=float), q_curr)
                jac_w_pad = pad_tracking_jacobian(jac_w, self.track_gamma)
                if side == 'left':
                    e_ori6[0:3] = e_ori
                    e_ori_left = e_ori
                else:
                    e_ori6[3:6] = e_ori
                    e_ori_right = e_ori
            jac_terms.append(
                (side, jac_v, jac_v_pad, e, jac_w, jac_w_pad, e_ori, ori_on))
            nrm = float(np.linalg.norm(e))
            if nrm > best_n:
                best_n = nrm
                z_star = float(xyz[2])
        if best_n >= 0.0:
            self._z_star = z_star
        self.last_e_ori = e_ori6
        self.last_ori_on = any_ori

        jumped = self._targets_jumped(xyz_left, xyz_right)
        jumped = self._ori_jumped(e_ori_left, e_ori_right) or jumped
        if jumped:
            self._pitch_pin = None
            self._y_prev[:] = 0.0
            self._dq_prev[:] = 0.0

        if len(active_slices) == 1 and not calibrating:
            idle = SLICE_RIGHT if active_slices[0] == SLICE_LEFT else SLICE_LEFT
            w = w.copy()
            w[idle] = w[idle] + self.w_idle_arm

        w_o = self._offset_weights(active_slices, calibrating)
        g_wo = self._idle_offset_grad(
            q0, q_pref, w_o, active_slices, calibrating)

        if self.n_horizon <= 1:
            dq = self._solve_n1(q0, w_o, g_wo, r, jac_terms, p_com, j_com)
            self._y_last = dq.reshape(1, N_DOF)
            self._y_w_pred = dq[SLICE_WAIST].copy()
            self._y_w_pred[_IDX_YAW] = 0.0
        else:
            dq = self._solve_horizon(
                q0, w, w_o, g_wo, r, jac_terms, p_com, j_com)
        e0 = self._active_norm(err, xyz_left, xyz_right)
        self.last_e0 = e0
        if calibrating:
            self.last_arrived = False
        else:
            ori_ok = self._ori_arrived_ok(jac_terms)
            dq = self._apply_arrived(
                dq, e0, jumped, q0=q0, e_z=self._working_ez(jac_terms),
                ori_ok=ori_ok)
        self._dq_prev = dq
        return dq, err

    @staticmethod
    def _split_target(
        target: Optional[np.ndarray],
        quat: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if target is None:
            return None, quat
        arr = np.asarray(target, dtype=float).reshape(-1)
        if arr.size == 3:
            return arr.copy(), quat
        if arr.size == 7:
            q_in = arr[3:7].copy() if quat is None else quat
            return arr[:3].copy(), q_in
        raise ValueError('target must be xyz (3,) or pose (7,)')

    def _offset_weights(
        self,
        active_slices: Sequence[slice],
        calibrating: bool = False,
    ) -> np.ndarray:
        w_o = np.full(N_DOF, self.w_o_arm, dtype=float)
        w_o[SLICE_WAIST] = 0.0
        if calibrating:
            w_o[SLICE_LEFT] = w_o[SLICE_LEFT] + self.w_idle_arm
            w_o[SLICE_RIGHT] = w_o[SLICE_RIGHT] + self.w_idle_arm
        elif len(active_slices) == 1:
            idle = SLICE_RIGHT if active_slices[0] == SLICE_LEFT else SLICE_LEFT
            w_o[idle] = w_o[idle] + self.w_idle_arm
        return w_o

    def _idle_offset_grad(
        self,
        q0: np.ndarray,
        q_pref: np.ndarray,
        w_o: np.ndarray,
        active_slices: Sequence[slice],
        calibrating: bool = False,
    ) -> np.ndarray:
        """Linear term of ½‖y_{N-1}+q_0-q_pref‖² on idle arm(s)."""
        g = np.zeros(N_DOF, dtype=float)
        q0v = np.asarray(q0, dtype=float).reshape(N_DOF)
        qpref = np.asarray(q_pref, dtype=float).reshape(N_DOF)
        w = np.asarray(w_o, dtype=float).reshape(N_DOF)
        if calibrating:
            for slc in (SLICE_LEFT, SLICE_RIGHT):
                g[slc] = w[slc] * (q0v[slc] - qpref[slc])
            return g
        if len(active_slices) != 1:
            return g
        idle = SLICE_RIGHT if active_slices[0] == SLICE_LEFT else SLICE_LEFT
        g[idle] = w[idle] * (q0v[idle] - qpref[idle])
        return g

    def _deep_z(self) -> float:
        return self._z_home - self.torso_drop_m

    def _pitch_at_lower_wall(self, pitch0: float) -> bool:
        eps = float(np.deg2rad(_PITCH_WALL_EPS_DEG))
        return float(pitch0) <= (-self.torso_pitch_max + eps)

    def _is_deep_drop(self, e_z: float, z_star: Optional[float] = None) -> bool:
        """Low-ball primitive: z* below HOME-d and still descending."""
        zs = self._z_star if z_star is None else float(z_star)
        self._refresh_deep_drop(float(e_z), zs)
        return self._deep_on

    def _refresh_deep_drop(self, e_z: float, z_star: float) -> None:
        """Latch deep drop; exit if z* > z_deep+δ or e_z>0."""
        ez = float(e_z)
        zs = float(z_star)
        z_deep = self._deep_z()
        if ez > 0.0 or zs > z_deep + self._deep_exit_m:
            self._deep_on = False
        elif zs < z_deep and ez < 0.0:
            self._deep_on = True

    def _damped_waist_yhat(
        self,
        j_pitch: np.ndarray,
        e_cap: np.ndarray,
        e_z: float,
    ) -> np.ndarray:
        """Deep drop: 2-col DLS, ŷ3=0, ŷ2≤0. Else 3-col DLS."""
        lam = self.waist_pred_lambda
        if self._is_deep_drop(e_z):
            y12 = damped_joint_ls(
                np.asarray(j_pitch, dtype=float).reshape(3, 3)[:, :2],
                e_cap, lam)
            return np.array(
                [float(y12[0]), min(float(y12[1]), 0.0), 0.0], dtype=float)
        return damped_pitch_ls(j_pitch, e_cap, lam)

    def _clip_yhat_pitch_sum(
        self,
        y_hat: np.ndarray,
        lo: float,
        hi: float,
        keep_j3_zero: bool,
        dump_j3: bool = False,
    ) -> np.ndarray:
        out = np.asarray(y_hat, dtype=float).reshape(3).copy()
        ps = float(out[0] + out[1] + out[2])
        if ps < lo:
            delta = lo - ps
        elif ps > hi:
            delta = hi - ps
        else:
            return out
        if keep_j3_zero and dump_j3 and ps < lo:
            out[1] = min(float(out[1]), 0.0)
            out[2] = lo - out[0] - out[1]
            return out
        if keep_j3_zero:
            out[0] += 0.5 * delta
            out[1] += 0.5 * delta
            out[2] = 0.0
        else:
            out += delta / 3.0
        return out

    def _waist_yhat_metric(self, e_z: float) -> np.ndarray:
        """Joint-space M on ŷ_w. Deep drop: m2>m1; wall slide uses m3=m1."""
        m1 = self.q_offset
        if self._is_deep_drop(e_z):
            m3 = m1 if self._pitch_at_lower_wall(self.last_pitch) else 20.0 * m1
            return np.array([m1, 2.0 * m1, m3], dtype=float)
        return np.array([m1, m1, m1], dtype=float)

    def _arm_residual_cap(self, e_w_z: float, e0_z: float, e_z: float) -> float:
        """Deep drop: xz floor + η·(D_xz-floor). Else full D_xz."""
        d_xz = self.vo_clamp_m
        if not self._is_deep_drop(e_z):
            return d_xz
        d_floor = min(self.vo_arm_floor_m, d_xz) if d_xz > 0.0 else 0.0
        eta = float(np.clip(
            abs(float(e_w_z)) / max(abs(float(e0_z)), 1e-3), 0.0, 1.0))
        return d_floor + (d_xz - d_floor) * eta

    def _assemble_vo(
        self,
        jac_f: np.ndarray,
        err: np.ndarray,
        pitch_lo: Optional[np.ndarray] = None,
        pitch_hi: Optional[np.ndarray] = None,
        e_z: Optional[float] = None,
        pitch_sum_lo: Optional[float] = None,
        pitch_sum_hi: Optional[float] = None,
        jac_omega: Optional[np.ndarray] = None,
        e_ori: Optional[np.ndarray] = None,
        ori_on: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """V_o: deep drop joint-space ||y^{1:3}-ŷ||_M; else cartesian J_w ŷ."""
        jac = np.asarray(jac_f, dtype=float)
        e = np.asarray(err, dtype=float).reshape(3)
        ez = float(e[2] if e_z is None else e_z)
        deep = self._is_deep_drop(ez)
        e_cap = clamp_arm_residual(e, self.vo_clamp_m, self.vo_clamp_y_m)
        j_pitch_raw = jac[:, _SLICE_PITCH]
        y_hat = self._damped_waist_yhat(j_pitch_raw, e_cap, ez)
        if pitch_lo is not None and pitch_hi is not None:
            y_hat = np.clip(
                y_hat,
                np.asarray(pitch_lo, dtype=float).reshape(3),
                np.asarray(pitch_hi, dtype=float).reshape(3))
        if pitch_sum_lo is not None and pitch_sum_hi is not None:
            y_hat = self._clip_yhat_pitch_sum(
                y_hat, float(pitch_sum_lo), float(pitch_sum_hi), deep,
                dump_j3=deep and self._pitch_at_lower_wall(self.last_pitch))
        j_w_raw = jac.copy()
        j_w_raw[:, SLICE_LEFT] = 0.0
        j_w_raw[:, SLICE_RIGHT] = 0.0
        j_w_raw[:, _IDX_YAW] = 0.0
        y_w = np.zeros(N_DOF, dtype=float)
        y_w[_SLICE_PITCH] = y_hat
        e_w = j_w_raw @ y_w
        e_tilde_raw = e.copy()
        e_tilde_raw[2] = e[2] - e_w[2]
        d_xz = self._arm_residual_cap(float(e_w[2]), float(e[2]), ez)
        j_a = jac.copy()
        j_a[:, _SLICE_PITCH] = 0.0
        e_tilde = clamp_arm_residual(e_tilde_raw, d_xz, self.vo_clamp_y_m)
        e_tilde = jparse_scale_task(j_a, e_tilde, self.vo_arm_gamma)
        j_a = pad_tracking_jacobian(j_a, self.vo_arm_gamma)
        qo = self._qo
        if deep:
            m_hat = self._waist_yhat_metric(ez)
            h_vo = np.zeros((N_DOF, N_DOF), dtype=float)
            g_vo = np.zeros(N_DOF, dtype=float)
            for i in range(3):
                h_vo[i, i] = m_hat[i]
                g_vo[i] = -m_hat[i] * y_hat[i]
        else:
            j_w = j_w_raw.copy()
            j_w[:, _SLICE_PITCH] = pad_tracking_jacobian(
                j_w[:, _SLICE_PITCH], self.waist_gamma)
            h_vo = j_w.T @ qo @ j_w
            g_vo = -(j_w.T @ (qo @ e_w))
        h_vo = h_vo + j_a.T @ qo @ j_a
        g_vo = g_vo - (j_a.T @ (qo @ e_tilde))
        if ori_on and self.q_ori > 0.0 and jac_omega is not None and e_ori is not None:
            j_aw = np.asarray(jac_omega, dtype=float).copy()
            j_aw[:, _SLICE_PITCH] = 0.0
            e_w = clamp_mag(np.asarray(e_ori, dtype=float).reshape(3),
                            self.vo_clamp_ori_rad)
            e_w = jparse_scale_task(j_aw, e_w, self.vo_arm_gamma)
            j_aw = pad_tracking_jacobian(j_aw, self.vo_arm_gamma)
            qo_w = self.q_ori * np.eye(3)
            h_vo = h_vo + j_aw.T @ qo_w @ j_aw
            g_vo = g_vo - (j_aw.T @ (qo_w @ e_w))
        return h_vo, g_vo

    def _com_terminal(
        self,
        p_com: np.ndarray,
        j_com: np.ndarray,
        n_vars: int,
        y_block: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Terminal V_com Hessian/gradient (18) and A y >= b on the y block.

        Cost uses J-PARSE-padded waist columns; the support box uses the
        true J_com so the polygon stays physically honest.
        """
        j_xy = np.asarray(j_com[:2], dtype=float)
        j_xy_cost = pad_waist_columns(j_xy, self.waist_gamma)
        p_xy = np.asarray(p_com[:2], dtype=float)
        qc = self.q_com
        h_com = qc * (j_xy_cost.T @ j_xy_cost)
        g_com = -qc * (j_xy_cost.T @ (self.c_sup - p_xy))
        lo_xy = np.array(
            [self.com_xmin + self.com_margin, self.com_ymin + self.com_margin])
        hi_xy = np.array(
            [self.com_xmax - self.com_margin, self.com_ymax - self.com_margin])
        a_term = np.vstack([j_xy, -j_xy])
        b_term = np.concatenate([lo_xy - p_xy, p_xy - hi_xy])
        a_full = np.zeros((4, n_vars))
        sl = slice(y_block * N_DOF, (y_block + 1) * N_DOF)
        a_full[:, sl] = a_term
        return h_com, g_com, a_full, b_term

    def _stage_w(self, w_diag: np.ndarray, e_z: float = 0.0) -> np.ndarray:
        """Path metric: diag(W)+Q_yaw on J4; W_syn off on a deep drop."""
        w = np.diag(np.asarray(w_diag, dtype=float).reshape(N_DOF))
        if self.q_yaw > 0.0:
            w[_IDX_YAW, _IDX_YAW] = w[_IDX_YAW, _IDX_YAW] + self.q_yaw
        if self.q_syn <= 0.0:
            return w
        return w + self.q_syn * (self._h_syn.T @ self._h_syn)

    def _torso_synergy_terminal(
        self,
        q0: np.ndarray,
        e_z: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Soft pitch except on a deep drop (hard pin). J4 heading always on."""
        h = np.zeros((N_DOF, N_DOF), dtype=float)
        g = np.zeros(N_DOF, dtype=float)
        ez = float(e_z)
        if self.q_torso > 0.0:
            pitch0 = float(q0[0] + q0[1] + q0[2])
            theta_des = float(np.clip(
                pitch0, -self.torso_pitch_max, self.torso_pitch_max))
            if ez > 0.0:
                theta_des = float(np.clip(
                    max(theta_des, self.torso_k_per_m * ez),
                    -self.torso_pitch_max,
                    self.torso_pitch_max))
            elif (self.torso_reach_pitch_follow and ez < 0.0
                    and not self._is_deep_drop(ez)):
                # 20261015 第二刀：下降方向 θ_des 跟 κ·e_z 走（镜像 C++）。
                theta_des = float(np.clip(
                    min(theta_des, self.torso_k_per_m * ez),
                    -self.torso_pitch_max,
                    self.torso_pitch_max))
            h = h + self.q_torso * (self._j_torso.T @ self._j_torso)
            g = g + (self.q_torso * (pitch0 - theta_des)) * self._j_torso.reshape(
                N_DOF)
        if self.q_yaw > 0.0:
            yaw0 = float(q0[_IDX_YAW] - self._q_yaw_pref)
            h[_IDX_YAW, _IDX_YAW] = h[_IDX_YAW, _IDX_YAW] + self.q_yaw
            g[_IDX_YAW] = g[_IDX_YAW] + self.q_yaw * yaw0
        return h, g

    def _apply_joint_barrier(
        self,
        p: np.ndarray,
        g: np.ndarray,
        q0: np.ndarray,
        r: np.ndarray,
        two_blocks: bool,
    ) -> None:
        """20261015 第一刀：腕状态屏障（镜像 C++ apply_joint_barrier）。

        d = 距近侧限位；进警戒区后该轴价格按分段二次上抬（封顶
        w_max_ratio×r_i），弹簧打在状态 q0+y 上往 d_warn 边界回拉。
        只压 y0 块的 g（执行步）；two_blocks 时 yN 块只加 P 防规划推深。
        回拉目标单拍限幅 0.5×cmd_reset_rad=0.075，防 re-seed 清热启动。
        """
        if (self.barrier_alpha <= 0.0
                or self.barrier_warn_rad <= self.barrier_floor_rad):
            return
        span = self.barrier_warn_rad - self.barrier_floor_rad
        pull_max = 0.5 * 0.15  # 0.5×cmd_reset_rad，与 C++ params 默认一致
        lim = self._constraints.limits
        q_min = np.asarray(lim.q_min, dtype=float).reshape(N_DOF)
        q_max = np.asarray(lim.q_max, dtype=float).reshape(N_DOF)
        r_arr = np.asarray(r, dtype=float).reshape(N_DOF)
        for idx in self.barrier_joints:
            if idx < 0 or idx >= N_DOF:
                continue
            d_hi = float(q_max[idx] - q0[idx])
            d_lo = float(q0[idx] - q_min[idx])
            d = min(d_hi, d_lo)
            if d >= self.barrier_warn_rad:
                continue
            s = (self.barrier_warn_rad - d) / span
            w_b = min(self.barrier_alpha * s * s,
                      self.barrier_w_max_ratio * float(r_arr[idx]))
            need = self.barrier_warn_rad - d
            y_pull = float(np.clip(
                -need if d_hi < d_lo else need, -pull_max, pull_max))
            p[idx, idx] += w_b
            g[idx] += -w_b * y_pull
            if two_blocks:
                p[N_DOF + idx, N_DOF + idx] += w_b

    @staticmethod
    def _working_ez(
        jac_terms: Sequence[Tuple],
    ) -> float:
        """e_z of the largest-norm TCP error (single-ball field case)."""
        best_n = -1.0
        best_z = 0.0
        for term in jac_terms:
            vec = np.asarray(term[3], dtype=float).reshape(3)
            nrm = float(np.linalg.norm(vec))
            if nrm > best_n:
                best_n = nrm
                best_z = float(vec[2])
        return best_z

    def _update_pitch_pin(self, q0: np.ndarray, e_z: float) -> None:
        """Pin absolute pitch to the pose that entered a deep drop."""
        if not self._is_deep_drop(e_z):
            self._pitch_pin = None
            return
        if self._pitch_pin is None:
            self._pitch_pin = float(q0[0] + q0[1] + q0[2])

    def _pitch_sum_bounds(
        self,
        q0: np.ndarray,
        e_z: float,
        for_step: bool = False,
    ) -> Tuple[float, float]:
        """Stay-feasible 1^T y. Deep drop: |θ-θ_pin|≤ε_θ from entry, not per tick."""
        del for_step
        pitch0 = float(q0[0] + q0[1] + q0[2])
        lo = min(-self.torso_pitch_max - pitch0, 0.0)
        hi = max(self.torso_pitch_max - pitch0, 0.0)
        return lo, hi

    def _pitch_ineq(
        self,
        q0: np.ndarray,
        n_vars: int,
        n_horizon: int,
        e_z: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stay-feasible pitch box on each QP node: A y >= b."""
        if self.torso_pitch_max <= 0.0:
            return np.zeros((0, n_vars)), np.zeros(0)
        j = self._j_torso.reshape(N_DOF)
        n_h = int(n_horizon)
        a_full = np.zeros((2 * n_h, n_vars))
        b_term = np.empty(2 * n_h)
        for k in range(n_h):
            lo, hi = self._pitch_sum_bounds(q0, e_z)
            sl = slice(k * N_DOF, (k + 1) * N_DOF)
            a_full[2 * k, sl] = j
            b_term[2 * k] = lo
            a_full[2 * k + 1, sl] = -j
            b_term[2 * k + 1] = -hi
        return a_full, b_term

    def _yaw_ineq(
        self,
        q0: np.ndarray,
        n_vars: int,
        n_horizon: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stay-feasible yaw box on every y_k: A y >= b."""
        if self.yaw_max <= 0.0:
            return np.zeros((0, n_vars)), np.zeros(0)
        yaw0 = float(q0[_IDX_YAW] - self._q_yaw_pref)
        lo = min(-self.yaw_max - yaw0, 0.0)
        hi = max(self.yaw_max - yaw0, 0.0)
        j = np.zeros(N_DOF, dtype=float)
        j[_IDX_YAW] = 1.0
        n_h = int(n_horizon)
        a_full = np.zeros((2 * n_h, n_vars))
        b_term = np.empty(2 * n_h)
        for k in range(n_h):
            sl = slice(k * N_DOF, (k + 1) * N_DOF)
            a_full[2 * k, sl] = j
            b_term[2 * k] = lo
            a_full[2 * k + 1, sl] = -j
            b_term[2 * k + 1] = -hi
        return a_full, b_term

    @staticmethod
    def _stack_ineq(
        *pairs: Tuple[np.ndarray, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        a_parts: List[np.ndarray] = []
        b_parts: List[np.ndarray] = []
        for a_mat, b_vec in pairs:
            if np.size(a_mat) == 0:
                continue
            a_parts.append(np.asarray(a_mat, dtype=float))
            b_parts.append(np.asarray(b_vec, dtype=float).reshape(-1))
        if not a_parts:
            return np.zeros((0, 0)), np.zeros(0)
        return np.vstack(a_parts), np.concatenate(b_parts)

    def _solve_n1(
        self,
        q0: np.ndarray,
        w_o: np.ndarray,
        g_wo: np.ndarray,
        r: np.ndarray,
        jac_terms: Sequence[Tuple],
        p_com: np.ndarray,
        j_com: np.ndarray,
    ) -> np.ndarray:
        """N=1: tracking vanishes; V_o + W_o + R + S + V_com."""
        p = np.diag(r + self._cost.S + w_o)
        p.flat[:: p.shape[0] + 1] += 1e-8
        g = -self._cost.S * self._dq_prev + np.asarray(g_wo, dtype=float).reshape(
            N_DOF)
        box = self._constraints.box_dq(q0, dq_max=self._effective_dq_max())
        pitch_lo = box.lb[_SLICE_PITCH]
        pitch_hi = box.ub[_SLICE_PITCH]
        e_z = self._working_ez(jac_terms)
        self._update_pitch_pin(q0, e_z)
        sum_lo, sum_hi = self._pitch_sum_bounds(q0, e_z)
        for term in jac_terms:
            jac, e = term[1], term[3]
            jac_w, e_ori, ori_on = term[4], term[6], term[7]
            h_vo, g_vo = self._assemble_vo(
                jac, e, pitch_lo, pitch_hi, e_z=e_z,
                pitch_sum_lo=sum_lo, pitch_sum_hi=sum_hi,
                jac_omega=jac_w, e_ori=e_ori, ori_on=ori_on)
            p = p + h_vo
            g = g + g_vo
        h_com, g_com, a_ineq, b_ineq = self._com_terminal(
            p_com, j_com, N_DOF, 0)
        h_ts, g_ts = self._torso_synergy_terminal(q0, e_z=e_z)
        a_pitch, b_pitch = self._pitch_ineq(q0, N_DOF, 1, e_z)
        a_yaw, b_yaw = self._yaw_ineq(q0, N_DOF, 1)
        a_ineq, b_ineq = self._stack_ineq(
            (a_ineq, b_ineq), (a_pitch, b_pitch), (a_yaw, b_yaw))
        p = p + h_com + h_ts
        g = g + g_com + g_ts
        self._apply_joint_barrier(p, g, q0, r, two_blocks=False)
        return solve_dense_box_ineq(
            p, g, box.lb, box.ub, a_ineq=a_ineq, b_ineq=b_ineq,
            x0=self._dq_prev)

    def _solve_horizon(
        self,
        q0: np.ndarray,
        w: np.ndarray,
        w_o: np.ndarray,
        g_wo: np.ndarray,
        r: np.ndarray,
        jac_terms: Sequence[Tuple],
        p_com: np.ndarray,
        j_com: np.ndarray,
    ) -> np.ndarray:
        h_track = np.zeros((N_DOF, N_DOF), dtype=float)
        for term in jac_terms:
            jac_v_pad, jac_w_pad, ori_on = term[2], term[5], term[7]
            h_track = h_track + self.q_pos * (jac_v_pad.T @ jac_v_pad)
            if ori_on and self.q_ori > 0.0:
                h_track = h_track + self.q_ori * (jac_w_pad.T @ jac_w_pad)
        e_z = self._working_ez(jac_terms)
        self._update_pitch_pin(q0, e_z)

        lim = self._constraints.limits
        dq_max = self._effective_dq_max()
        lo_full, hi_full = y_bounds(
            q0, lim.q_min, lim.q_max, dq_max, self._constraints.dt,
            self.n_horizon, margin=self.q_limit_margin)
        n = N_DOF
        lo = np.concatenate([lo_full[:n], lo_full[-n:]])
        hi = np.concatenate([hi_full[:n], hi_full[-n:]])
        pitch_lo = lo_full[-n:][_SLICE_PITCH]
        pitch_hi = hi_full[-n:][_SLICE_PITCH]

        h_vo = np.zeros((N_DOF, N_DOF), dtype=float)
        g_vo = np.zeros(N_DOF, dtype=float)
        sum_lo, sum_hi = self._pitch_sum_bounds(q0, e_z)
        for term in jac_terms:
            jac, e = term[1], term[3]
            jac_w, e_ori, ori_on = term[4], term[6], term[7]
            h_i, g_i = self._assemble_vo(
                jac, e, pitch_lo, pitch_hi, e_z=e_z,
                pitch_sum_lo=sum_lo, pitch_sum_hi=sum_hi,
                jac_omega=jac_w, e_ori=e_ori, ori_on=ori_on)
            h_vo = h_vo + h_i
            g_vo = g_vo + g_i
        n_qp = self._n_qp
        nv = n * n_qp
        h_com, g_com, a_ineq, b_ineq = self._com_terminal(
            p_com, j_com, nv, n_qp - 1)
        h_ts, g_ts = self._torso_synergy_terminal(q0, e_z=e_z)
        a_pitch, b_pitch = self._pitch_ineq(q0, nv, n_qp, e_z)
        a_yaw, b_yaw = self._yaw_ineq(q0, nv, n_qp)
        a_ineq, b_ineq = self._stack_ineq(
            (a_ineq, b_ineq), (a_pitch, b_pitch), (a_yaw, b_yaw))
        h_vo = h_vo + h_com + h_ts
        g_vo = g_vo + g_com + g_ts
        idx = np.arange(4)
        h_vo[idx, idx] += np.asarray(r, dtype=float).reshape(N_DOF)[idx]

        # 20261015 第一刀：assemble 与 solve 之间插入屏障（镜像 C++ 调用序）。
        assemble_endpoints(
            self.n_horizon, h_track, self._stage_w(w, e_z),
            self.track_alpha, h_vo, g_vo, w_o, g_wo,
            r, self._cost.S, self._dq_prev,
            self._p_work, self._g_work)
        self._apply_joint_barrier(
            self._p_work, self._g_work, q0, r, two_blocks=True)
        if a_ineq is None or b_ineq is None or np.size(a_ineq) == 0:
            y = solve_dense_box_active(
                self._p_work, self._g_work, lo, hi, x0=self._y_prev)
        else:
            y = solve_dense_box_ineq(
                self._p_work, self._g_work, lo, hi,
                a_ineq=a_ineq, b_ineq=b_ineq, x0=self._y_prev)
        self._y_prev = shift_warm_endpoints(y, self.n_horizon)
        yb = y.reshape(n_qp, N_DOF)
        self._y_last = yb
        self._y_w_pred = yb[-1, SLICE_WAIST].copy()
        self._y_w_pred[_IDX_YAW] = 0.0
        return yb[0].copy()

    def _ref_betas(self, n_horizon: int) -> np.ndarray:
        """β_k = 1-(1-α)^{k+1} on the path; β_{N-1} = 1 so the terminal tracks q_a."""
        n_h = int(n_horizon)
        k = np.arange(n_h, dtype=float)
        betas = 1.0 - np.power(1.0 - self.track_alpha, k + 1.0)
        betas[-1] = 1.0
        return betas

    def _active_norm(
        self,
        vec6: np.ndarray,
        target_left: Optional[np.ndarray],
        target_right: Optional[np.ndarray],
    ) -> float:
        vals: List[float] = []
        if target_left is not None:
            vals.append(float(np.linalg.norm(vec6[0:3])))
        if target_right is not None:
            vals.append(float(np.linalg.norm(vec6[3:6])))
        return max(vals) if vals else 0.0

    def _targets_jumped(
        self,
        target_left: Optional[np.ndarray],
        target_right: Optional[np.ndarray],
    ) -> bool:
        jumped = self._one_target_jumped(target_left, self._prev_left)
        jumped = self._one_target_jumped(target_right, self._prev_right) or jumped
        self._prev_left = None if target_left is None else np.asarray(
            target_left, dtype=float).reshape(3).copy()
        self._prev_right = None if target_right is None else np.asarray(
            target_right, dtype=float).reshape(3).copy()
        return jumped

    def _ori_jumped(
        self,
        e_ori_left: np.ndarray,
        e_ori_right: np.ndarray,
    ) -> bool:
        jumped = float(np.linalg.norm(
            e_ori_left - self._e_ori_prev_left)) > _TARGET_JUMP_ORI_RAD
        jumped = jumped or float(np.linalg.norm(
            e_ori_right - self._e_ori_prev_right)) > _TARGET_JUMP_ORI_RAD
        self._e_ori_prev_left = np.asarray(e_ori_left, dtype=float).reshape(3).copy()
        self._e_ori_prev_right = np.asarray(e_ori_right, dtype=float).reshape(3).copy()
        return jumped

    def _ori_arrived_ok(self, jac_terms: Sequence[Tuple]) -> bool:
        if self.arrived_ori_rad <= 0.0:
            return True
        for term in jac_terms:
            if not term[7]:
                continue
            if float(np.linalg.norm(term[6])) >= self.arrived_ori_rad:
                return False
        return True

    @staticmethod
    def _one_target_jumped(
        cur: Optional[np.ndarray],
        prev: Optional[np.ndarray],
    ) -> bool:
        if cur is None and prev is None:
            return False
        if cur is None or prev is None:
            return True
        return float(np.linalg.norm(
            np.asarray(cur, dtype=float).reshape(3) - prev)) > _TARGET_JUMP_M

    def _clear_hold_warm(self) -> None:
        self._dq_prev[:] = 0.0
        self._y_prev[:] = 0.0
        self._y_w_pred[:] = 0.0
        self._pitch_pin = None

    def _apply_arrived(
        self,
        dq: np.ndarray,
        e0: float,
        jumped: bool,
        q0: Optional[np.ndarray] = None,
        e_z: Optional[float] = None,
        ori_ok: bool = True,
    ) -> np.ndarray:
        """Stop on TCP (10/20 mm) ∧ ori, or stalled progress (position only)."""
        if self.arrived_err <= 0.0:
            self.last_arrived = False
            return dq
        e0 = float(e0)
        if jumped:
            self.last_arrived = False
            self._e0_hist.clear()
            self._e0_hist.append(e0)
            self._e0_prev = e0
            return dq
        improvement = self._e0_prev - e0
        self._e0_hist.append(e0)
        stalled = False
        if (len(self._e0_hist) >= self.arrived_stall_ticks
                and e0 > self.arrived_err
                and self.arrived_stall_m > 0.0):
            hist = np.fromiter(self._e0_hist, dtype=float)
            mean_step = float(np.mean(np.abs(np.diff(hist))))
            span = float(hist.max() - hist.min())
            # Far and near: a monotone crawl must not look like a flat wall.
            stalled = (
                mean_step < self.arrived_stall_m
                and span < self.arrived_stall_m)
        if stalled and self._stall_blocked_by_wall_slide(q0, e_z, e0):
            stalled = False
        stay_close = self.last_arrived and e0 < self.arrived_exit and ori_ok
        stay_stall = (
            self.last_arrived and stalled and improvement < 0.002)
        enter_close = (not self.last_arrived) and e0 < self.arrived_err and ori_ok
        enter_stall = (not self.last_arrived) and stalled
        self._e0_prev = e0
        if stay_close or stay_stall or enter_close or enter_stall:
            self.last_arrived = True
            self._clear_hold_warm()
            return np.zeros(N_DOF, dtype=float)
        self.last_arrived = False
        return dq

    def _stall_blocked_by_wall_slide(
        self,
        q0: Optional[np.ndarray],
        e_z: Optional[float],
        e0: float,
    ) -> bool:
        """Pitch-wall fake dead point: large e0, leftover height, J2 slack."""
        if q0 is None or e_z is None:
            return False
        if float(e0) <= _STALL_WALL_E0_M:
            return False
        q = np.asarray(q0, dtype=float).reshape(N_DOF)
        if not self._pitch_at_lower_wall(float(q[0] + q[1] + q[2])):
            return False
        if abs(float(e_z)) <= _STALL_WALL_EZ_M:
            return False
        j2_min = float(self._constraints.limits.q_min[1])
        slack = float(q[1]) - j2_min
        return slack > float(np.deg2rad(_STALL_J2_SLACK_DEG))
