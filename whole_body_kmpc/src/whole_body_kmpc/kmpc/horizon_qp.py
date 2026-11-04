"""Frozen-Jacobian N-step kinematic QP in cumulative displacement y.

Decision: y_k = q_{k+1} - q_0. Only Δq_0 = y_0 is executed.

MPC for tracking: stage cost follows q_a = q_0 + y_{N-1}; the ball only
enters the terminal offset V_o. Cartesian N>1 uses assemble_endpoints
(y_0, y_{N-1}) so the QP is 36-D; yaml N still sizes the terminal box.
assemble_horizon keeps the older block-diagonal stage cost for tests;
assemble_tracking keeps the full stacked path for tests.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from kmpc.qp_solver import solve_dense_box_active, solve_dense_box_ineq
from kmpc.types import N_DOF


def y_bounds(
    q0: np.ndarray,
    q_min: np.ndarray,
    q_max: np.ndarray,
    dq_max: np.ndarray,
    dt: float,
    n_horizon: int,
    margin: float = 0.0,
    step_cap: Optional[np.ndarray] = None,
    step_lo: Optional[np.ndarray] = None,
    step_hi: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Box on stacked y: danger-zone joint limits ∩ velocity envelope.

    Current q0 is always feasible (stay). margin shrinks the joint box by δ
    so the plan avoids riding q_min/q_max.
    """
    q0 = np.asarray(q0, dtype=float).reshape(N_DOF)
    hard_lo = q_min - q0
    hard_hi = q_max - q0
    m = max(float(margin), 0.0)
    soft_lo = np.minimum(q_min + m - q0, 0.0)
    soft_hi = np.maximum(q_max - m - q0, 0.0)
    joint_lo = np.maximum(soft_lo, hard_lo)
    joint_hi = np.minimum(soft_hi, hard_hi)
    joint_lo = np.minimum(joint_lo, joint_hi)

    vdt = np.asarray(dq_max, dtype=float).reshape(N_DOF) * float(dt)
    v_hi = vdt.copy()
    v_lo = -vdt.copy()
    if step_cap is not None:
        cap = np.asarray(step_cap, dtype=float).reshape(-1)
        v_hi[: cap.size] = np.minimum(v_hi[: cap.size], cap)
        v_lo[: cap.size] = np.maximum(v_lo[: cap.size], -cap)
    if step_lo is not None:
        slo = np.asarray(step_lo, dtype=float).reshape(-1)
        v_lo[: slo.size] = np.maximum(v_lo[: slo.size], slo)
    if step_hi is not None:
        shi = np.asarray(step_hi, dtype=float).reshape(-1)
        v_hi[: shi.size] = np.minimum(v_hi[: shi.size], shi)

    n = N_DOF
    nv = n * int(n_horizon)
    lo = np.empty(nv)
    hi = np.empty(nv)
    for k in range(int(n_horizon)):
        sl = slice(k * n, (k + 1) * n)
        travel_lo = (k + 1) * v_lo
        travel_hi = (k + 1) * v_hi
        lo[sl] = np.maximum(joint_lo, travel_lo)
        hi[sl] = np.minimum(joint_hi, travel_hi)
        lo[sl] = np.minimum(lo[sl], hi[sl])
    return lo, hi


def _stage_gradients(
    n_horizon: int,
    n: int,
    g_stage: np.ndarray,
) -> np.ndarray:
    """Per-step linear term. One (n,) vector is broadcast to every k."""
    g = np.asarray(g_stage, dtype=float)
    n_h = int(n_horizon)
    if g.ndim == 1 and g.size == n:
        return np.broadcast_to(g, (n_h, n))
    return g.reshape(n_h, n)


def assemble_horizon(
    n_horizon: int,
    h_stage: np.ndarray,
    g_stage: np.ndarray,
    r: np.ndarray,
    s: np.ndarray,
    dq_prev: np.ndarray,
    p_out: np.ndarray,
    g_out: np.ndarray,
) -> None:
    """Fill 0.5 y^T P y + g^T y. Stage cost is block-diagonal in y; R/S couple time."""
    n = N_DOF
    n_h = int(n_horizon)
    p_out.fill(0.0)
    g_out.fill(0.0)
    rd = np.diag(r)
    sd = np.diag(s)
    g_rows = _stage_gradients(n_h, n, g_stage)

    for k in range(n_h):
        sl = slice(k * n, (k + 1) * n)
        p_out[sl, sl] += h_stage
        g_out[sl] += g_rows[k]

    # ||Δq_0||_R^2, Δq_0 = y_0
    p_out[0:n, 0:n] += rd
    for k in range(1, n_h):
        sl = slice(k * n, (k + 1) * n)
        slp = slice((k - 1) * n, k * n)
        p_out[sl, sl] += rd
        p_out[slp, slp] += rd
        p_out[sl, slp] -= rd
        p_out[slp, sl] -= rd

    # ||Δq_0 - Δq_prev||_S^2
    p_out[0:n, 0:n] += sd
    g_out[0:n] -= s * dq_prev
    if n_h >= 2:
        # ||Δq_1 - Δq_0||_S^2 = ||y_1 - 2 y_0||_S^2
        p_out[n:2 * n, n:2 * n] += sd
        p_out[0:n, 0:n] += 4.0 * sd
        p_out[n:2 * n, 0:n] -= 2.0 * sd
        p_out[0:n, n:2 * n] -= 2.0 * sd
    for k in range(2, n_h):
        # ||Δq_k - Δq_{k-1}||_S^2 = ||y_k - 2 y_{k-1} + y_{k-2}||_S^2
        sl = slice(k * n, (k + 1) * n)
        sl1 = slice((k - 1) * n, k * n)
        sl2 = slice((k - 2) * n, (k - 1) * n)
        p_out[sl, sl] += sd
        p_out[sl1, sl1] += 4.0 * sd
        p_out[sl2, sl2] += sd
        p_out[sl, sl1] -= 2.0 * sd
        p_out[sl1, sl] -= 2.0 * sd
        p_out[sl, sl2] += sd
        p_out[sl2, sl] += sd
        p_out[sl1, sl2] -= 2.0 * sd
        p_out[sl2, sl1] -= 2.0 * sd

    p_out.flat[:: p_out.shape[0] + 1] += 1e-8


def _add_rs(p_out: np.ndarray, g_out: np.ndarray, r: np.ndarray, s: np.ndarray,
            dq_prev: np.ndarray, n_h: int) -> None:
    n = N_DOF
    rd = np.diag(r)
    sd = np.diag(s)
    p_out[0:n, 0:n] += rd
    for k in range(1, n_h):
        sl = slice(k * n, (k + 1) * n)
        slp = slice((k - 1) * n, k * n)
        p_out[sl, sl] += rd
        p_out[slp, slp] += rd
        p_out[sl, slp] -= rd
        p_out[slp, sl] -= rd
    p_out[0:n, 0:n] += sd
    g_out[0:n] -= s * dq_prev
    if n_h >= 2:
        p_out[n:2 * n, n:2 * n] += sd
        p_out[0:n, 0:n] += 4.0 * sd
        p_out[n:2 * n, 0:n] -= 2.0 * sd
        p_out[0:n, n:2 * n] -= 2.0 * sd
    for k in range(2, n_h):
        sl = slice(k * n, (k + 1) * n)
        sl1 = slice((k - 1) * n, k * n)
        sl2 = slice((k - 2) * n, (k - 1) * n)
        p_out[sl, sl] += sd
        p_out[sl1, sl1] += 4.0 * sd
        p_out[sl2, sl2] += sd
        p_out[sl, sl1] -= 2.0 * sd
        p_out[sl1, sl] -= 2.0 * sd
        p_out[sl, sl2] += sd
        p_out[sl2, sl] += sd
        p_out[sl1, sl2] -= 2.0 * sd
        p_out[sl2, sl1] -= 2.0 * sd


def assemble_tracking(
    n_horizon: int,
    h_track: np.ndarray,
    w_stage: np.ndarray,
    betas: np.ndarray,
    h_vo: np.ndarray,
    g_vo: np.ndarray,
    w_o: np.ndarray,
    g_wo: np.ndarray,
    r: np.ndarray,
    s: np.ndarray,
    dq_prev: np.ndarray,
    p_out: np.ndarray,
    g_out: np.ndarray,
) -> None:
    """0.5 y^T P y + g^T y for tracking q_a = q_0 + y_{N-1}.

    Stage k < N-1: ||J (y_k - β_k y_{N-1})||_Q^2 + ||y_k - β_k y_{N-1}||_W^2.
    Terminal: V_o (h_vo, g_vo) and W_o. β_{N-1} must be 1 so those two
    stage terms vanish at k = N-1. R/S still act on Δq.

    w_stage is either the (n,) diagonal of W or a full (n, n) PSD metric
    (W_syn = diag(w) + Q_syn H_syn^T H_syn).
    """
    n = N_DOF
    n_h = int(n_horizon)
    p_out.fill(0.0)
    g_out.fill(0.0)
    h_j = np.asarray(h_track, dtype=float).reshape(n, n)
    w_arr = np.asarray(w_stage, dtype=float)
    if w_arr.ndim == 1:
        w_d = np.diag(w_arr.reshape(n))
    else:
        w_d = 0.5 * (w_arr.reshape(n, n) + w_arr.reshape(n, n).T)
    beta = np.asarray(betas, dtype=float).reshape(n_h)
    sl_n = slice((n_h - 1) * n, n_h * n)

    for k in range(n_h - 1):
        sl = slice(k * n, (k + 1) * n)
        bk = float(beta[k])
        h_w = h_j + w_d
        p_out[sl, sl] += h_w
        p_out[sl_n, sl_n] += (bk * bk) * h_w
        cross = -bk * h_w
        p_out[sl, sl_n] += cross
        p_out[sl_n, sl] += cross

    p_out[sl_n, sl_n] += np.asarray(h_vo, dtype=float).reshape(n, n)
    p_out[sl_n, sl_n] += np.diag(np.asarray(w_o, dtype=float).reshape(n))
    g_out[sl_n] += np.asarray(g_vo, dtype=float).reshape(n)
    g_out[sl_n] += np.asarray(g_wo, dtype=float).reshape(n)

    _add_rs(p_out, g_out, r, s, dq_prev, n_h)
    p_out.flat[:: p_out.shape[0] + 1] += 1e-8


def endpoint_bounds(
    q0: np.ndarray,
    q_min: np.ndarray,
    q_max: np.ndarray,
    dq_max: np.ndarray,
    dt: float,
    n_horizon: int,
    margin: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Box on stacked (y_0, y_{N-1}): one-step ∩ travel-N, same as y_bounds."""
    lo, hi = y_bounds(
        q0, q_min, q_max, dq_max, dt, n_horizon, margin=margin)
    n = N_DOF
    return (
        np.concatenate([lo[:n], lo[-n:]]),
        np.concatenate([hi[:n], hi[-n:]]),
    )


def assemble_endpoints(
    n_horizon: int,
    h_track: np.ndarray,
    w_stage: np.ndarray,
    beta0: float,
    h_vo: np.ndarray,
    g_vo: np.ndarray,
    w_o: np.ndarray,
    g_wo: np.ndarray,
    r: np.ndarray,
    s: np.ndarray,
    dq_prev: np.ndarray,
    p_out: np.ndarray,
    g_out: np.ndarray,
) -> None:
    """0.5 x^T P x + g^T x for x = (y_0, y_{N-1}).

    Path tracking only at k=0: ||J (y_0 - β_0 y_{N-1})||_Q^2
    + ||y_0 - β_0 y_{N-1}||_W^2. Terminal V_o / W_o on y_{N-1}.
    ||y_0||_R^2 + ||y_0 - Δq_prev||_S^2. Intermediate Δq R is the
    uniform chord ||y_{N-1}-y_0||_R^2 / (N-1). No stacked S blocks.
    """
    n = N_DOF
    p_out.fill(0.0)
    g_out.fill(0.0)
    h_j = np.asarray(h_track, dtype=float).reshape(n, n)
    w_arr = np.asarray(w_stage, dtype=float)
    if w_arr.ndim == 1:
        w_d = np.diag(w_arr.reshape(n))
    else:
        w_d = 0.5 * (w_arr.reshape(n, n) + w_arr.reshape(n, n).T)
    h_w = h_j + w_d
    b = float(beta0)
    sl0 = slice(0, n)
    sln = slice(n, 2 * n)
    p_out[sl0, sl0] += h_w
    p_out[sln, sln] += (b * b) * h_w
    cross = -b * h_w
    p_out[sl0, sln] += cross
    p_out[sln, sl0] += cross

    p_out[sln, sln] += np.asarray(h_vo, dtype=float).reshape(n, n)
    p_out[sln, sln] += np.diag(np.asarray(w_o, dtype=float).reshape(n))
    g_out[sln] += np.asarray(g_vo, dtype=float).reshape(n)
    g_out[sln] += np.asarray(g_wo, dtype=float).reshape(n)

    rd = np.diag(np.asarray(r, dtype=float).reshape(n))
    sd = np.diag(np.asarray(s, dtype=float).reshape(n))
    p_out[sl0, sl0] += rd + sd
    g_out[sl0] -= np.asarray(s, dtype=float).reshape(n) * np.asarray(
        dq_prev, dtype=float).reshape(n)

    n_h = int(n_horizon)
    if n_h > 1:
        rch = rd / float(n_h - 1)
        p_out[sl0, sl0] += rch
        p_out[sln, sln] += rch
        p_out[sl0, sln] -= rch
        p_out[sln, sl0] -= rch
    p_out.flat[:: p_out.shape[0] + 1] += 1e-8


def shift_warm_y(y: np.ndarray, n_horizon: int) -> np.ndarray:
    """Shift last plan: new_y_k = old_y_{k+1} - old_y_0."""
    n = N_DOF
    yb = np.asarray(y, dtype=float).reshape(int(n_horizon), n)
    out = np.zeros_like(yb)
    if n_horizon <= 1:
        return out.reshape(-1)
    out[:-1] = yb[1:] - yb[0]
    out[-1] = out[-2]
    return out.reshape(-1)


def shift_warm_endpoints(y: np.ndarray, n_horizon: int) -> np.ndarray:
    """Warm-start (y_0, y_{N-1}) after executing y_0: chord remainder."""
    n = N_DOF
    yb = np.asarray(y, dtype=float).reshape(2, n)
    out = np.zeros_like(yb)
    remain = yb[1] - yb[0]
    n_h = max(int(n_horizon), 2)
    out[0] = remain / float(n_h - 1)
    out[1] = remain
    return out.reshape(-1)


def solve_horizon(
    n_horizon: int,
    h_stage: np.ndarray,
    g_stage: np.ndarray,
    r: np.ndarray,
    s: np.ndarray,
    dq_prev: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    p_work: np.ndarray,
    g_work: np.ndarray,
    y0: Optional[np.ndarray] = None,
) -> np.ndarray:
    assemble_horizon(
        n_horizon, h_stage, g_stage, r, s, dq_prev, p_work, g_work)
    return solve_dense_box_active(p_work, g_work, lo, hi, x0=y0)


def solve_tracking(
    n_horizon: int,
    h_track: np.ndarray,
    w_stage: np.ndarray,
    betas: np.ndarray,
    h_vo: np.ndarray,
    g_vo: np.ndarray,
    w_o: np.ndarray,
    g_wo: np.ndarray,
    r: np.ndarray,
    s: np.ndarray,
    dq_prev: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    p_work: np.ndarray,
    g_work: np.ndarray,
    y0: Optional[np.ndarray] = None,
    a_ineq: Optional[np.ndarray] = None,
    b_ineq: Optional[np.ndarray] = None,
) -> np.ndarray:
    assemble_tracking(
        n_horizon, h_track, w_stage, betas, h_vo, g_vo, w_o, g_wo,
        r, s, dq_prev, p_work, g_work)
    if a_ineq is None or b_ineq is None or np.size(a_ineq) == 0:
        return solve_dense_box_active(p_work, g_work, lo, hi, x0=y0)
    return solve_dense_box_ineq(
        p_work, g_work, lo, hi, a_ineq=a_ineq, b_ineq=b_ineq, x0=y0)


def solve_endpoints(
    n_horizon: int,
    h_track: np.ndarray,
    w_stage: np.ndarray,
    beta0: float,
    h_vo: np.ndarray,
    g_vo: np.ndarray,
    w_o: np.ndarray,
    g_wo: np.ndarray,
    r: np.ndarray,
    s: np.ndarray,
    dq_prev: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    p_work: np.ndarray,
    g_work: np.ndarray,
    y0: Optional[np.ndarray] = None,
    a_ineq: Optional[np.ndarray] = None,
    b_ineq: Optional[np.ndarray] = None,
) -> np.ndarray:
    assemble_endpoints(
        n_horizon, h_track, w_stage, beta0, h_vo, g_vo, w_o, g_wo,
        r, s, dq_prev, p_work, g_work)
    if a_ineq is None or b_ineq is None or np.size(a_ineq) == 0:
        return solve_dense_box_active(p_work, g_work, lo, hi, x0=y0)
    return solve_dense_box_ineq(
        p_work, g_work, lo, hi, a_ineq=a_ineq, b_ineq=b_ineq, x0=y0)
