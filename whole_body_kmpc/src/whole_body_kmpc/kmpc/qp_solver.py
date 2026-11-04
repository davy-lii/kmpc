"""QP solvers.

P0 uses the closed-form diagonal + box clip (W, R, S are diagonal).
Cartesian / horizon QPs are dense; bound-constrained Newton for N>1.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class QpSolverError(RuntimeError):
    pass


def solve_diag_box(
    p_diag: np.ndarray,
    g: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    x0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Minimize 0.5 x^T diag(p) x + g^T x subject to lb <= x <= ub.

    Unconstrained minimizer is x = -g / p. Box is applied by clipping,
    which is exact for separable (diagonal) quadratics.
    """
    del x0  # warm start unused for N=1 closed form; kept for P1 API
    p = np.asarray(p_diag, dtype=float)
    if np.any(p <= 0.0):
        raise QpSolverError('P diagonal must be strictly positive')
    x = -np.asarray(g, dtype=float) / p
    lo = np.asarray(lb, dtype=float)
    hi = np.asarray(ub, dtype=float)
    if np.any(lo > hi + 1e-12):
        raise QpSolverError('infeasible box: lb > ub')
    return np.clip(x, lo, hi)


def solve_dense_box(
    p_mat: np.ndarray,
    g: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    x0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Minimize 0.5 x^T P x + g^T x subject to lb <= x <= ub.

    Unconstrained Newton step then clip. Enough for 18-DOF N=1 with rate limits.
    """
    del x0
    p = np.asarray(p_mat, dtype=float)
    n = p.shape[0]
    p = 0.5 * (p + p.T) + 1e-8 * np.eye(n)
    try:
        x = np.linalg.solve(p, -np.asarray(g, dtype=float))
    except np.linalg.LinAlgError as exc:
        raise QpSolverError('dense P is singular') from exc
    lo = np.asarray(lb, dtype=float)
    hi = np.asarray(ub, dtype=float)
    if np.any(lo > hi + 1e-12):
        raise QpSolverError('infeasible box: lb > ub')
    return np.clip(x, lo, hi)


def solve_dense_box_active(
    p_mat: np.ndarray,
    g: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    x0: Optional[np.ndarray] = None,
    max_iter: int = 3,
) -> np.ndarray:
    """Bound-constrained dense QP. Active-set Newton so later-node limits
    can change the first step (clip-only would not reallocate waist/arm).
    """
    p = np.asarray(p_mat, dtype=float)
    g = np.asarray(g, dtype=float).reshape(-1)
    n = g.size
    p = 0.5 * (p + p.T) + 1e-8 * np.eye(n)
    lo = np.asarray(lb, dtype=float).reshape(n)
    hi = np.asarray(ub, dtype=float).reshape(n)
    if np.any(lo > hi + 1e-12):
        raise QpSolverError('infeasible box: lb > ub')
    del x0
    try:
        x_free = np.linalg.solve(p, -g)
    except np.linalg.LinAlgError as exc:
        raise QpSolverError('dense P is singular') from exc
    x = np.clip(x_free, lo, hi)
    if np.all((x_free >= lo - 1e-12) & (x_free <= hi + 1e-12)):
        return x

    eps = 1e-10
    for _ in range(int(max_iter)):
        grad = p @ x + g
        at_lo = (x <= lo + eps) & (grad >= -eps)
        at_hi = (x >= hi - eps) & (grad <= eps)
        active = at_lo | at_hi
        free = np.flatnonzero(~active)
        if free.size == 0:
            break
        act = np.flatnonzero(active)
        rhs = -g[free]
        if act.size:
            rhs = rhs - p[np.ix_(free, act)] @ x[act]
        p_ff = p[np.ix_(free, free)]
        try:
            xf = np.linalg.solve(p_ff, rhs)
        except np.linalg.LinAlgError:
            xf = np.linalg.lstsq(p_ff, rhs, rcond=None)[0]
        x_new = x.copy()
        x_new[free] = xf
        x_new = np.clip(x_new, lo, hi)
        if float(np.max(np.abs(x_new - x))) < 1e-9:
            x = x_new
            break
        x = x_new
    return x


def solve_dense_box_ineq(
    p_mat: np.ndarray,
    g: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    a_ineq: Optional[np.ndarray] = None,
    b_ineq: Optional[np.ndarray] = None,
    x0: Optional[np.ndarray] = None,
    penalty: float = 80.0,
    max_iter: int = 8,
) -> np.ndarray:
    """Box QP plus linearized inequalities A x >= b.

    Violated rows become growing quadratic penalties and are re-solved
    with the bound-constrained active-set Newton (same as the box QP).
    """
    x = solve_dense_box_active(p_mat, g, lb, ub, x0=x0)
    if a_ineq is None or b_ineq is None or np.size(a_ineq) == 0:
        return x
    a_mat = np.asarray(a_ineq, dtype=float)
    if a_mat.ndim == 1:
        a_mat = a_mat.reshape(1, -1)
    b_vec = np.asarray(b_ineq, dtype=float).reshape(-1)
    if a_mat.shape[0] != b_vec.size:
        raise QpSolverError('A and b size mismatch')
    p0 = np.asarray(p_mat, dtype=float)
    g0 = np.asarray(g, dtype=float)
    weight = float(penalty)
    for _ in range(max_iter):
        viol = a_mat @ x - b_vec
        if np.all(viol >= -1e-6):
            return x
        p_work = p0.copy()
        g_work = g0.copy()
        for i, gap in enumerate(viol):
            if gap >= -1e-6:
                continue
            row = a_mat[i]
            p_work = p_work + weight * np.outer(row, row)
            g_work = g_work - weight * b_vec[i] * row
        x = solve_dense_box_active(p_work, g_work, lb, ub)
        weight *= 4.0
    return x
