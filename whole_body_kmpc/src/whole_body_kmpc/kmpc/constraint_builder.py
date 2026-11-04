"""Hard box constraints on Δq: joint limits and velocity limits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from kmpc.limits import JointLimits


@dataclass(frozen=True)
class DqBox:
    lb: np.ndarray
    ub: np.ndarray


class ConstraintBuilder:
    def __init__(self, limits: JointLimits, dt: float) -> None:
        if dt <= 0.0:
            raise ValueError('dt must be > 0')
        self.limits = limits
        self.dt = float(dt)

    def box_dq(
        self,
        q0: np.ndarray,
        dq_max: Optional[np.ndarray] = None,
    ) -> DqBox:
        q0 = np.asarray(q0, dtype=float)
        vmax = (
            self.limits.dq_max
            if dq_max is None
            else np.asarray(dq_max, dtype=float).reshape(-1))
        max_step = vmax * self.dt
        lb = np.maximum(self.limits.q_min - q0, -max_step)
        ub = np.minimum(self.limits.q_max - q0, max_step)
        # If q0 is already outside [q_min, q_max], keep a feasible box.
        lb = np.minimum(lb, ub)
        return DqBox(lb=lb, ub=ub)
