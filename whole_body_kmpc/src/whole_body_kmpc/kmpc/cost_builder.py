"""Diagonal cost weights for the joint-space QP.

W is uniform w_joint. expand() R is the wrist-scale r_arm used to
calibrate λ; cartesian / joint trackers overwrite R with λ diag(M(q0)).
"""

from dataclasses import dataclass

import numpy as np

from kmpc.types import N_DOF


@dataclass(frozen=True)
class DiagCost:
    W: np.ndarray
    R: np.ndarray
    S: np.ndarray

    @property
    def P_diag(self) -> np.ndarray:
        """Hessian diagonal for 1-step joint QP: W + R + S."""
        return self.W + self.R + self.S


class CostBuilder:
    def __init__(
        self,
        w_joint: float,
        r_waist: float,
        r_arm: float,
        s_joint: float,
        lambda_dls: float = 0.0,
    ) -> None:
        del r_waist
        if min(w_joint, r_arm, s_joint) < 0.0:
            raise ValueError('cost weights must be >= 0')
        self.w_joint = float(w_joint)
        self.r_arm = float(r_arm)
        self.s_joint = float(s_joint)
        self.lambda_dls = float(lambda_dls)

    def expand(self) -> DiagCost:
        w = np.full(N_DOF, self.w_joint, dtype=float)
        r = np.full(N_DOF, self.r_arm, dtype=float)
        s = np.full(N_DOF, self.s_joint, dtype=float)
        # lambda_dls is kept as a constructor arg for YAML compatibility.
        # It is not added to R (that used to double-count damping). The
        # tracking Hessian already has a 1e-8 diagonal in assemble_tracking.
        return DiagCost(W=w, R=r, S=s)

    def linear_term(
        self,
        q0: np.ndarray,
        q_des: np.ndarray,
        dq_prev: np.ndarray,
        cost: DiagCost,
    ) -> np.ndarray:
        """g in 0.5 x^T P x + g^T x, x = Δq."""
        return cost.W * (q0 - q_des) - cost.S * dq_prev
