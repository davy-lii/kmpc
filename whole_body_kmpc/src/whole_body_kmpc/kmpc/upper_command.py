"""Upper-layer commands for Planner -> Tracker (not QP decision variables)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from kmpc.types import N_DOF


class TrackingState(str, Enum):
    CALIBRATING = 'CALIBRATING'
    TRACKING = 'TRACKING'


@dataclass
class UpperCommand:
    """Planner output consumed by CartesianTracker each control tick."""

    tracking_state: TrackingState = TrackingState.TRACKING
    waist_locked: bool = False
    q_calib: Optional[np.ndarray] = None
    # left / right / both — None means infer from Cartesian targets.
    active_arm: Optional[str] = None

    def copy(self) -> UpperCommand:
        q = None if self.q_calib is None else np.asarray(
            self.q_calib, dtype=float).reshape(N_DOF).copy()
        return UpperCommand(
            tracking_state=self.tracking_state,
            waist_locked=self.waist_locked,
            q_calib=q,
            active_arm=self.active_arm,
        )


DEFAULT_UPPER_COMMAND = UpperCommand()
