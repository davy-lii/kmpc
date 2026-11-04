"""Pinocchio FK/J placeholder. Runtime uses `urdf_kinematics.UrdfKinematics`."""

from __future__ import annotations

from typing import Optional

try:
    import pinocchio as pin  # type: ignore
except ImportError:
    pin = None


class PinocchioModel:
    def __init__(self, urdf_path: Optional[str] = None) -> None:
        self.urdf_path = urdf_path
        self.available = pin is not None
        self._model = None
        if self.available and urdf_path:
            self._model = pin.buildModelFromUrdf(urdf_path)

    def fk_and_jacobians(self, q18):
        raise NotImplementedError('Pinocchio FK/J is P1; P0 uses joint-space tracking')
