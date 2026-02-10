"""Controls/Camera/WalkingControls.py: Implements walking style camera navigations (global up/down)."""

from dataclasses import dataclass

from .FlyingControls import FlyingControls
from .utils import apply_quaternion, swing_twist_decomposition


@dataclass
class WalkingControls(FlyingControls):
    """Camera class for mouse and keyboard navigation in the model viewer implementing
    walking (locked to the ground plane + global up/down) style navigation."""
    def _recalculate_directions(self):
        _, yaw_rotation = swing_twist_decomposition(self.rotation, self._yaw_axis)
        self._backward = apply_quaternion(yaw_rotation, self._BACKWARD)
        self._right = apply_quaternion(yaw_rotation, self._RIGHT)
        self._up = self._UP
