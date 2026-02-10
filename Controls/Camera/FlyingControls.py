"""Controls/Camera/FlyingControls.py: Implements flying (spaceship) style camera navigations."""

import math
import typing
from dataclasses import dataclass, field

import numpy as np
import quaternion as quat

from .BaseControls import BaseControls
from .utils import apply_quaternion


@dataclass
class FlyingControls(BaseControls):
    """Camera class for mouse and keyboard navigation in the model viewer implementing
    flying (spaceship) style navigation."""
    # Local direction vectors
    _right: np.ndarray = field(init=False, repr=False)
    _up: np.ndarray = field(init=False, repr=False)
    _backward: np.ndarray = field(init=False, repr=False)
    _start_rotation: quat.quaternion = field(init=False, repr=False)

    _yaw_axis: np.ndarray = field(init=False, repr=False)
    # Pitch axis is set during rotation, as it depends on the current yaw rotation.
    # _roll_axis: np.ndarray = field(init=False, repr=False)  TODO: Implement roll axis for camera rotation

    supports_capture: typing.ClassVar[bool] = True

    def __post_init__(self, initial_c2w, initial_w2c, register_callback, unregister_callback) -> None:
        self._yaw_axis = self._UP
        super().__post_init__(initial_c2w, initial_w2c, register_callback=register_callback, unregister_callback=unregister_callback)

    def forward(self, dt: float) -> None:
        """Moves the camera forward by an amount depending on the time since the last frame."""
        self.has_moved = True
        self.position -= self.travel_speed * self._backward * self.travel_speed_scale * dt

    def backward(self, dt: float) -> None:
        """Moves the camera backward by an amount depending on the time since the last frame."""
        self.has_moved = True
        self.position += self.travel_speed * self._backward * self.travel_speed_scale * dt

    def left(self, dt: float) -> None:
        """Moves the camera left by an amount depending on the time since the last frame."""
        self.has_moved = True
        self.position -= self.travel_speed * self._right * self.travel_speed_scale * dt

    def right(self, dt: float) -> None:
        """Moves the camera right by an amount depending on the time since the last frame."""
        self.has_moved = True
        self.position += self.travel_speed * self._right * self.travel_speed_scale * dt

    def up(self, dt: float) -> None:
        """Moves the camera up by an amount depending on the time since the last frame."""
        self.has_moved = True
        self.position += self.travel_speed * self._up * self.travel_speed_scale * dt

    def down(self, dt: float) -> None:
        """Moves the camera down by an amount depending on the time since the last frame."""
        self.has_moved = True
        self.position -= self.travel_speed * self._up * self.travel_speed_scale * dt

    def start_rotation(self):
        """Starts a rotation operation."""
        self.has_moved = True
        self._start_rotation = self.rotation.copy()

    def rotate(self, mouse_delta: tuple[float, float]) -> None:
        """Rotates the camera by the given amount."""
        if mouse_delta[0] == 0.0 or mouse_delta[1] == 0.0:
            return
        yaw = (1 if self.invert_rotation_x else -1) * mouse_delta[0] * self.rotation_speed * math.pi / 180.0
        yaw_quat = quat.from_rotation_vector(yaw * self._yaw_axis)

        # Pitch axis is LOCAL camera x-direction, but we cannot use self._left, since that depends on self.rotation,
        #   which we're calculating here. As a result, using self.rotation would cause numerical instability around the poles.
        #   Instead, we use the initial rotation transformed by the current yaw rotation, giving us the correct local x-direction,
        #   without introducing numerical instability.
        pitch = (1 if self.invert_rotation_y else -1) * mouse_delta[1] * self.rotation_speed * math.pi / 180.0
        pitch_axis_transform = yaw_quat * self._start_rotation
        pitch_axis = apply_quaternion(pitch_axis_transform, self._RIGHT)
        pitch_quat = quat.from_rotation_vector(pitch * pitch_axis)

        # Combine yaw and pitch rotations
        delta_rotation = pitch_quat * yaw_quat
        self.rotation = delta_rotation * self._start_rotation

        # Update left/up/forward vectors based on the new rotation
        self._recalculate_directions()

    def stop_rotation(self):
        """Ends a rotation operation."""
        pass

    def zoom(self, amount: float) -> None:
        """Zooms the camera by the given amount by moving forward or backward."""
        self.has_moved = True
        self.position -= self.zoom_speed * amount * self._backward

    def _recalculate_directions(self):
        self._right = apply_quaternion(self.rotation, self._RIGHT)
        self._up = apply_quaternion(self.rotation, self._UP)
        self._backward = apply_quaternion(self.rotation, self._BACKWARD)

    @property
    def w2c(self) -> np.ndarray:
        """Returns the world-to-camera matrix."""
        w2c = np.eye(4)
        w2c[:3, :3] = quat.as_rotation_matrix(self.rotation).transpose()
        w2c[:3, 3] = -apply_quaternion(self.rotation.conjugate(), self.position)
        return self._model_transform @ w2c @ self._model_transform.T

    @w2c.setter
    def w2c(self, w2c: np.ndarray) -> None:
        """Sets the position and rotation from the given world-to-camera matrix."""
        self.has_moved = False
        w2c = self._model_transform.T @ w2c @ self._model_transform

        self.rotation = quat.from_rotation_matrix(w2c[:3, :3].transpose(), nonorthogonal=False)
        self.position = w2c[:3, 3]
        self.position = -apply_quaternion(self.rotation, self.position)
        self._recalculate_directions()

    @property
    def c2w(self) -> np.ndarray:
        """Returns the world-to-camera matrix."""
        c2w = np.eye(4)
        c2w[:3, :3] = quat.as_rotation_matrix(self.rotation)
        c2w[:3, 3] = self.position
        return self._model_transform.T @ c2w @ self._model_transform

    @c2w.setter
    def c2w(self, c2w: np.ndarray):
        """Sets the position and rotation from the given world-to-camera matrix."""
        self.has_moved = False
        c2w = self._model_transform @ c2w @ self._model_transform.T

        self.rotation = quat.from_rotation_matrix(c2w[:3, :3], nonorthogonal=False)
        self.position = c2w[:3, 3]
        self._recalculate_directions()
