"""Controls/Camera/OrbitalControls.py: Implements orbital style camera navigation."""

import math
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import quaternion as quat

from .BaseControls import BaseControls
from .utils import apply_quaternion, EasingAnimation, KeyframeAnimation
from ICGui.Controls.utils import InputCallback
from ICGui.State.Volatile import GlobalState
from ICGui.util.Enums import Action
from Logging import Logger


@dataclass
class OrbitalControls(BaseControls):
    """Camera class for mouse and keyboard navigation in the model viewer implementing orbital style navigation."""
    distance: float = field(init=False, repr=False)
    panning_speed: float = field(kw_only=True, default=0.0005)
    _start_rotation: quat.quaternion = field(init=False, repr=False)
    _start_position: np.ndarray = field(init=False, repr=False)
    _panning_scale: float = field(init=False, repr=False, default=1.0)
    _yaw_axis: np.ndarray = field(init=False, repr=False)
    _pitch_axis: np.ndarray = field(init=False, repr=False)  # Pitch axis is set at start of rotation
    # _roll_axis: np.ndarray = field(init=False, repr=False)  TODO: Implement roll axis for camera rotation

    supports_panning: ClassVar[bool] = True

    def __post_init__(self, initial_c2w, initial_w2c, register_callback, unregister_callback) -> None:
        if (initial_c2w is None) == (initial_w2c is None):
            raise ValueError('Either initial_c2w or initial_w2c must be provided, but not both.')
        if initial_c2w is not None:
            self.c2w = initial_c2w
        else:
            self.w2c = initial_w2c

        # Register controls
        if register_callback is not None:
            # Do not register movement callbacks for orbital camera
            unregister_callback(Action.FORWARD)
            unregister_callback(Action.BACKWARD)
            unregister_callback(Action.LEFT)
            unregister_callback(Action.RIGHT)
            unregister_callback(Action.UP)
            unregister_callback(Action.DOWN)
            register_callback(Action.SPEED_UP, InputCallback(self.increase_travel_speed, continuous=False, interrupt_animation=True), 'Increase Speed')
            register_callback(Action.SPEED_DOWN, InputCallback(self.decrease_travel_speed, continuous=False, interrupt_animation=True), 'Decrease Speed')
            register_callback(Action.ANIMATE, InputCallback(self.animate, continuous=False, interrupt_animation=False), 'Animate Camera')

        self._yaw_axis = self._UP  # Yaw axis is WORLD up direction

    def forward(self, dt: float) -> None:
        """Moves the camera forward by an amount depending on the time since the last frame."""
        self.has_moved = True
        self.position -= apply_quaternion(self.rotation, self.travel_speed * self._BACKWARD * self.travel_speed_scale * dt)

    def backward(self, dt: float) -> None:
        """Moves the camera backward by an amount depending on the time since the last frame."""
        self.has_moved = True
        self.position += apply_quaternion(self.rotation, self.travel_speed * self._BACKWARD * self.travel_speed_scale * dt)

    def left(self, dt: float) -> None:
        """Moves the camera left by an amount depending on the time since the last frame."""
        self.has_moved = True
        self.position -= apply_quaternion(self.rotation, self.travel_speed * self._RIGHT * self.travel_speed_scale * dt)

    def right(self, dt: float) -> None:
        """Moves the camera right by an amount depending on the time since the last frame."""
        self.has_moved = True
        self.position += apply_quaternion(self.rotation, self.travel_speed * self._RIGHT * self.travel_speed_scale * dt)

    def up(self, dt: float) -> None:
        """Moves the camera up by an amount depending on the time since the last frame."""
        self.has_moved = True
        self.position -= apply_quaternion(self.rotation, self.travel_speed * self._DOWN * self.travel_speed_scale * dt)

    def down(self, dt: float) -> None:
        """Moves the camera down by an amount depending on the time since the last frame."""
        self.has_moved = True
        self.position += apply_quaternion(self.rotation, self.travel_speed * self._DOWN * self.travel_speed_scale * dt)

    def start_rotation(self):
        """Starts a rotation operation."""
        self.has_moved = True
        self._start_rotation = self.rotation.copy()
        # Pitch axis is LOCAL camera x-direction *at the start of the rotation*, as it is applied first before yaw
        self._pitch_axis = apply_quaternion(self._start_rotation, self._RIGHT)

    def rotate(self, mouse_delta: tuple[float, float]) -> None:
        """Rotates the camera by the given amount."""
        if mouse_delta[0] == 0.0 or mouse_delta[1] == 0.0:
            return
        yaw = (1 if self.invert_rotation_x else -1) * mouse_delta[0] * self.rotation_speed * math.pi / 90.0
        yaw_quat = quat.from_rotation_vector(yaw * self._yaw_axis)

        pitch = (1 if self.invert_rotation_y else -1) * mouse_delta[1] * self.rotation_speed * math.pi / 90.0
        pitch_quat = quat.from_rotation_vector(pitch * self._pitch_axis)

        # Combine yaw and pitch rotations
        delta_rotation = yaw_quat * pitch_quat
        self.rotation = delta_rotation * self._start_rotation

    def stop_rotation(self):
        """Ends a rotation operation."""
        pass

    def start_panning(self):
        """Starts a panning operation."""
        self.has_moved = True
        self._start_position = self.position.copy()
        # We do not need to use self.travel_speed here (which depends on the scene scale), as panning speed is
        #  proportional to the distance, which is by default already set to a higher / lower value for larger / smaller scenes.
        self._panning_scale = self.panning_speed * self.travel_speed_scale * self.distance

    def pan(self, mouse_delta: tuple[float, float]) -> None:
        """Pans the camera by the given amount."""
        if mouse_delta[0] == 0.0 or mouse_delta[1] == 0.0:
            return

        x_scale = -1 if self.invert_panning_x else 1
        y_scale = -1 if self.invert_panning_y else 1
        self.position = (
            self._start_position
            + x_scale * self._panning_scale * -mouse_delta[0] * apply_quaternion(self.rotation, self._RIGHT)
            + y_scale * self._panning_scale *  mouse_delta[1] * apply_quaternion(self.rotation, self._UP)
        )

    def zoom(self, amount: float) -> None:
        """Zooms the camera by the given amount."""
        self.has_moved = True
        # Ensure we are always moving some minimum distance, and ensure we cannot zoom in too far.
        self.distance = max(self.distance - amount * max(self.distance * self.zoom_speed * 0.5,
                                                         self.zoom_speed / 20.0),
                            0.01)

    def ease_to(self, *, to_w2c: np.ndarray = None, to_c2w: np.ndarray = None, duration: float = 1.0) -> BaseControls:
        """Eases the camera to the given target camera."""
        target_camera: OrbitalControls = super().ease_to(to_w2c=to_w2c, to_c2w= to_c2w, duration=duration)  # type: ignore[return-value]
        if GlobalState().skip_animations:
            return target_camera
        self._animation.from_attributes['distance'] = self.distance  # type: ignore
        self._animation.to_attributes['distance'] = target_camera.distance  # type: ignore
        return target_camera

    def animate(self):
        Logger.log_info("Starting orbital camera animation")
        self._animation = KeyframeAnimation(
            duration=16.0,
            loop=True,
            keyframes=[{
                'position': self.position,
                'rotation': self.rotation,
            }, {
                'position': self.position,
                'rotation': quat.from_rotation_vector(math.pi / 2.0 * self._yaw_axis) * self.rotation,
            }, {
                'position': self.position,
                'rotation': quat.from_rotation_vector(math.pi * self._yaw_axis) * self.rotation,
            }, {
                'position': self.position,
                'rotation': quat.from_rotation_vector(3.0 * math.pi / 2.0 * self._yaw_axis) * self.rotation,
            }, {
                'position': self.position,
                'rotation': self.rotation,
            }],
            interpolation={
                'rotation': EasingAnimation.slerp,
            }
        )
        self.has_moved = False

    @property
    def w2c(self) -> np.ndarray:
        """Returns the world-to-camera matrix."""
        w2c = np.eye(4)
        # Rotation is transposed to convert from c2w to w2c
        w2c[:3, :3] = quat.as_rotation_matrix(self.rotation).transpose()

        # Transform to world coordinates by translating the position by self.distance in the local forward direction, yielding the c2w translation
        position = self.position + self.distance * w2c[2, :3]
        # Transform the c2w translation to w2c translation
        w2c[:3, 3] = -apply_quaternion(self.rotation.conjugate(), position)
        return self._model_transform @ w2c @ self._model_transform

    @w2c.setter
    def w2c(self, w2c: np.ndarray) -> None:
        """Sets the position and rotation from the given world-to-camera matrix."""
        self.has_moved = False
        w2c = self._model_transform.T @ w2c @ self._model_transform

        # Extract the c2w rotation by transposing the w2c rotation and store as quaternion
        self.rotation: quat.quaternion = quat.from_rotation_matrix(w2c[:3, :3].transpose(), nonorthogonal=False)

        # Obtain the c2w camera translation
        cam_pos = -apply_quaternion(self.rotation, w2c[:3, 3])

        # Initialize camera orbit focus point as the distance from (0, 0, 0), forward from the camera position
        self.distance = np.linalg.norm(cam_pos)
        self.position = cam_pos - self.distance * w2c[2, :3]

    @property
    def c2w(self) -> np.ndarray:
        """Returns the camera-to-world matrix."""
        c2w = np.eye(4)
        # Rotation is already stored as c2w rotation
        c2w[:3, :3] = quat.as_rotation_matrix(self.rotation)

        # Transform to world coordinates by translating the position by self.distance in the local forward direction, yielding the c2w translation
        position = self.position + self.distance * c2w[:3, 2]
        # Transform the c2w translation to w2c translation
        c2w[:3, 3] = position
        return self._model_transform.T @ c2w @ self._model_transform

    @c2w.setter
    def c2w(self, c2w: np.ndarray) -> None:
        """Sets the position and rotation from the given camera-to-world matrix."""
        self.has_moved = False
        c2w = self._model_transform @ c2w @ self._model_transform.T

        # Extract the c2w rotation and store as quaternion
        self.rotation: quat.quaternion = quat.from_rotation_matrix(c2w[:3, :3], nonorthogonal=False)
        # Obtain the c2w camera translation
        cam_pos = c2w[:3, 3]

        # Initialize camera orbit focus point as the distance from (0, 0, 0), forward from the camera position
        self.distance = np.linalg.norm(cam_pos)
        self.position = cam_pos - self.distance * c2w[:3, 2]
