# -- coding: utf-8 --

"""GuiControls/GuiControls.py: Implements mouse and keyboard navigation for the model viewer."""

import math
from dataclasses import dataclass, field, InitVar
from typing import Callable, ClassVar

import imgui
import torch
from kornia.geometry import rotation_matrix_to_quaternion, quaternion_to_rotation_matrix

from ICGui.Backends import BaseBackend
from ICGui.util.GuiAction import Action
from ICGui.GuiComponents import BaseGuiComponent

from .BaseCamera import BaseCamera
from .OrbitalCamera import OrbitalCamera
from .FlyingCamera import FlyingCamera


@dataclass(slots=True)
class GuiCamera:
    """Simple camera class for mouse and keyboard navigation in the model viewer."""
    initial_w2c: InitVar[torch.Tensor]
    backend: BaseBackend = field(repr=False)

    cuda: bool = False
    travel_speed: float = field(kw_only=True, default=1.0)
    rotation_speed: float = field(kw_only=True, default=0.05)
    zoom_speed: float = field(kw_only=True, default=0.25)
    get_key_mapping: Callable[[Action], tuple[int]] = field(kw_only=True, default=lambda action: [], repr=False)

    _invert_x: bool = field(kw_only=True, default=False)
    _invert_y: bool = field(kw_only=True, default=False)

    _camera: BaseCamera = field(init=False, repr=False)
    _camera_idx: int = field(init=False, repr=False, default=1)
    _CAMERA_TYPES: ClassVar[tuple[str, type[BaseCamera]]] = [
        ('Orbital', OrbitalCamera),
        ('Walking', FlyingCamera),
    ]
    easing: bool = True
    easing_time: float = 1.0
    easing_function: Callable[[float], float] = \
        lambda x: (4 * x * x * x) if x < 0.5 else (1 - math.pow(-2 * x + 2, 3) / 2)
    _easing_current_time: float = 1.0
    _ease_from: torch.Tensor | None = None
    _ease_to: torch.Tensor | None = None
    _ease_from_rotation: torch.Tensor | None = None
    _ease_to_rotation: torch.Tensor | None = None

    _is_mouse_rotating: bool = field(init=False, default=False, repr=False)
    _mouse_captured: bool = field(init=False, default=False, repr=False)
    _screenshot: bool = field(init=False, default=False)
    _tile_windows: bool = field(init=False, default=False)

    def __post_init__(self, initial_w2c) -> None:
        self._camera = self._CAMERA_TYPES[self._camera_idx][1](
            initial_w2c,
            travel_speed=self.travel_speed,
            rotation_speed=self.rotation_speed,
            zoom_speed=self.zoom_speed,
            invert_x=self._invert_x,
            invert_y=self._invert_y,
            cuda=self.cuda,
        )

        BaseGuiComponent.registerExternalHotkey('Move Forward', self.get_key_mapping(Action.FORWARD))
        BaseGuiComponent.registerExternalHotkey('Move Backward', self.get_key_mapping(Action.BACKWARD))
        BaseGuiComponent.registerExternalHotkey('Move Left', self.get_key_mapping(Action.LEFT))
        BaseGuiComponent.registerExternalHotkey('Move Right', self.get_key_mapping(Action.RIGHT))
        BaseGuiComponent.registerExternalHotkey('Move Up', self.get_key_mapping(Action.UP))
        BaseGuiComponent.registerExternalHotkey('Move Down', self.get_key_mapping(Action.DOWN))
        BaseGuiComponent.registerExternalHotkey('Increase Movement Speed', self.get_key_mapping(Action.SPEED_UP))
        BaseGuiComponent.registerExternalHotkey('Decrease Movement Speed', self.get_key_mapping(Action.SPEED_DOWN))
        BaseGuiComponent.registerExternalHotkey('Take Screenshot', self.get_key_mapping(Action.SCREENSHOT))
        BaseGuiComponent.registerExternalHotkey('Tile Open Windows', self.get_key_mapping(Action.TILE_WINDOWS))
        BaseGuiComponent.registerExternalHotkey('Toggle Fullscreen', self.get_key_mapping(Action.FULLSCREEN))
        BaseGuiComponent.registerExternalHotkey('Rotate View', ['Left Mouse Button'])
        BaseGuiComponent.registerExternalHotkey('Zoom', ['Mouse Wheel'])
        BaseGuiComponent.registerExternalHotkey('Cycle Camera Controls',
                                                self.get_key_mapping(Action.CYCLE_CAMERA_CONTROLS))
        BaseGuiComponent.registerExternalHotkey('Capture Mouse (in Flying Camera)',
                                                ['Double Click Left Mouse Button'])

    def chooseControls(self, idx: int):
        """Chooses the controls to use."""
        self._camera_idx = idx % len(self._CAMERA_TYPES)
        self._camera = self._CAMERA_TYPES[self._camera_idx][1](
            self._camera.w2c,
            travel_speed=self._camera.travel_speed,
            travel_speed_scale=self._camera.travel_speed_scale,
            rotation_speed=self._camera.rotation_speed,
            zoom_speed=self._camera.zoom_speed,
            invert_x=self._camera.invert_x,
            invert_y=self._camera.invert_y,
            cuda=self.cuda,
        )

        # Reset mouse capture if necessary
        if not self._camera.SUPPORTS_CAPTURE and self._mouse_captured:
            self.backend.captureMouse(False)
            self._mouse_captured = False
            self._is_mouse_rotating = False

    # pylint: disable=too-many-branches
    def handleInputs(self, delta_time: float):
        """Detects mouse and keyboard inputs and updates the camera accordingly. Also handles easing."""
        self._handleKeyboardInputs(delta_time)
        self._handleMouseInputs()
        self._performEasing(delta_time)

    def _handleKeyboardInputs(self, delta_time: float):
        if imgui.get_io().want_capture_keyboard:
            return
        if imgui.is_any_item_focused():
            return

        if any(imgui.is_key_down(k) for k in self.get_key_mapping(Action.FORWARD)):
            self._camera.forward(delta_time)
            self.easing = False
        if any(imgui.is_key_down(k) for k in self.get_key_mapping(Action.BACKWARD)):
            self._camera.backward(delta_time)
            self.easing = False
        if any(imgui.is_key_down(k) for k in self.get_key_mapping(Action.LEFT)):
            self._camera.left(delta_time)
            self.easing = False
        if any(imgui.is_key_down(k) for k in self.get_key_mapping(Action.RIGHT)):
            self._camera.right(delta_time)
            self.easing = False
        if any(imgui.is_key_down(k) for k in self.get_key_mapping(Action.UP)):
            self._camera.up(delta_time)
            self.easing = False
        if any(imgui.is_key_down(k) for k in self.get_key_mapping(Action.DOWN)):
            self._camera.down(delta_time)
            self.easing = False
        if any(imgui.is_key_pressed(k) for k in self.get_key_mapping(Action.SPEED_UP)):
            self._camera.increaseTravelSpeed()
            self.easing = False
        if any(imgui.is_key_pressed(k) for k in self.get_key_mapping(Action.SPEED_DOWN)):
            self._camera.decreaseTravelSpeed()
            self.easing = False
        # noinspection PyArgumentList
        # (Incorrect annotation in pyimgui bindings)
        if any(imgui.is_key_pressed(k, repeat=False) for k in self.get_key_mapping(Action.FULLSCREEN)):
            self.backend.toggleFullscreen()
        # noinspection PyArgumentList
        # (Incorrect annotation in pyimgui bindings)
        if any(imgui.is_key_pressed(k, repeat=False) for k in self.get_key_mapping(Action.SCREENSHOT)):
            self._screenshot = True
        # noinspection PyArgumentList
        # (Incorrect annotation in pyimgui bindings)
        if any(imgui.is_key_pressed(k, repeat=False) for k in self.get_key_mapping(Action.TILE_WINDOWS)):
            self._tile_windows = True
        # noinspection PyArgumentList
        # (Incorrect annotation in pyimgui bindings)
        if any(imgui.is_key_pressed(k, repeat=False) for k in self.get_key_mapping(Action.CYCLE_CAMERA_CONTROLS)):
            self.chooseControls(self._camera_idx + 1)

    def _handleMouseInputs(self):
        if self._camera.SUPPORTS_CAPTURE and imgui.is_mouse_double_clicked(0) and not imgui.get_io().want_capture_mouse:
            self._mouse_captured = not self._mouse_captured
            self._is_mouse_rotating = self._mouse_captured
            self.backend.captureMouse(self._mouse_captured)
            self.easing = False
            if self._mouse_captured:
                self._camera.startRotation()

        # Allow cancelling mouse capture with ESC
        if any(imgui.is_key_pressed(k) for k in self.backend.getKeyMapping(Action.CANCEL)):
            self._mouse_captured = False
            self._is_mouse_rotating = False
            self.backend.captureMouse(False)
            self.easing = False

        if imgui.is_mouse_released(0):
            if not self._mouse_captured:
                self.backend.captureMouse(False)
                self._is_mouse_rotating = False
            self.easing = False
        if imgui.is_mouse_clicked(0) and not imgui.get_io().want_capture_mouse:
            if not self._mouse_captured:
                self.backend.captureMouse(True)
                self._is_mouse_rotating = True
                self._camera.startRotation()
            self.easing = False
        if self._is_mouse_rotating:
            mouse_delta = self.backend.mouse_delta
            self._camera.rotate(mouse_delta)
            self.easing = False

        if imgui.get_io().mouse_wheel != 0 and not imgui.get_io().want_capture_mouse:
            self._camera.zoom(imgui.get_io().mouse_wheel)
            self.easing = False

    def _performEasing(self, dt: float):
        """Performs easing between the current and target camera."""
        if not self.easing:
            return
        if self._ease_from is None or self._ease_to is None:
            return
        if self._easing_current_time >= 1.0:
            self.w2c = self._ease_to
            self._ease_from = None
            self._ease_to = None
            self._ease_from_rotation = None
            self._ease_to_rotation = None
            return

        self._easing_current_time = min(1.0, self._easing_current_time + dt / self.easing_time)
        t = self.easing_function(self._easing_current_time)
        interpolated = self._ease_from * (1 - t) + self._ease_to * t
        interpolated_rotation = quaternion_to_rotation_matrix(
            self._ease_from_rotation * (1 - t) + self._ease_to_rotation * t
        )
        interpolated_rotation[:, 2] *= -1

        interpolated[:3, :3] = interpolated_rotation
        self.w2c = interpolated

    def easeTo(self, target_w2c: torch.Tensor):
        """Eases the camera to the given target camera."""
        self.easing = True
        self._ease_from = self.w2c
        self._ease_to = target_w2c
        ease_from_rotation = self._ease_from[:3, :3].clone()
        ease_from_rotation[:, 2] *= -1
        ease_to_rotation = self._ease_to[:3, :3].clone()
        ease_to_rotation[:, 2] *= -1
        self._ease_from_rotation = rotation_matrix_to_quaternion(ease_from_rotation).cuda()
        self._ease_to_rotation = rotation_matrix_to_quaternion(ease_to_rotation).cuda()
        self._easing_current_time = 0.0

    @property
    def easing_target(self) -> torch.Tensor | None:
        """Returns the target world-to-camera matrix for easing."""
        return self._ease_to

    @property
    def screenshot(self) -> bool:
        """Returns whether a screenshot should be taken."""
        screenshot = self._screenshot
        self._screenshot = False

        return screenshot

    @property
    def tile_windows(self) -> bool:
        """Returns whether the open windows should be tiled."""
        tile_windows = self._tile_windows
        self._tile_windows = False

        return tile_windows

    @property
    def camera_controls(self) -> list[str]:
        """Returns an iterator over the available camera controls."""
        return [self._CAMERA_TYPES[i][0] for i in range(len(self._CAMERA_TYPES))]

    @property
    def current_camera_controls(self) -> int:
        """Returns the index of the current camera controls."""
        return self._camera_idx

    @property
    def invert_x(self) -> bool:
        """Returns whether the x-axis should be inverted."""
        return self._camera.invert_x

    @invert_x.setter
    def invert_x(self, invert_x: bool) -> None:
        """Sets whether the x-axis should be inverted."""
        self._camera.invert_x = invert_x

    @property
    def invert_y(self) -> bool:
        """Returns whether the y-axis should be inverted."""
        return self._camera.invert_y

    @invert_y.setter
    def invert_y(self, invert_y: bool) -> None:
        """Sets whether the y-axis should be inverted."""
        self._camera.invert_y = invert_y

    @property
    def mouse_sensitivity(self) -> float:
        """Returns the mouse sensitivity."""
        return self._camera.rotation_speed

    @mouse_sensitivity.setter
    def mouse_sensitivity(self, mouse_sensitivity: float) -> None:
        """Sets the mouse sensitivity."""
        self._camera.rotation_speed = mouse_sensitivity

    @property
    def movement_speed(self) -> float:
        """Returns the movement speed."""
        return self._camera.travel_speed_scale

    @movement_speed.setter
    def movement_speed(self, movement_speed: float) -> None:
        """Sets the movement speed."""
        self._camera.travel_speed_scale = movement_speed

    @property
    def w2c(self) -> torch.Tensor:
        """Returns the world-to-camera matrix."""
        return self._camera.w2c

    @w2c.setter
    def w2c(self, w2c: torch.Tensor) -> None:
        """Sets the position and rotation from the given world-to-camera matrix."""
        self._camera.w2c = w2c

    @property
    def c2w(self) -> torch.Tensor:
        """Returns the camera-to-world matrix."""
        return self._camera.c2w

    @c2w.setter
    def c2w(self, c2w: torch.Tensor) -> None:
        """Sets the position and rotation from the given camera-to-world matrix."""
        self._camera.c2w = c2w
