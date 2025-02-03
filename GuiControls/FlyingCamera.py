# -- coding: utf-8 --

"""GuiControls/FlyingCamera.py: Implements flying style camera navigations."""

import math
import typing
from dataclasses import dataclass, field

import torch

from .BaseCamera import BaseCamera


@dataclass
class FlyingCamera(BaseCamera):
    """Camera class for mouse and keyboard navigation in the model viewer implementing flying style navigation."""
    _planar_rotation: torch.Tensor = field(init=False, repr=False)
    _forward: torch.Tensor = field(init=False, repr=False)
    _left: torch.Tensor = field(init=False, repr=False)
    _up: torch.Tensor = field(init=False, repr=False)

    _start_rotation: torch.Tensor = field(init=False, repr=False)
    _planar_start_rotation: torch.Tensor = field(init=False, repr=False)

    SUPPORTS_CAPTURE: typing.ClassVar[bool] = True

    def __post_init__(self, initial_w2c) -> None:
        super().__post_init__(initial_w2c)
        self.w2c = initial_w2c

    def forward(self, dt: float) -> None:
        """Moves the camera forward by an amount depending on the time since the last frame."""
        self.position += self.travel_speed * self._forward * self.travel_speed_scale * dt

    def backward(self, dt: float) -> None:
        """Moves the camera backward by an amount depending on the time since the last frame."""
        self.position += self.travel_speed * -self._forward * self.travel_speed_scale * dt

    def left(self, dt: float) -> None:
        """Moves the camera left by an amount depending on the time since the last frame."""
        self.position += self.travel_speed * self._left * self.travel_speed_scale * dt

    def right(self, dt: float) -> None:
        """Moves the camera right by an amount depending on the time since the last frame."""
        self.position += self.travel_speed * -self._left * self.travel_speed_scale * dt

    def up(self, dt: float) -> None:
        """Moves the camera up by an amount depending on the time since the last frame."""
        self.position += self.travel_speed * -self._up * self.travel_speed_scale * dt

    def down(self, dt: float) -> None:
        """Moves the camera down by an amount depending on the time since the last frame."""
        self.position += self.travel_speed * self._up * self.travel_speed_scale * dt

    def startRotation(self):
        """Starts a rotation operation."""
        self._start_rotation = self.rotation.clone()

    def rotate(self, mouse_delta: tuple[float, float]) -> None:
        """Rotates the camera by the given amount."""
        if mouse_delta[0] != 0.0 or mouse_delta[1] != 0.0:
            # Factor of 0.5 to ensure consistent mouse sensitivity across different Camera implementations
            yaw = self.calculateRotationMatrix(
                (-1 if self.invert_x else 1) * mouse_delta[0] * self.rotation_speed * 0.5 * math.pi * 2.0,
                torch.tensor([0.0, 0.0, 1.0]),
            )
            pitch = self.calculateRotationMatrix(
                (1 if self.invert_y else -1) * mouse_delta[1] * self.rotation_speed * 0.5 * math.pi * 1.0,
                self._start_rotation[:, 0]
            )
            self.rotation = torch.mm(yaw, torch.mm(pitch, self._start_rotation))
            self._planar_rotation = torch.mm(yaw, self._planar_start_rotation)

            self._recalculateDirections()

    def stopRotation(self):
        """Ends a rotation operation."""
        pass

    def zoom(self, amount: float) -> None:
        """Zooms the camera by the given amount by moving forward or backward."""
        self.position += self.zoom_speed * amount * self._forward

    def _recalculateDirections(self):
        self._forward: torch.Tensor = torch.matmul(self.rotation, self._FORWARD)
        self._left: torch.Tensor = torch.matmul(self.rotation, self._LEFT)
        self._up: torch.Tensor = -torch.matmul(self.rotation, self._UP)

        self._forward /= torch.linalg.norm(self._forward)
        self._left /= torch.linalg.norm(self._left)
        self._up /= torch.linalg.norm(self._up)

    @property
    def w2c(self) -> torch.Tensor:
        """Returns the world-to-camera matrix."""
        rotation = torch.eye(4, device='cuda' if self.cuda else 'cpu')
        rotation[:3, :3] = self.rotation.t()
        translation = torch.eye(4, device='cuda' if self.cuda else 'cpu')
        translation[:3, 3] = self.position

        return torch.mm(rotation, translation)

    @w2c.setter
    def w2c(self, w2c: torch.Tensor) -> None:
        """Sets the position and rotation from the given world-to-camera matrix."""
        self.rotation: torch.Tensor = w2c[:3, :3].t().to(device='cuda' if self.cuda else 'cpu')
        rotation = torch.eye(4, device='cuda' if self.cuda else 'cpu')
        rotation[:3, :3] = self.rotation.t()
        self.position = (torch.inverse(rotation) @ w2c)[:3, 3].to(device='cuda' if self.cuda else 'cpu')

        self._start_rotation: torch.Tensor = self.rotation.clone()
        self._planar_start_rotation: torch.Tensor = self.rotation.clone()
        self._recalculateDirections()

    @property
    def c2w(self) -> torch.Tensor:
        """Returns the camera-to-world matrix."""
        return torch.linalg.inv(self.w2c)

    @c2w.setter
    def c2w(self, c2w: torch.Tensor) -> None:
        """Sets the position and rotation from the given camera-to-world matrix."""
        self.w2c = torch.linalg.inv(c2w)
