# -- coding: utf-8 --

"""GuiControls/OrbitalCamera.py: Implements orbital style camera navigation."""

import math
from dataclasses import dataclass, field

import torch

from .BaseCamera import BaseCamera


@dataclass
class OrbitalCamera(BaseCamera):
    """Camera class for mouse and keyboard navigation in the model viewer implementing orbital style navigation."""
    _start_rotation: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self, initial_w2c) -> None:
        super().__post_init__(initial_w2c)

        self._translation: torch.Tensor = torch.tensor([0.0, 0.0, 0.0], device='cuda' if self.cuda else 'cpu')
        self._start_rotation: torch.Tensor = self.rotation.clone()

    def forward(self, dt: float) -> None:
        """Moves the camera forward by an amount depending on the time since the last frame."""
        self._translation += self.rotation @ (self.travel_speed * self._FORWARD * self.travel_speed_scale * dt)

    def backward(self, dt: float) -> None:
        """Moves the camera backward by an amount depending on the time since the last frame."""
        self._translation += self.rotation @ (self.travel_speed * -self._FORWARD * self.travel_speed_scale * dt)

    def left(self, dt: float) -> None:
        """Moves the camera left by an amount depending on the time since the last frame."""
        self._translation += self.rotation @ (self.travel_speed * self._LEFT * self.travel_speed_scale * dt)

    def right(self, dt: float) -> None:
        """Moves the camera right by an amount depending on the time since the last frame."""
        self._translation += self.rotation @ (self.travel_speed * -self._LEFT * self.travel_speed_scale * dt)

    def up(self, dt: float) -> None:
        """Moves the camera up by an amount depending on the time since the last frame."""
        self._translation += self.rotation @ (self.travel_speed * self._UP * self.travel_speed_scale * dt)

    def down(self, dt: float) -> None:
        """Moves the camera down by an amount depending on the time since the last frame."""
        self._translation += self.rotation @ (self.travel_speed * -self._UP * self.travel_speed_scale * dt)

    def startRotation(self):
        """Starts a rotation operation."""
        self._start_rotation = self.rotation.clone()

    def rotate(self, mouse_delta: tuple[float, float]) -> None:
        """Rotates the camera by the given amount."""
        if mouse_delta[0] != 0.0 or mouse_delta[1] != 0.0:
            yaw = self.calculateRotationMatrix(
                (1 if self.invert_x else -1) * mouse_delta[0] * self.rotation_speed * math.pi * 2.0,
                torch.tensor([0.0, 0.0, 1.0]),
            )
            pitch = self.calculateRotationMatrix(
                (1 if self.invert_y else -1) * mouse_delta[1] * self.rotation_speed * math.pi * 1.0,
                self._start_rotation[:, 0],
            )
            self.rotation = torch.mm(yaw, torch.mm(pitch, self._start_rotation))

    def stopRotation(self):
        """Ends a rotation operation."""
        pass

    def zoom(self, amount: float) -> None:
        """Zooms the camera by the given amount."""
        self.position += self.zoom_speed * amount * -self.position

    @property
    def w2c(self) -> torch.Tensor:
        """Returns the world-to-camera matrix."""
        w2c = torch.eye(4, device='cuda' if self.cuda else 'cpu')
        w2c[:3, :3] = self.rotation.t()
        w2c[:3, 3] = self.position

        translation = torch.eye(4, device='cuda' if self.cuda else 'cpu')
        translation[:3, 3] = self._translation

        return w2c @ translation

    @w2c.setter
    def w2c(self, w2c: torch.Tensor) -> None:
        """Sets the position and rotation from the given world-to-camera matrix."""
        self.position = w2c[:3, 3].to(device='cuda' if self.cuda else 'cpu').clone()
        self.rotation = w2c[:3, :3].t().to(device='cuda' if self.cuda else 'cpu').clone()
        self._translation = torch.tensor([0.0, 0.0, 0.0], device='cuda' if self.cuda else 'cpu')

    @property
    def c2w(self) -> torch.Tensor:
        """Returns the camera-to-world matrix."""
        return torch.linalg.inv(self.w2c)

    @c2w.setter
    def c2w(self, c2w: torch.Tensor) -> None:
        """Sets the position and rotation from the given camera-to-world matrix."""
        self.w2c = torch.linalg.inv(c2w)
        self._translation = torch.tensor([0.0, 0.0, 0.0], device='cuda' if self.cuda else 'cpu')
