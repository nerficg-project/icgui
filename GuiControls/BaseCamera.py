# -- coding: utf-8 --

"""GuiControls/BaseCamera.py: Base class for all different camera navigation implementations"""

import math
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, InitVar

import torch


@dataclass
class BaseCamera(ABC):
    """Abstract base class for all different camera navigation implementations."""
    initial_w2c: InitVar[torch.Tensor]
    position: torch.Tensor = field(init=False)
    rotation: torch.Tensor = field(init=False)

    cuda: bool = field(kw_only=True, default=False)
    travel_speed: float = field(kw_only=True, default=1.0)
    travel_speed_scale: float = field(kw_only=True, default=1.0)
    rotation_speed: float = field(kw_only=True, default=0.05)
    zoom_speed: float = field(kw_only=True, default=0.25)

    invert_x: bool = field(kw_only=True, default=False)
    invert_y: bool = field(kw_only=True, default=False)

    _FORWARD: torch.Tensor = field(init=False, repr=False)  # pylint: disable=invalid-name
    _LEFT: torch.Tensor = field(init=False, repr=False)  # pylint: disable=invalid-name
    _UP: torch.Tensor = field(init=False, repr=False)  # pylint: disable=invalid-name

    SUPPORTS_CAPTURE: typing.ClassVar[bool] = False

    def __post_init__(self, initial_w2c) -> None:
        self.position: torch.Tensor = initial_w2c[:3, 3].to(device='cuda' if self.cuda else 'cpu')
        self.rotation: torch.Tensor = initial_w2c[:3, :3].t().to(device='cuda' if self.cuda else 'cpu')

        self._FORWARD: torch.Tensor = torch.tensor([0, 0, 1], device='cuda' if self.cuda else 'cpu', dtype=torch.float)
        self._LEFT: torch.Tensor = torch.tensor([1, 0, 0], device='cuda' if self.cuda else 'cpu', dtype=torch.float)
        self._UP: torch.Tensor = torch.tensor([0, 1, 0], device='cuda' if self.cuda else 'cpu', dtype=torch.float)

    @abstractmethod
    def forward(self, dt: float):
        """Moves the camera forward."""
        raise NotImplementedError

    @abstractmethod
    def backward(self, dt: float):
        """Moves the camera backward."""
        raise NotImplementedError

    @abstractmethod
    def left(self, dt: float):
        """Moves the camera left."""
        raise NotImplementedError

    @abstractmethod
    def right(self, dt: float):
        """Moves the camera right."""
        raise NotImplementedError

    @abstractmethod
    def up(self, dt: float):
        """Moves the camera up."""
        raise NotImplementedError

    @abstractmethod
    def down(self, dt: float):
        """Moves the camera down."""
        raise NotImplementedError

    @abstractmethod
    def startRotation(self):
        """Starts the rotation of the camera."""
        raise NotImplementedError

    @abstractmethod
    def rotate(self, mouse_delta: tuple[float, float]):
        """Rotates the camera."""
        raise NotImplementedError

    @abstractmethod
    def stopRotation(self):
        """Stops the rotation of the camera."""
        raise NotImplementedError

    @abstractmethod
    def zoom(self, amount: float):
        """Zooms the camera."""
        raise NotImplementedError

    def increaseTravelSpeed(self):
        """Increases the travel speed of the camera."""
        self.travel_speed_scale *= 1.1

    def decreaseTravelSpeed(self):
        """Decreases the travel speed of the camera."""
        self.travel_speed_scale *= 0.9

    @property
    @abstractmethod
    def w2c(self) -> torch.Tensor:
        """Returns the world-to-camera matrix."""
        raise NotImplementedError

    @w2c.setter
    @abstractmethod
    def w2c(self, w2c: torch.Tensor) -> None:
        """Sets the position and rotation from the given world-to-camera matrix."""
        raise NotImplementedError

    @property
    @abstractmethod
    def c2w(self) -> torch.Tensor:
        """Returns the camera-to-world matrix."""
        raise NotImplementedError

    @c2w.setter
    @abstractmethod
    def c2w(self, c2w: torch.Tensor) -> None:
        """Sets the position and rotation from the given camera-to-world matrix."""
        raise NotImplementedError

    def calculateRotationMatrix(self, angle: float, axis: torch.Tensor) -> torch.Tensor:
        """Helper function to calculate a rotation matrix for a given angle and axis."""
        cos_angle = math.cos(angle * math.pi / 180)
        sin_angle = math.sin(angle * math.pi / 180)
        x = axis[0]
        y = axis[1]
        z = axis[2]
        return torch.tensor([
            [x * x * (1 - cos_angle) + cos_angle,
             x * y * (1 - cos_angle) - z * sin_angle,
             x * z * (1 - cos_angle) + y * sin_angle],

            [y * x * (1 - cos_angle) + z * sin_angle,
             y * y * (1 - cos_angle) + cos_angle,
             y * z * (1 - cos_angle) - x * sin_angle],

            [z * x * (1 - cos_angle) - y * sin_angle,
             z * y * (1 - cos_angle) + x * sin_angle,
             z * z * (1 - cos_angle) + cos_angle],
        ], device='cuda' if self.cuda else 'cpu')
