# -- coding: utf-8 --

"""Base/Viewer.py: Abstract Base Class for the model viewer, which displays the model output in the imgui window."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Literal

from ICGui.Viewers.Base.Extras import BaseExtra
from ICGui.Backends import BaseBackend

from Cameras.Base import BaseCamera


class BaseViewer(ABC):
    """Base Class for the model viewer, which displays the model output in the imgui window."""
    def __init__(self, backend: BaseBackend, config: 'NeRFConfig', resolution_factor: float = 1.0):
        super().__init__()
        self.config: 'NeRFConfig' = config
        self._backend: BaseBackend = backend
        self._resolution_factor: float = resolution_factor

    def initialize(self):
        """Initializes the viewer, called after the GUI is initialized."""
        pass

    @abstractmethod
    def render(self):
        """Renders the current frame to the window."""
        pass

    @abstractmethod
    def renderExtras(self, extras_enabled: dict[str, bool], camera_override: BaseCamera | None = None):
        """Renders implementation defined extras, of which a list is defined as property extras."""
        pass

    @abstractmethod
    def resize(self, window_size: tuple[int, int], resolution_factor: float | None = None):
        """Resizes the viewer to the given window size and resolution factor. If the resolution factor is not given,
        the current resolution factor is used"""
        if resolution_factor is not None:
            self._resolution_factor = resolution_factor

    @abstractmethod
    def drawToTexture(self, img: Any, texture: Any = None, channels: int = 4, interpolate: bool = False):
        """Draws the given RGBA image to a texture. Details about the format are determined by the implementation.
        If interpolate is set to True, inputs of wrong should be interpolated, otherwise they should be
        padded / cropped."""
        pass

    @abstractmethod
    def draw(self, imgs: Mapping[str, Any], color_mode: str, color_map: str = 'Grayscale', interpolate=False):
        """Draws the given image to the viewer, automatically determining the correct method to call based on the
        color mode (e.g. 'rgb' or 'depth'). If interpolate is set to True, inputs of wrong should be interpolated, #
        otherwise they should be padded / cropped."""
        pass

    @abstractmethod
    def setMode(self, mode: Literal['full', 'scaled']):
        """Sets the mode to draw textures at full resolution or upscale them."""
        pass

    @abstractmethod
    def saveScreenshot(self, path: Path):
        """Saves the current frame including extras to the user's screenshot directory, taken from the current frame
        buffer."""
        pass

    @property
    @abstractmethod
    def extras(self) -> list[BaseExtra]:
        """Returns a list of extras that can be rendered."""
        pass
