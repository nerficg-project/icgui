# -- coding: utf-8 --

"""Backends/Base.py: Abstract Base Class for the imgui backend initialization."""

from abc import ABC, abstractmethod
from typing import Any, Callable

import imgui

from ICGui.util.GuiAction import Action


class BaseBackend(ABC):
    """Abstract Base Class providing the interface for initializing and interacting with the imgui backend."""
    def __init__(self, window_name: str, initial_window_dimensions: tuple[int, int], *,
                 vsync: bool = True, resize_callback: Callable[[int, int], None] = None):
        super().__init__()
        self.window_name: str = window_name
        self._window_dimensions: tuple[int, int] = initial_window_dimensions
        self.resolution_correction: tuple[int, int] = (0, 0)  # Workaround to match drawable area in resolution inputs
        self.vsync: bool = vsync
        self.resize_callback = resize_callback

        self.window: Any = None

    @property
    @abstractmethod
    def window_size(self) -> tuple[int, int]:
        """Returns the size of the window as (width, height)."""
        pass

    @abstractmethod
    def getTrueResolution(self) -> tuple[int, int]:
        """Returns the true resolution of the window as (width, height)."""
        pass

    @abstractmethod
    def beforeFrame(self) -> bool:
        """Called before each frame is rendered. If false is returned, the GUI should be closed."""
        imgui.new_frame()
        return True

    @abstractmethod
    def clearFrame(self):
        """Called before finalizing the frame after the rendering loop."""
        pass

    @abstractmethod
    def afterFrame(self):
        """Finalize and draw the frame."""
        imgui.render()

    @abstractmethod
    def getKeyMapping(self, action: Action) -> tuple[int]:
        """Returns the key mapping used by PyImGui for the given action, as defined by the backend."""
        pass

    @abstractmethod
    def getKeyCode(self, key: str):
        """Returns the key code for the given key, as defined by the backend."""
        pass

    @abstractmethod
    def captureMouse(self, capture: bool):
        """Captures the mouse cursor and constrains it to the center of the window."""
        pass

    @abstractmethod
    def toggleFullscreen(self):
        """Toggles the fullscreen mode of the window."""
        pass

    @abstractmethod
    def resizeWindow(self, width: int, height: int):
        """Resizes the window to the given dimensions (as drawable area)."""
        pass

    @property
    @abstractmethod
    def mouse_delta(self) -> tuple[float, float]:
        """Returns the relative mouse movement since the last frame. Still available after captureMouse()."""
        pass

    @abstractmethod
    def __enter__(self) -> 'BaseBackend':
        """Initializes the rendering backend."""
        return self

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        """Does any necessary cleanup after the GUI is closed."""
        pass
