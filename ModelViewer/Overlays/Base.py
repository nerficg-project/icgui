"""ModelViewer/Overlays/Base.py: Abstract Base Class for 3D OpenGL overlays that can be rendered on top of the model output."""

from abc import abstractmethod
from typing import Any

from Datasets.utils import View


class BaseOverlay:
    """Abstract Base Class for optional OpenGL extras that can be rendered using shaders on top of the model output."""
    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of the extra."""
        pass

    @abstractmethod
    def render_options(self):
        """Renders imgui inputs for the options of the extra."""
        pass

    @abstractmethod
    def render(self, view: View, extra_params: dict[str, Any]):
        """Renders the extra using OpenGL."""
        pass
