# -- coding: utf-8 --

"""SDL2Viewer/Extras/Base.py: Abstract Base Class for OpenGL extras that can be rendered on top of the model output."""

from abc import abstractmethod
from typing import Any

from ...Base.Extras import BaseExtra

from Cameras.Base import BaseCamera


class BaseGLExtra(BaseExtra):
    """Abstract Base Class for optional OpenGL extras that can be rendered using shaders on top of the model output."""

    @abstractmethod
    def render(self, camera: BaseCamera, extra_params: dict[str, Any]):
        """Renders the extra using OpenGL."""
        pass
