# -- coding: utf-8 --

"""SDL2Viewer/Extras/Base.py: Abstract Base Class for rendered extras that can be configured from the GUI."""

from abc import ABC, abstractmethod


class BaseExtra(ABC):
    """Abstract Base Class for optional rendered extras that can be configured from the GUI."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the name of the extra."""
        pass

    @abstractmethod
    def renderOptions(self):
        """Renders imgui inputs for the options of the extra."""
        pass
