"""GUI backends for the ImGui framework, with additional implementations to render tensors created by the models."""

# Set environment variables to allow pyimgui to work on Wayland

from os import environ
if 'PYOPENGL_PLATFORM' not in environ:
    environ['PYOPENGL_PLATFORM'] = 'glx'

from .Base import BaseBackend
from .SDL2 import CustomBackend as SDL2Backend

__all__ = ['BaseBackend', 'SDL2Backend']
