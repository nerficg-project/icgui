# -- coding: utf-8 --

"""util/GuiAction.py: Enum to provide backend independent keyboard action mappings."""

from enum import Enum


class Action(Enum):
    """Enum for keyboard actions."""
    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3
    UP = 4
    DOWN = 5
    SCREENSHOT = 6
    TILE_WINDOWS = 7
    SPEED_UP = 8
    SPEED_DOWN = 9
    CYCLE_CAMERA_CONTROLS = 10
    FULLSCREEN = 11
    CANCEL = 12
