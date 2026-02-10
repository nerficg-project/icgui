"""util/Enums.py: Enum definitions for various actions and states in the GUI."""

from enum import Enum, auto


class Action(Enum):
    """Enum for keyboard / mouse actions."""
    FORWARD = auto()
    BACKWARD = auto()
    LEFT = auto()
    RIGHT = auto()
    UP = auto()
    DOWN = auto()
    SCREENSHOT = auto()
    TILE_WINDOWS = auto()
    SPEED_UP = auto()
    SPEED_DOWN = auto()
    CYCLE_CAMERA_CONTROLS = auto()
    FULLSCREEN = auto()
    EXIT_FULLSCREEN = auto()
    CAPTURE_MOUSE = auto()
    CANCEL_MOUSE_CAPTURE = auto()
    ROTATE_VIEW = auto()
    PAN_VIEW = auto()
    ZOOM = auto()
    ANIMATE = auto()
    NEXT_POSE = auto()
    PREVIOUS_POSE = auto()
    NEXT_DATASET_SPLIT = auto()
    PREVIOUS_DATASET_SPLIT = auto()


class Modifier(Enum):
    """Enum for keyboard modifiers."""
    SHIFT = auto()
    CTRL = auto()
    ALT = auto()
    NONE = auto()


class CallbackType(Enum):
    """Enum for GUI backend callback types."""
    RESIZE = auto()


class TimeAnimation(Enum):
    """Enum for different time modes"""
    SINUSOIDAL = 0
    LINEAR_FORWARD = auto()
    LINEAR_REVERSE = auto()
    LINEAR_BOUNCE = auto()
