"""Controls/utils.py: Utilities related to input handling for the GUI controls."""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

from imgui_bundle import imgui

from ICGui.util.Enums import Action, Modifier


@dataclass(slots=True)
class InputCallback:
    """Represents a callback for input events, such as mouse or keyboard actions."""
    callback: Callable[[float], None] | Callable[[], None]
    continuous: bool = True  # If True, the callback is called continuously while the input is held down
    interrupt_animation: bool = False  # If True, the key interrupts any ongoing easing operation


MODIFIER_CODES: dict[Modifier, imgui.Key] = {
    Modifier.CTRL: imgui.Key.mod_ctrl,
    Modifier.SHIFT: imgui.Key.mod_shift,
    Modifier.ALT: imgui.Key.mod_alt,
    Modifier.NONE: imgui.Key.mod_none,
}

__MODIFIER_CTRL_KEYMAP: OrderedDict[imgui.Key, set[Action | str]] = OrderedDict({
    imgui.Key.a: {Action.ANIMATE},
})

__MODIFIER_NONE_KEYMAP: OrderedDict[imgui.Key, set[Action | str]] = OrderedDict({
    imgui.Key.w: {Action.FORWARD},
    imgui.Key.s: {Action.BACKWARD},
    imgui.Key.a: {Action.LEFT},
    imgui.Key.d: {Action.RIGHT},
    imgui.Key.up_arrow: {Action.NEXT_DATASET_SPLIT},
    imgui.Key.down_arrow: {Action.PREVIOUS_DATASET_SPLIT},
    imgui.Key.left_arrow: {Action.PREVIOUS_POSE},
    imgui.Key.right_arrow: {Action.NEXT_POSE},
    imgui.Key.r: {Action.UP},
    imgui.Key.space: {Action.UP},
    imgui.Key.f: {Action.DOWN},
    imgui.Key.left_shift: {Action.DOWN},
    imgui.Key.left_ctrl: {Action.DOWN},
    imgui.Key.right_shift: {Action.DOWN},
    imgui.Key.right_ctrl: {Action.DOWN},
    imgui.Key.keypad_add: {Action.SPEED_UP},
    imgui.Key.equal: {Action.SPEED_UP},
    imgui.Key.page_up: {Action.SPEED_UP},
    imgui.Key.keypad_subtract: {Action.SPEED_DOWN},
    imgui.Key.minus: {Action.SPEED_DOWN},
    imgui.Key.page_down: {Action.SPEED_DOWN},
    imgui.Key.f2: {Action.SCREENSHOT},
    imgui.Key.print_screen: {Action.SCREENSHOT},
    imgui.Key.f11: {Action.FULLSCREEN},
    imgui.Key.f12: {Action.TILE_WINDOWS},
    imgui.Key.home: {Action.TILE_WINDOWS},
    imgui.Key.left_alt: {Action.CYCLE_CAMERA_CONTROLS},
    imgui.Key.right_alt: {Action.CYCLE_CAMERA_CONTROLS},
    imgui.Key.end: {Action.CYCLE_CAMERA_CONTROLS},
    imgui.Key.escape: {Action.CANCEL_MOUSE_CAPTURE, Action.EXIT_FULLSCREEN},
    imgui.Key.f1: {'SHOW_CONFIGURATION'},
    imgui.Key.c: {'SHOW_CONFIGURATION'},
    imgui.Key.g: {'TOGGLE_GROUND_TRUTH'},
    imgui.Key.j: {'JUMP_TO_CLOSEST_POSE'},
    imgui.Key.p: {'PAUSE_ANIMATION'},
    imgui.Key.pause: {'PAUSE_ANIMATION'},
})

# Ordered dict keeps precedence of the modifiers consistent
DEFAULT_KEYMAP: OrderedDict[Modifier, OrderedDict[int, set[Action | str]]] = OrderedDict({
    Modifier.CTRL: __MODIFIER_CTRL_KEYMAP,
    Modifier.SHIFT: OrderedDict(),
    Modifier.ALT: OrderedDict(),
    Modifier.NONE: __MODIFIER_NONE_KEYMAP,
})

keymap_warnings_emitted = set()
