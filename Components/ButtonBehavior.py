"""Components/ButtonBehavior.py: Adds button-like behavior to other components for ImGui rendering."""

from contextlib import contextmanager
from dataclasses import dataclass

from imgui_bundle import imgui


@dataclass
class ButtonResult:
    pressed: bool


@contextmanager
def button_behavior(btn_id: str):
    """Context manager to handle button-like behavior to arbitrary ImGui components. Result is available from the
    returned object *after* the context manager exits. May work unexpectedly for multi-line components,
    if the last line is shorter than the previous lines."""
    result = ButtonResult(False)
    cursor_start = imgui.get_cursor_pos()
    try:
        yield result
    finally:
        imgui.same_line(spacing=0)
        cursor_end = imgui.get_cursor_pos()
        width = cursor_end.x - cursor_start.x
        height = cursor_end.y - cursor_start.y + imgui.get_item_rect_size().y

        imgui.set_cursor_pos(cursor_start)
        result.pressed = imgui.invisible_button(f'##invisible_button_{btn_id}', (width, height))
