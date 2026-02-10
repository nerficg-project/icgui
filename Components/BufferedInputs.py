"""Components/BufferedInput.py: Provides int / float inputs with increment decrement buttons,
which only return true, when the value has been confirmed (button pressed or input field deactivated).
This custom component is required, since ImGui no longer support the ReturnTrueOnEnter flag for scalar
inputs, and IsItemDeactivated() does not work correctly with the increment and decrement buttons.
"""

from typing import Callable, TypeVar

from imgui_bundle import imgui


_input_buffer: dict[str, int] = {}
_was_active: dict[str, bool] = {}


def buffered_input_float(label: str, v: float, step: float = 0.1, step_fast: float = 1.0,
                         flags: int = 0, fmt: str = None) -> tuple[bool, float]:
    """Renders an input field for a float with buffering, returning the changed value only when the input is deactivated."""
    return _buffered_input(imgui.input_float, label, v,
                           step=step, step_fast=step_fast,
                           flags=flags, fmt=fmt)


def buffered_input_int(label: str, v: int, step: int = 1, step_fast: int = 10,
                       flags: int = 0, fmt: str = None) -> tuple[bool, int]:
     """Renders an input field for an integer with buffering, returning the changed value only when the input is deactivated."""
     return _buffered_input(imgui.input_int, label, v,
                            step=step, step_fast=step_fast,
                            flags=flags, fmt=fmt)


_T = TypeVar('_T', int, float)

def _buffered_input(input_func: Callable[..., tuple[bool, _T]], label: str, v: _T,
                    step: _T = 1, step_fast: _T = 10, flags: int = 0, fmt: str = None) -> tuple[bool, _T]:
    """Renders an input field for a given input function with buffering."""
    if not _was_active.setdefault(label, False):
        _input_buffer[label] = v

    # Calculate the width of the input field, and the buttons
    button_width = imgui.calc_text_size('+').x + 2 * imgui.get_style().frame_padding.x
    input_width = imgui.calc_item_width()
    input_width -= 2 * imgui.get_style().item_inner_spacing.x
    input_width -= 2 * button_width

    # Render the input field
    imgui.push_item_width(input_width)
    opt = {'format': fmt} if fmt is not None else {}
    _, _input_buffer[label] = input_func(f'##input_{label}', _input_buffer[label],
                                         step=0, step_fast=0, flags=flags, **opt)
    imgui.pop_item_width()
    imgui.same_line(spacing=imgui.get_style().item_inner_spacing.x)

    # Get input state
    _was_active[label] = imgui.is_item_active()
    return_changed = imgui.is_item_deactivated() and v != _input_buffer[label]

    # Render the increment and decrement buttons
    imgui.push_item_flag(imgui.ItemFlags_.button_repeat, True)
    if imgui.button(f'-##{label}', (button_width, 0)):
        if imgui.is_key_pressed(imgui.Key.mod_ctrl):
            _input_buffer[label] -= step_fast
        else:
            _input_buffer[label] -= step
        return_changed = True
    imgui.same_line(spacing=imgui.get_style().item_inner_spacing.x)

    if imgui.button(f'+##{label}', (button_width, 0)):
        if imgui.is_key_pressed(imgui.Key.mod_ctrl):
            _input_buffer[label] += step_fast
        else:
            _input_buffer[label] += step
        return_changed = True
    imgui.pop_item_flag()
    imgui.same_line(spacing=imgui.get_style().item_inner_spacing.x)

    imgui.text(label)

    return return_changed, _input_buffer[label]
