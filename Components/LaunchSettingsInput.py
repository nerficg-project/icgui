"""Components/LaunchSettingsInput.py: Single dispatch functions for automatically rendering the appropriate
input field for launch settings."""

import math
from dataclasses import Field, MISSING
from functools import singledispatch
from pathlib import Path
from typing import Any, Collection, Mapping, TypeVar

from imgui_bundle import imgui, imgui_toggle, icons_fontawesome_6

from ICGui.Backend.FontManager import FontManager, FontSpec
from ICGui.Components.FileInput import file_input
from ICGui.Components.FontInput import font_input
from ICGui.Components.StyledToggle import styled_toggle
from ICGui.util.Validation import highlight_invalid

try:
    from matplotlib import font_manager
except ImportError:
    font_manager = None


_Number = int | float
_T = TypeVar('_T')
@singledispatch
def settings_input(value: _T, metadata: Mapping[str, Any]) -> _T | None:
    """Single dispatch function to render the appropriate input field based on the type of value."""
    raise NotImplementedError('No input implemented for this type')


def resettable_input(value: _T, field: Field, locked: bool = False):
    """Renders a settings input field with a reset button to change settings back to their default."""
    reset = False
    if (field.default is not MISSING or field.default_factory is not MISSING) and not locked:
        # Render reset button
        reset = imgui.button(icons_fontawesome_6.ICON_FA_ARROWS_ROTATE + '##reset_' + field.name)
        if imgui.is_item_hovered():
            imgui.begin_tooltip()
            imgui.text('Reset to default')
            imgui.end_tooltip()
        imgui.same_line()

    # Set the input field as disabled if locked, and render the input
    try:
        imgui.begin_disabled(locked)
        result = settings_input(value, field.metadata)
    finally:
        # Always end the disabled state, even if an exception is raised
        imgui.end_disabled()

    # If reset is pressed, return the default value
    if reset and field.default is not MISSING:
        return field.default
    elif reset and field.default_factory is not MISSING:
        return field.default_factory()
    return result


@settings_input.register
def _(value: bool, metadata: Mapping[str, Any]):
    """Shows a toggle input."""
    label = metadata.get('name', '!!MISSING NAME!!')

    changed, value = styled_toggle(label, value)
    if changed:
        return value
    else:
        return None


@settings_input.register
def _(value: str, metadata: Mapping[str, Any]):
    """Shows a text input field."""
    label = metadata.get('name', '!!MISSING NAME!!')

    # Apply error styles if the value is invalid
    with highlight_invalid(value, metadata, error_styles=(imgui.Col_.frame_bg,)):
        changed, value = imgui.input_text(label, value)

    if changed:
        return value
    else:
        return None


@settings_input.register
def _(value: _Number, metadata: Mapping[str, Any]):
    """Shows a numeric input field."""
    # Extract arguments
    label = metadata.get('name', '!!MISSING NAME!!')
    min_val: _Number = metadata.get('min', -math.inf)
    max_val: _Number = metadata.get('max', math.inf)
    input_style, args = _get_input_args(metadata)
    input_type: str = type(value).__name__.lower()  # 'int' or 'float'

    # Get corresponding imgui function
    input_func = imgui.__dict__[f'{input_style}_{input_type}']
    # Apply error styles if the value is invalid
    with highlight_invalid(value, metadata, error_styles=(imgui.Col_.frame_bg,)):
        changed, value = input_func(label, value, *args)

    if changed:
        return min(max(value, min_val), max_val)  # Clamp value
    return None


@settings_input.register
def _(value: list, metadata: Mapping[str, Any]):
    """Shows a numeric list input field."""
    # Extract arguments
    assert 1 < len(value) <= 4, f'Invalid number of values: {len(value)}. Expected between 2 and 4.'
    label = metadata.get('name', '!!MISSING NAME!!')
    min_val: _Number = metadata.get('min', -math.inf)
    max_val: _Number = metadata.get('max', math.inf)
    input_style, args = _get_input_args(metadata, multiple=True)
    input_type: str = type(value[0]).__name__.lower()  # 'int' or 'float'
    assert input_type in ('int', 'float'), f'Invalid input type: {input_type}'

    # Get corresponding imgui function
    input_func = imgui.__dict__[f'{input_style}_{input_type}{len(value)}']
    # Apply error styles if the value is invalid
    with highlight_invalid(value, metadata, error_styles=(imgui.Col_.frame_bg,)):
        changed, value = input_func(label, value, *args)

    if changed:
        return list(map(lambda v: min(max(v, min_val), max_val), value))  # Clamp values
    return None


# Note: Currently, only Path options can be None, if other options are added in the future, we have to do more
#       thorough type checking in case of None
@settings_input.register
def _(value: Path | None, metadata: Mapping[str, Any]) -> Path | None:
    """Shows a file path input field."""
    label = metadata.get('name', '!!MISSING NAME!!')
    valid_extensions: Collection[str] = metadata.get('ext', ())

    # Apply error styles if the value is invalid
    with highlight_invalid(value, metadata, error_styles=(imgui.Col_.frame_bg,)):
        _, value = file_input(label, value, extensions=valid_extensions)
    return value


@settings_input.register
def _(value: FontSpec, metadata: Mapping[str, Any]) -> FontSpec | None:
    """Shows a font specification input field."""
    label = metadata.get('name', '!!MISSING NAME!!')

    # Apply error styles if the value is invalid
    with highlight_invalid(value, metadata, error_styles=(imgui.Col_.frame_bg,)):
        changed, value = font_input(label, value)

    if changed:
        return value
    else:
        return None


def _get_input_args(metadata: Mapping[str, Any], multiple=False) -> tuple[str, list[_Number]]:
    """Extracts the appropriate arguments for the input field based on the metadata."""
    label = metadata.get('name', '!!MISSING NAME!!')
    min_val: _Number = metadata.get('min', -math.inf)
    max_val: _Number = metadata.get('max', math.inf)
    step: _Number = metadata.get('step', 1)
    step_fast: _Number = metadata.get('step_fast', 10)
    input_style: str = metadata.get('input_style', 'input')  # 'input', 'slider', 'drag'
    assert input_style in ('input', 'slider', 'drag'), f'Invalid input style: {input_style}'

    if input_style == 'input':
        if multiple:
            return input_style, []  # Multi-Inputs don't support steps
        return input_style, [step, step_fast]
    elif input_style == 'slider':
        return input_style, [min_val, max_val]
    elif input_style == 'drag':
        return input_style, [step, min_val, max_val]
    else:
        raise ValueError(f'Invalid input style: {input_style} for label: {label}')
