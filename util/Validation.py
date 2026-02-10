"""util/Validation.py: Validation methods for the launch configuration."""

import math
from contextlib import contextmanager
from dataclasses import dataclass
from functools import singledispatch
from pathlib import Path
from typing import Any, Mapping, Collection

from imgui_bundle import imgui

from ICGui.Backend.FontManager import FontSpec
from ICGui.Components.Colors import UI_COLORS

try:
    from matplotlib import font_manager
except ImportError:
    font_manager = None

_Number = int | float


@dataclass
class ValidationResult:
    valid: bool
    reason: str = ''


@singledispatch
def validate_field(value: Any, metadata: Mapping[str, Any]) -> ValidationResult:
    return ValidationResult(valid=True, reason='No validation implemented for this type')


@validate_field.register
def _(value: _Number, metadata: Mapping[str, Any]) -> ValidationResult:
    """Validates a numeric input value."""
    min_val = metadata.get('min', -math.inf)
    max_val = metadata.get('max', math.inf)
    if value > max_val:
        return ValidationResult(valid=False, reason=f'Value {value} exceeds maximum {max_val}')
    if value < min_val:
        return ValidationResult(valid=False, reason=f'Value {value} is below minimum {min_val}')
    return ValidationResult(valid=True)


@validate_field.register
def _(value: list, metadata: Mapping[str, Any]) -> ValidationResult:
    """Validates a list of numeric input values."""
    for val in value:
        valid = validate_field(val, metadata)
        if not valid.valid:
            return valid
    return ValidationResult(valid=True)


@validate_field.register
def _(value: Path, metadata: Mapping[str, Any]) -> ValidationResult:
    """Validates a file path input."""
    if value is None:
        return ValidationResult(valid=False, reason='No path specified')
    if not value.exists():
        return ValidationResult(valid=False, reason=f'Path does not exist')
    if not value.is_file():
        return ValidationResult(valid=False, reason=f'Not a valid file')

    ext: str | None = metadata.get('ext', None)
    if ext is not None and len(ext) > 0 and value.suffix not in ext:
        return ValidationResult(valid=False, reason=f'Incorrect file extension, expected any of: {", ".join(ext)})')

    return ValidationResult(valid=True)


@validate_field.register
def _(value: str, metadata: Mapping[str, Any]) -> ValidationResult:
    return ValidationResult(valid=True, reason='No validation implemented for this string type')


@validate_field.register
def _(value: FontSpec, metadata: Mapping[str, Any]) -> ValidationResult:
    family_result = validate_font_family(value.name, metadata)
    if not family_result.valid:
        return family_result

    size = value.size
    if size is None or size <= 0:
        return ValidationResult(valid=False, reason='Font size must be a positive number')
    if size > 144:
        return ValidationResult(valid=False, reason='Font size is too large, maximum is 144')

    return ValidationResult(valid=True)


def validate_font_family(value: str, metadata: Mapping[str, Any]) -> ValidationResult:
    """Validates a font family input."""
    if value == 'Default':
        return ValidationResult(valid=True)  # Default font is always valid

    # Ignore font if matplotlib is not installed
    if font_manager is None:
        return ValidationResult(valid=True, reason='Matplotlib not installed, skipping font validation')

    # Can a font with the given family be found?
    try:
        font = font_manager.findfont(font_manager.FontProperties(family=value, style='normal'), fontext='ttf',
                                     fallback_to_default=False)
    except ValueError:
        return ValidationResult(valid=False, reason=f'Invalid font family: {value}')

    # Can the font file be opened?
    try:
        with open(font, 'rb') as _:
            pass
    except OSError:
        return ValidationResult(valid=False, reason=f'Unable to read font file: {font}')

    return ValidationResult(valid=True)


@contextmanager
def highlight_invalid(value, metadata, error_styles: Collection[imgui.Col_]):
    """Context manager to temporarily set the accept button colors"""
    # TODO: Show reason to the user for invalid fields
    result = validate_field(value, metadata)
    try:
        if not result.valid:
            for style in error_styles:
                imgui.push_style_color(style, UI_COLORS['ERROR'][style])
        yield
    finally:
        if not result.valid:
            imgui.pop_style_color(len(error_styles))
