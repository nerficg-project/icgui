"""Components/Colors.py: Constants defining UI colors and function to help apply them."""

from contextlib import contextmanager

from imgui_bundle import imgui


COLORS: dict[str, imgui.ImVec4] = {
    'ERROR': imgui.ImColor.hsv(0.961, 0.557, 0.310).value,
    'ERROR_BRIGHT': imgui.ImColor.hsv(0.961, 0.557, 0.510).value,
    'ERROR_STRONG': imgui.ImColor.hsv(0.961, 0.757, 0.310).value,
    'ERROR_TEXT': imgui.ImColor.hsv(0.961, 0.557, 0.710).value,
    'WARNING': imgui.ImColor.hsv(0.131, 0.557, 0.310).value,
    'WARNING_BRIGHT': imgui.ImColor.hsv(0.131, 0.557, 0.510).value,
    'WARNING_STRONG': imgui.ImColor.hsv(0.131, 0.757, 0.310).value,
    'ACCEPT': imgui.ImColor.hsv(0.300, 0.557, 0.310).value,
    'ACCEPT_BRIGHT': imgui.ImColor.hsv(0.300, 0.557, 0.510).value,
    'ACCEPT_STRONG': imgui.ImColor.hsv(0.300, 0.757, 0.310).value,
    'MODIFIED': imgui.ImColor.hsv(0.153, 0.557, 0.310).value,
}

UI_COLORS: dict[str, dict[imgui.Col_, imgui.ImVec4]] = {
    'ERROR': {
        imgui.Col_.frame_bg: COLORS['ERROR'],
        imgui.Col_.button: COLORS['ERROR'],
        imgui.Col_.button_hovered: COLORS['ERROR_BRIGHT'],
        imgui.Col_.button_active: COLORS['ERROR_STRONG'],
        imgui.Col_.text_disabled: COLORS['ERROR_TEXT'],
    },
    'WARNING': {
        imgui.Col_.frame_bg: COLORS['WARNING'],
        imgui.Col_.button: COLORS['WARNING'],
        imgui.Col_.button_hovered: COLORS['WARNING_BRIGHT'],
        imgui.Col_.button_active: COLORS['WARNING_STRONG'],
    },
    'ACCEPT': {
        imgui.Col_.frame_bg: COLORS['ACCEPT'],
        imgui.Col_.button: COLORS['ACCEPT'],
        imgui.Col_.button_hovered: COLORS['ACCEPT_BRIGHT'],
        imgui.Col_.button_active: COLORS['ACCEPT_STRONG'],
    },
    'MODIFIED': {
        imgui.Col_.frame_bg: COLORS['MODIFIED'],
    },
    'DEFAULT': {},
}


@contextmanager
def apply_ui_color(key: str):
    """Context manager to temporarily set the given UI colors"""
    try:
        for style_key, color in UI_COLORS[key].items():
            imgui.push_style_color(style_key, color)
        yield
    finally:
        imgui.pop_style_color(len(UI_COLORS[key].items()))
