"""Components/StyledToggle.py: Provides a pre-styled toggle input for the GUI."""

from imgui_bundle import imgui, imgui_toggle


_toggle_style = None


def styled_toggle(label: str, value: bool) -> tuple[bool, bool]:
    """Renders a styled toggle button. Returns (changed, value)."""
    global _toggle_style
    if _toggle_style is None:
        _toggle_style = imgui_toggle.ios_style(0.16)
        _toggle_style.on.palette.frame = imgui.get_style_color_vec4(imgui.Col_.button)
        _toggle_style.on.palette.frame_hover = imgui.get_style_color_vec4(imgui.Col_.button_hovered)
        _toggle_style.off.palette.frame = imgui.ImVec4(0.5, 0.5, 0.5, 1.0)
        _toggle_style.off.palette.frame_hover = imgui.ImVec4(0.68, 0.69, 0.71, 1.0)

    imgui.push_style_var(imgui.StyleVar_.frame_padding, (0.0, 0.0))
    changed, value = imgui_toggle.toggle(label, value, config=_toggle_style)
    imgui.pop_style_var()

    return changed, value

