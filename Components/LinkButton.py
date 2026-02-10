"""Components/LinkButton.py: Provides a button component that opens a URL in the user's browser."""

import webbrowser

from imgui_bundle import imgui

from ICGui.Components.ButtonBehavior import button_behavior


def link_button(label: str, url: str, size: imgui.ImVec2 = None) -> bool:
    """Renders a button to open a specific URL in the user's web browser. Returns True if pressed."""
    if imgui.button(label, size=size):
        webbrowser.open(url)
        return True
    return False


def link_text(label: str, url: str) -> bool:
    """Renders a text link to open a specific URL in the user's web browser. Returns True if pressed."""
    with button_behavior(f'{label}##linkbutton') as result:
        imgui.text(label)
    if result.pressed:
        webbrowser.open(url)
        return True
    return False
