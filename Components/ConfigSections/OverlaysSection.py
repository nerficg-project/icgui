"""Components/ConfigSections/OverlaysSection.py: OpenGL extra overlays configuration section for the config window."""

from dataclasses import dataclass, field
from typing import Any

from imgui_bundle import imgui, icons_fontawesome_6 as fa

from ICGui.Components.StyledToggle import styled_toggle
from ICGui.State.Volatile import OverlayState
from .Section import Section


@dataclass
class OverlaysSection(Section):
    name: str = f'{fa.ICON_FA_LAYER_GROUP} Overlays'
    always_open: bool = False
    default_open: bool = False

    _available_settings: dict[str, dict[str, Any]] = field(default_factory=lambda: {})

    def __post_init__(self):
        super().__post_init__()

    def _render(self):
        overlays = OverlayState()

        for overlay in overlays.available_overlays:
            overlays.enabled.setdefault(overlay.name, False)
            _, overlays.enabled[overlay.name] = styled_toggle(overlay.name, overlays.enabled[overlay.name])

            if overlays.enabled[overlay.name]:
                overlay.render_options()
                imgui.spacing()
