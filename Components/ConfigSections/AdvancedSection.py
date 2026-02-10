"""Components/ConfigSections/AdvancedSection.py: Advanced settings section for the config window."""

from dataclasses import dataclass, field
from functools import singledispatchmethod
from typing import Any

from imgui_bundle import imgui, icons_fontawesome_6 as fa

import Framework
from ICGui.Components.BufferedInputs import buffered_input_int, buffered_input_float
from ICGui.Components.StyledToggle import styled_toggle
from ICGui.State.Volatile import GlobalState
from .Section import Section


@dataclass
class AdvancedSection(Section):
    name: str = f'{fa.ICON_FA_TOOLBOX} Advanced Settings'
    always_open: bool = False
    default_open: bool = False

    _available_settings: dict[str, dict[str, Any]] = field(init=False, default_factory=lambda: {})
    _render_interval: int = field(init=False, default_factory=lambda: Framework.config.TRAINING.GUI.RENDER_INTERVAL)

    def __post_init__(self):
        super().__post_init__()

    def _render(self):
        if GlobalState().shared.is_training:
            self._render_training_settings()

        self._render_renderer_settings()

    def _render_training_settings(self):
        changed, self._render_interval = imgui.slider_int(
            'Render Interval', self._render_interval,
            1, 50
        )
        if changed:
            GlobalState().shared.configurable_changes = {
                'CallbackStride._gui_render_frame': self._render_interval,
            }

    def _render_renderer_settings(self):
        imgui.separator_text('Renderer')
        self._available_settings.setdefault('Renderer', {})

        changes = {}
        for key, value in self._available_settings['Renderer'].items():
            key_str = key.replace('Renderer.', '').replace('_', ' ').title()
            changed, value = self._render_setting(value, label=key_str)

            if changed:
                changes[key] = value
                self._available_settings['Renderer'][key] = value
        if changes:
            GlobalState().shared.configurable_changes = changes

    def _update_available_settings(self):
        new_configurables = GlobalState().shared.configurable_advertisements

        if new_configurables is None:
            return

        self._available_settings.setdefault('Renderer', {})
        for key, value in new_configurables:
            if key not in self._available_settings['Renderer']:
                self._available_settings['Renderer'][key] = value
            else:
                self._available_settings['Renderer'][key] = value

    @singledispatchmethod
    def _render_setting(self, value: Any, /, *, label: str) -> tuple[bool, Any]:
        """Renders a setting based on its type."""
        raise NotImplementedError(f'Unsupported type for setting {label}: {type(value)}')

    @_render_setting.register
    def _(self, value: bool, /, *, label: str) -> tuple[bool, bool]:
        """Renders a boolean setting."""
        return styled_toggle(label, value)

    @_render_setting.register
    def _(self, value: int, /, *, label: str) -> tuple[bool, int]:
        """Renders an integer setting."""
        return buffered_input_int(label, value)

    @_render_setting.register
    def _(self, value: float, /, *, label: str) -> tuple[bool, float]:
        """Renders a float setting."""
        return buffered_input_float(label, value)

    @_render_setting.register
    def _(self, value: str, /, *, label: str) -> tuple[bool, str]:
        """Renders a string setting."""
        return imgui.input_text(label, value, flags=imgui.InputTextFlags_.enter_returns_true)

    def render(self):
        self._update_available_settings()

        # Do not render the section if there are no available settings
        if sum(len(self._available_settings[key]) for key in self._available_settings) < 1:
            return
        super().render()
