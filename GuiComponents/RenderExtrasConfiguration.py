# -- coding: utf-8 --

"""GuiComponents/RenderExtrasConfiguration.py: Configuration for the GUI extras (Bounding Box, Frustrums, etc.)"""

import imgui
from sdl2 import SDL_SCANCODE_F5, SDL_SCANCODE_E

from ICGui.GuiComponents.Base import GuiComponent
from ICGui.GuiConfig import NeRFConfig
from ICGui.Viewers.Base.Extras import BaseExtra


class RenderExtrasConfiguration(GuiComponent):
    """Informational component for the GUI, showing FPS and hotkeys."""
    _HOTKEYS = [SDL_SCANCODE_F5, SDL_SCANCODE_E]

    def __init__(self, config: NeRFConfig):
        super().__init__('Render Extras Config', RenderExtrasConfiguration._HOTKEYS, config=config,
                         closable=True, default_open=False)
        self._extras_enabled: dict[str, bool] = {}

    # pylint: disable=arguments-differ
    def _render(self, extras: list[BaseExtra]):
        """Renders the GUI component."""
        for extra in extras:
            if extra.name not in self._extras_enabled:
                self._extras_enabled[extra.name] = False

            _, self._extras_enabled[extra.name] = imgui.checkbox(extra.name, self._extras_enabled[extra.name])

            if self._extras_enabled[extra.name]:
                extra.renderOptions()
                imgui.spacing()

        imgui.spacing()
        _, self._config.screenshot.include_extras = imgui.checkbox(
            'Include Extras in Screenshots',
            self._config.screenshot.include_extras
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip('! Incompatible with custom screenshot resolutions !')

    @property
    def extras_enabled(self) -> dict[str, bool]:
        """Returns a dictionary of enabled states for each extra rendering."""
        return self._extras_enabled
