"""Components/ConfigSections/FoveatedSection.py: Foveation settings section for the config window."""

from dataclasses import dataclass, field, InitVar
from typing import Any

from imgui_bundle import imgui, imgui_ctx, icons_fontawesome_6 as fa

from ICGui.Components.BufferedInputs import buffered_input_int
from ICGui.Components.ConfigSections.AdvancedSection import AdvancedSection
from ICGui.Components.StyledToggle import styled_toggle
from ICGui.State.Volatile import GlobalState
from .Section import Section
from ..HelpIndicator import help_indicator


@dataclass
class FoveatedSection(Section):
    advanced_section: InitVar[AdvancedSection] = None

    name: str = f'{fa.ICON_FA_EYE} Foveated Rendering'
    always_open: bool = False
    default_open: bool = False
    _available_settings: dict[str, dict[str, Any]] = field(init=False)

    def __post_init__(self, advanced_section: AdvancedSection):
        super().__post_init__()
        # Foveation settings are handled by the Renderer, so we need access to
        #   its settings. To avoid desynchronization, we share the same dictionary
        #   reference with the Advanced Settings Section.
        self._available_settings = advanced_section._available_settings
        advanced_section.ignore_prefix += ('Renderer.FOVEATION_',)

    def _render_foveation_section(self):
        renderer_settings = self._available_settings.get('Renderer', {})
        if 'Renderer.FOVEATION_ENABLED' in renderer_settings:
            changed, val = styled_toggle('Enable Foveation', renderer_settings['Renderer.FOVEATION_ENABLED'])
            if changed:
                self._available_settings['Renderer']['Renderer.FOVEATION_ENABLED'] = val
                GlobalState().shared.configurable_changes = {
                    'Renderer.FOVEATION_ENABLED': val,
                }
        if not renderer_settings.get('Renderer.FOVEATION_ENABLED', False):
            return

        # Visibility Mask Selection
        if 'Renderer.FOVEATION_VISIBILITY_MASK_INDEX' in renderer_settings:
            changed, val = buffered_input_int('Visibility Mask Index', renderer_settings['Renderer.FOVEATION_VISIBILITY_MASK_INDEX'])
            help_indicator('Use the visibility mask from the visibility_masks subdirectory of the Method folder. '
                           'Generates a circular mask if index < 0 or no masks exist in the expected directory.')
            if changed:
                # Only allow non-negative integers and -1 (implies mask disabled)
                if val < -1:
                    val = -1
                self._available_settings['Renderer']['Renderer.FOVEATION_VISIBILITY_MASK_INDEX'] = val
                GlobalState().shared.configurable_changes = {
                    'Renderer.FOVEATION_VISIBILITY_MASK_INDEX': val,
                }

    def _render_visualization_section(self):
        renderer_settings = self._available_settings.get('Renderer', {})
        if not renderer_settings.get('Renderer.FOVEATION_ENABLED', False):
            return

        if 'Renderer.FOVEATION_VISUALIZE_TILE_GRID' in renderer_settings:
            changed, val = styled_toggle('Visualize Tile Grid', renderer_settings['Renderer.FOVEATION_VISUALIZE_TILE_GRID'])
            if changed:
                self._available_settings['Renderer']['Renderer.FOVEATION_VISUALIZE_TILE_GRID'] = val
                GlobalState().shared.configurable_changes = {
                    'Renderer.FOVEATION_VISUALIZE_TILE_GRID': val,
                }
        if 'Renderer.FOVEATION_VISUALIZE_GAZE' in renderer_settings:
            changed, val = styled_toggle('Visualize Gaze', renderer_settings['Renderer.FOVEATION_VISUALIZE_GAZE'])
            if changed:
                self._available_settings['Renderer']['Renderer.FOVEATION_VISUALIZE_GAZE'] = val
                GlobalState().shared.configurable_changes = {
                    'Renderer.FOVEATION_VISUALIZE_GAZE': val,
                }
        if not renderer_settings['Renderer.FOVEATION_VISUALIZE_GAZE']:
            return

        if 'Renderer.FOVEATION_VISUALIZATION_TYPE' in renderer_settings:
            changed, selected_indicator = imgui.combo('Gaze Indicator', renderer_settings['Renderer.FOVEATION_VISUALIZATION_TYPE'],
                                                      ['Circle', 'Square', 'Crosshair'])
            if changed:
                self._available_settings['Renderer']['Renderer.FOVEATION_VISUALIZATION_TYPE'] = selected_indicator
                GlobalState().shared.configurable_changes = {
                    'Renderer.FOVEATION_VISUALIZATION_TYPE': selected_indicator,
                }
        if 'Renderer.FOVEATION_VISUALIZATION_SIZE' in renderer_settings:
            changed, val = buffered_input_int('Gaze Indicator Size', renderer_settings['Renderer.FOVEATION_VISUALIZATION_SIZE'])
            if changed:
                if val < 1:
                    val = 1
                self._available_settings['Renderer']['Renderer.FOVEATION_VISUALIZATION_SIZE'] = val
                GlobalState().shared.configurable_changes = {
                    'Renderer.FOVEATION_VISUALIZATION_SIZE': val,
                }
        if 'Renderer.FOVEATION_VISUALIZATION_COLOR' in renderer_settings:
            changed, val = imgui.color_edit3('Gaze Indicator Color', renderer_settings['Renderer.FOVEATION_VISUALIZATION_COLOR'])
            if changed:
                self._available_settings['Renderer']['Renderer.FOVEATION_VISUALIZATION_COLOR'] = val
                GlobalState().shared.configurable_changes = {
                    'Renderer.FOVEATION_VISUALIZATION_COLOR': val,
                }

    def _render(self):
        with imgui_ctx.push_id('foveated_section'):
            self._render_foveation_section()
            self._render_visualization_section()

    def render(self):
        # Do not render the section if the FOVEATION_ENABLED key does not exist
        if 'Renderer' not in self._available_settings or 'Renderer.FOVEATION_ENABLED' not in self._available_settings['Renderer']:
            return
        super().render()
