"""Components/ConfigSections/ScreenshotSection.py: Screenshot configuration section for the config window."""

from dataclasses import dataclass

from imgui_bundle import imgui, icons_fontawesome_6 as fa

from ICGui.Components.HelpIndicator import help_indicator
from ICGui.Components.StyledToggle import styled_toggle
from ICGui.State.Volatile import GlobalState, ScreenshotState
from .Section import Section


@dataclass
class ScreenshotSection(Section):
    name: str = f'{fa.ICON_FA_IMAGE} Screenshot'
    always_open: bool = False
    default_open: bool = False

    def __post_init__(self):
        super().__post_init__()

    def _render(self):
        self._render_screenshot_button()
        self._render_screenshot_settings()

    @staticmethod
    def _render_screenshot_settings():
        screenshots = ScreenshotState()
        _, screenshots.include_overlays = styled_toggle(
            'Include Overlays',
            screenshots.include_overlays
        )
        help_indicator('Include overlays from the overlay settings (e.g. camera frustums) '
                       'in the screenshot. Otherwise the raw model output is saved. '
                       'Not compatible with overriding the screenshot resolution.')

        if screenshots.include_overlays:
            return

        _, override_resolution = styled_toggle(
            'Override Screenshot Resolution',
            screenshots.resolution_override is not None
        )
        help_indicator('Not compatible with including overlays in screenshots.')

        if override_resolution:
            if screenshots.resolution_override is None:
                screenshots.resolution_override = GlobalState().window_size

            _, resolution = imgui.input_int2('Screenshot Resolution',
                                             list(screenshots.resolution_override))
            screenshots.resolution_override = (max(1, resolution[0]), max(1, resolution[1]))
        else:
            screenshots.resolution_override = None

    @staticmethod
    def _render_screenshot_button():
        if imgui.button(f'{fa.ICON_FA_CAMERA} Take Screenshot'):
            ScreenshotState().take()
