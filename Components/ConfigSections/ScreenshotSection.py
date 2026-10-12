"""Components/ConfigSections/ScreenshotSection.py: Screenshot configuration section for the config window."""
from copy import deepcopy
from dataclasses import dataclass, field

from imgui_bundle import imgui, icons_fontawesome_6 as fa

from Cameras.Perspective import PerspectiveCamera
from Datasets.utils import View
from ICGui.Components.HelpIndicator import help_indicator
from ICGui.Components.StyledToggle import styled_toggle
from ICGui.State.Volatile import GlobalState, CameraState, ViewerOutputState, ColorMapState
from ICGui.util.Enums import Action
from ICGui.util.Screenshots import get_screenshot_path
from .Section import Section


@dataclass
class ScreenshotSection(Section):
    name: str = f'{fa.ICON_FA_IMAGE} Screenshot'
    always_open: bool = False
    default_open: bool = False

    _include_overlays: bool = field(init=False, default=False)
    _resolution_override: tuple[int, int] | None = field(init=False, default=None)

    def __post_init__(self):
        super().__post_init__()

    def update(self):
        hotkey_pressed = GlobalState().input_manager.is_pressed(Action.SCREENSHOT)
        if hotkey_pressed:
            self._take_screenshot()

    def _render(self):
        if imgui.button(f'{fa.ICON_FA_CAMERA} Take Screenshot'):
            self._take_screenshot()
        self._render_screenshot_settings()

    def _render_screenshot_settings(self):
        _, self._include_overlays = styled_toggle(
            'Include Overlays',
            self._include_overlays
        )
        help_indicator('Include overlays from the overlay settings (e.g. camera frustums) '
                       'in the screenshot. Otherwise the raw model output is saved. '
                       'Not compatible with overriding the screenshot resolution.')

        # Other settings are incompatible with including overlays
        if self._include_overlays:
            return

        _, override_resolution = styled_toggle(
            'Override Screenshot Resolution',
            self._resolution_override is not None
        )
        help_indicator('Not compatible with including overlays in screenshots.')

        if override_resolution:
            # On first enabling, set override resolution to current window size
            if self._resolution_override is None:
                self._resolution_override = GlobalState().window_size

            _, resolution = imgui.input_int2('Screenshot Resolution', list(self._resolution_override))  # type: ignore
            self._resolution_override = (max(1, resolution[0]), max(1, resolution[1]))  # Clamp to > 0
        else:
            self._resolution_override = None

    def _take_screenshot(self):
        state = GlobalState()
        if self._include_overlays:
            state.viewer.save_screenshot(get_screenshot_path('overlays'))  # type: ignore
        else:
            # Request a clean render of the current frame to be saved
            camera_state = CameraState()
            if self._resolution_override is None:
                screenshot_view: View = camera_state.current_view  # type: ignore
            else:
                screenshot_view: View = deepcopy(camera_state.current_view)  # type: ignore
                if isinstance(camera_state.current_view.camera, PerspectiveCamera):  # type: ignore
                    unscaled_cam: PerspectiveCamera = camera_state.unscaled_camera  # type: ignore
                    screenshot_cam: PerspectiveCamera = screenshot_view.camera  # type: ignore

                    factor_x = self._resolution_override[0] / camera_state.current_camera.width
                    factor_y = self._resolution_override[1] / camera_state.current_camera.height
                    screenshot_cam.center_x = (unscaled_cam.center_x - unscaled_cam.width / 2) * factor_x + \
                                              self._resolution_override[0] / 2
                    screenshot_cam.center_y = (unscaled_cam.center_y - unscaled_cam.height / 2) * factor_y + \
                                              self._resolution_override[1] / 2
                screenshot_view.camera.width = self._resolution_override[0]
                screenshot_view.camera.height = self._resolution_override[1]

            state.shared.screenshot(  # type: ignore
                screenshot_view,
                ViewerOutputState().selected_output,
                ColorMapState().dict,
            )
