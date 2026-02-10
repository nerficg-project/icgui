"""Components/ConfigSections/ResolutionSection.py: Resolution / resizing configuration section for the config window."""

from dataclasses import dataclass

from imgui_bundle import imgui, icons_fontawesome_6 as fa

from ICGui.Components.BufferedInputs import buffered_input_float
from ICGui.Components.HelpIndicator import help_indicator
from ICGui.Components.StyledToggle import styled_toggle
from ICGui.State.Volatile import GlobalState, ResolutionScaleState
from .Section import Section


@dataclass
class ResolutionSection(Section):
    name: str = f'{fa.ICON_FA_EXPAND} Resolution'
    always_open: bool = False
    default_open: bool = False

    def __post_init__(self):
        super().__post_init__()

    def _render(self):
        self._render_resolution()
        self._render_resolution_scaling()
        self._render_filtering()
        self._render_adaptive_resolution_scaling()

    @staticmethod
    def _render_resolution():
        _, value = imgui.input_int2('Resolution', list(GlobalState().window_size))
        if imgui.is_item_deactivated_after_edit():
            min_res = (64, 36)  # Minimum resolution
            resolution = (
                max(min_res[0], value[0]),
                max(min_res[1], value[1])
            )
            GlobalState().backend_window.resize_window(*resolution)
        help_indicator('Allows changing the window resolution to precise values. '
                       'Note that this may not work on all platforms.')

    @staticmethod
    def _render_resolution_scaling():
        scaling = ResolutionScaleState()

        # Scaling factor
        changed, resolution_factor = imgui.slider_float(
            'Resolution Factor', scaling.factor,
            scaling.min_scale, scaling.max_scale,
        )
        if changed:
            scaling.factor = max(scaling.min_scale, min(scaling.max_scale, resolution_factor))
        help_indicator('The resolution factor of the model, i.e. if the window '
                       'resolution is 800x400, setting the factor to 0.5 causes '
                       'the model to render at 400x200. This does not affect '
                       'the window size.')

    @staticmethod
    def _render_adaptive_resolution_scaling():
        scaling = ResolutionScaleState()

        # Toggle adaptive resolution scaling
        _, scaling.adaptive = styled_toggle('Adaptive Resolution Scaling', scaling.adaptive)
        help_indicator('Automatically adjusts the resolution factor to maintain '
                       'a target FPS. Note that this may not work well with some '
                       'rendering methods, such as those based on 3DGS, since overdraw '
                       'may result in worse performance at lower resolutions.')

        if not scaling.adaptive:
            return

        # Adaptive resolution scaling options
        changed, target_fps = buffered_input_float(
            'Target FPS', scaling.target_fps,
            step=1.0, step_fast=5.0, fmt='%.0f',
        )
        if changed:
            scaling.target_fps = max(5.0, target_fps)

        _, scaling.max_adaptive_scale = imgui.slider_float(
            'Max Scale',
            scaling.max_adaptive_scale,
            scaling.min_scale, scaling.max_scale
        )
        help_indicator('Even when the target FPS is exceeded, do not exceed this scaling factor.')

        _, dampening = imgui.slider_float(
            'Dampening',
            1.0 - scaling.dampening,
            0.0, 1.0
        )
        scaling.dampening = 1.0 - dampening
        help_indicator('Higher dampening makes resolution changes smoother, but '
                       'react more slowly to fluctuating performance.')

    @staticmethod
    def _render_filtering():
        state = ResolutionScaleState()
        changed, minify = imgui.combo('Minification Filter', state.minification_filter, state.MINIFICATION_FILTER_NAMES)
        if changed:
            state.minification_filter = minify
        help_indicator('The texture filtering method to use when the rendered image is downscaled '
                       'to fit the window.')

        changed, magnify = imgui.combo('Magnification Filter', state.magnification_filter, state.MAGNIFICATION_FILTER_NAMES)
        if changed:
            state.magnification_filter = magnify
        help_indicator('The texture filtering method to use when the rendered image is upscaled '
                       'to fit the window.')
