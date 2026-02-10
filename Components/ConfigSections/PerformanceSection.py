"""Components/ConfigSections/PerformanceSection.py: Performance information section for the config window."""

import math
from dataclasses import dataclass

from imgui_bundle import imgui, icons_fontawesome_6 as fa

import Framework
from ICGui.Backend import FontManager
from ICGui.Components.ButtonBehavior import button_behavior
from ICGui.State.Volatile import GlobalState
from .Section import Section
from ..HelpIndicator import help_indicator


@dataclass
class PerformanceSection(Section):
    name: str = f'{fa.ICON_FA_STOPWATCH} Performance'
    always_open: bool = True
    default_open: bool = False

    _fps_as_ms: bool = False
    _training_percentage: bool = False

    def _render(self):
        self._render_fps()
        self._render_training_info()

    def _render_fps(self):
        state = GlobalState()

        with button_behavior('GUI FPS') as button:
            if self._fps_as_ms:
                imgui.text('Frame times (GUI):')
                imgui.same_line()
                FontManager.muted_color(imgui.text)(f'{imgui.get_io().delta_time * 1000:.2f}ms')
            else:
                imgui.text('FPS (GUI):')
                imgui.same_line()
                FontManager.muted_color(imgui.text)(f'{imgui.get_io().framerate:.2f}')
        self._fps_as_ms ^= button.pressed

        with button_behavior('Model FPS') as button:
            if self._fps_as_ms:
                imgui.text('Frame times (Model):')
                imgui.same_line()
                if state.model_frametime == 'N/A':
                    fps_text = 'N/A'
                else:
                    fps_text = f'{state.model_frametime} ± {state.model_frametime_std}'
                FontManager.muted_color(imgui.text)(fps_text)
            else:
                imgui.text('FPS (Model):')
                imgui.same_line()
                if math.isnan(state.model_framerate):
                    fps_text = 'N/A'
                else:
                    fps_text = f'{state.model_framerate:.2f} ± {state.model_framerate_std:.2f}'
                FontManager.muted_color(imgui.text)(fps_text)
            help_indicator('Measured by taking the average frame time over the last '
                           'n frames as specified in the launcher. Click to toggle '
                           'between FPS and frame time.')
        self._fps_as_ms ^= button.pressed

    def _render_training_info(self):
        state = GlobalState()
        is_training = state.shared.is_training
        training_iter = state.shared.training_iteration

        if is_training and training_iter is not None and Framework.config.TRAINING.NUM_ITERATIONS:
            progress = training_iter / Framework.config.TRAINING.NUM_ITERATIONS
            if self._training_percentage:
                overlay_text = f'{progress * 100:.1f}%'
            else:
                if training_iter >= Framework.config.TRAINING.NUM_ITERATIONS:
                    overlay_text = 'Processing post-training callbacks...'
                else:
                    overlay_text = f'{training_iter} / {Framework.config.TRAINING.NUM_ITERATIONS}'

            with button_behavior('Training Progress') as button:
                imgui.progress_bar(progress, overlay=overlay_text, size_arg=(0, 0))
                imgui.same_line()
                imgui.text('Training Progress')
            self._training_percentage ^= button.pressed
        else:
            imgui.text(fa.ICON_FA_BARS_PROGRESS)
            imgui.same_line()
            FontManager.muted_color(imgui.text)('Not Training')
