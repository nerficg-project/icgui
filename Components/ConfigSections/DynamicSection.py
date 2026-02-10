"""Components/ConfigSections/DynamicSection.py: Configuration section for timestep-related settings."""

from dataclasses import dataclass
from typing import ClassVar

from imgui_bundle import imgui, icons_fontawesome_6 as fa

from ICGui.Components.HelpIndicator import help_indicator
from ICGui.Components.StyledToggle import styled_toggle
from ICGui.Controls import InputCallback
from ICGui.State.Volatile import GlobalState, TimeState, CameraState
from ICGui.util.Enums import TimeAnimation
from .Section import Section


@dataclass
class DynamicSection(Section):
    name: str = f'{fa.ICON_FA_PERSON_RUNNING} Dynamic Scenes'
    always_open: bool = False
    default_open: bool = False

    _ANIMATION_NAMES: ClassVar[list[str]] = [
        name.title().replace('_', ' ') for name in TimeAnimation.__members__.keys()
    ]

    class DynamicError(Exception):
        pass

    def __post_init__(self):
        super().__post_init__()

        # Check if the dataset contains multiple timesteps
        timesteps = set()
        for _, split in CameraState().dataset_poses.items():
            for sample in split:
                timesteps.add(sample.timestamp)
        if len(timesteps) == 1:
            raise self.DynamicError('The dataset does not contain multiple timesteps.')

        GlobalState().input_manager.register_callback(
            'PAUSE_ANIMATION',
            InputCallback(self.toggle_pause, continuous=False, interrupt_animation=True),
            'Pause Time',
        )

    def _render(self):
        self._render_time()

    def _render_time(self):
        time_state = TimeState()
        old_time = time_state.time

        # Current Time
        min_time, max_time = 0.0, 1.0
        changed, new_time = imgui.slider_float('Time', time_state.timestamp,
                                               min_time, max_time)
        if changed:
            time_state.time = max(min_time, min(max_time, new_time))

        # Pause Toggle
        if imgui.button(f'{fa.ICON_FA_PLAY} Play' if time_state.paused else f'{fa.ICON_FA_PAUSE} Pause'):
            self.toggle_pause()
        imgui.push_item_width(imgui.calc_item_width() - imgui.get_item_rect_size().x - imgui.get_style().item_inner_spacing.x)
        imgui.same_line(spacing=imgui.get_style().item_inner_spacing.x)

        # Speed
        min_speed, max_speed = 1e-6, 64.0
        changed, new_speed = imgui.drag_float('Speed', time_state.speed, 0.005,
                                              0.0, 64.0)
        if changed:
            time_state.speed = max(min_speed, min(max_speed, new_speed))
            time_state.time = old_time  # Reset time since adjusting speed changes the timestep calculation
        imgui.pop_item_width()

        # Animation Mode
        changed, new_anim = imgui.combo('Time Scaling', time_state.animation.value,
                                        self._ANIMATION_NAMES)
        if changed:
            time_state.animation = TimeAnimation(new_anim)

        # Discrete Time Toggle
        changed, time_state.discrete_time = styled_toggle('Use Discrete Timesteps',
                                                          time_state.discrete_time)
        help_indicator('Snap to the nearest discrete timestep in the dataset.')
        if changed:
            time_state.update(0.0)  # Immediately update the timestamp

    @staticmethod
    def toggle_pause():
        """Toggle the visibility of the ground truth."""
        TimeState().paused = not TimeState().paused
