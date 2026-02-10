"""Components/ConfigSections/ControlsSection.py: GUI Controls configuration section for the config window."""

from dataclasses import dataclass

from imgui_bundle import imgui, icons_fontawesome_6 as fa

from ICGui.Components.StyledToggle import styled_toggle
from ICGui.State.Volatile import GlobalState
from .Section import Section


@dataclass
class ControlsSection(Section):
    name: str = f'{fa.ICON_FA_GAMEPAD} Viewer Controls'
    always_open: bool = False
    default_open: bool = False

    def __post_init__(self):
        super().__post_init__()

    def _render(self):
        input_manager = GlobalState().input_manager

        # Axes inversion
        _, input_manager.control_scheme.invert_rotation_x = styled_toggle(
            'Invert Horizontal Rotation',
            input_manager.control_scheme.invert_rotation_x
        )
        _, input_manager.control_scheme.invert_rotation_y = styled_toggle(
            'Invert Vertical Rotation',
            input_manager.control_scheme.invert_rotation_y
        )
        if input_manager.control_scheme.supports_panning:
            _, input_manager.control_scheme.invert_panning_x = styled_toggle(
                'Invert Horizontal Panning',
                input_manager.control_scheme.invert_panning_x
            )
            _, input_manager.control_scheme.invert_panning_y = styled_toggle(
                'Invert Vertical Panning',
                input_manager.control_scheme.invert_panning_y
            )

        # Control scheme selection (Flying / Orbital)
        changed, selected_controls = imgui.combo(
            'Control Scheme',
            input_manager.current_camera_controls_idx,
            input_manager.camera_controls,
        )
        if changed:
            input_manager.choose_camera(selected_controls)

        # Input sensitivity
        _, input_manager.control_scheme.rotation_speed = imgui.drag_float(
            'Rotation Speed',
            input_manager.control_scheme.rotation_speed,
            v_speed=0.0001,
            v_min=0.0001,
            v_max=1.0,
        )
        _, input_manager.control_scheme.travel_speed = imgui.drag_float(
            'Movement Speed',
            input_manager.control_scheme.travel_speed,
            v_speed=0.01,
            v_min=0.01,
            v_max=10.0,
        )
        _, input_manager.control_scheme.zoom_speed = imgui.drag_float(
            'Zoom Speed',
            input_manager.control_scheme.zoom_speed,
            v_speed=0.005,
            v_min=0.001,
            v_max=10.0,
        )
