"""Components/ConfigSections/CameraSection.py: Camera projection configuration section for the config window."""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import torch
from imgui_bundle import imgui, icons_fontawesome_6 as fa

from Cameras.Base import BaseCamera
from Cameras.Perspective import PerspectiveCamera
from Cameras.utils import BaseDistortion, SharedCameraSettings
from ICGui.Components.Colors import apply_ui_color
from ICGui.Components.EpsilonRangeInput import epsilon_range_input
from ICGui.Components.FocalInput import focal_input
from ICGui.Components.HelpIndicator import help_indicator
from ICGui.Components.StyledToggle import styled_toggle
from ICGui.State.Volatile import CameraState
from ICGui.util.Cameras import CAMERA_TYPES, DISTORTION_TYPES, ConfigurableProperty, get_camera_idx, get_distortion_idx, \
    VIRTUAL_CAMERA_SETTINGS
from .Section import Section


@dataclass
class CameraSection(Section):
    name: str = f'{fa.ICON_FA_VIDEO} Camera'
    always_open: bool = False
    default_open: bool = False

    _camera_cache: dict[str, BaseCamera] = field(default_factory=dict, init=False)
    _distortion_cache: dict[str, BaseDistortion] = field(default_factory=dict, init=False)
    _selected_camera: int = 0
    _selected_distortion: int = 0
    _lock_aspect: bool = True

    def __post_init__(self):
        super().__post_init__()

        # Try to determine which camera and distortion types are used, then add them to the cache
        self._selected_camera = get_camera_idx(CameraState().current_camera)
        self._selected_distortion = get_distortion_idx(
            getattr(CameraState().current_camera, 'distortion', None)
        )
        self._camera_cache[CAMERA_TYPES[self._selected_camera].name] = CameraState().current_camera
        if self._selected_distortion > -1:
            self._distortion_cache[DISTORTION_TYPES[self._selected_distortion].name] = (
                getattr(CameraState().current_camera, 'distortion', None)
            )

    def _render(self):
        self._render_camera_selection()
        self._render_camera_properties()

        if hasattr(CameraState().current_camera, 'distortion'):
            self._render_distortion_selection()
            self._render_distortion_properties()
        imgui.spacing()

        _, CameraState().constant_fov = styled_toggle(
            'Constant FOV on Window Resize',
            CameraState().constant_fov
        )
        help_indicator('Maintain the same field of view when resizing the image. '
                       'Intuitively, this means increasing the window size "zooms" '
                       'the model, instead of expanding the visible area.')
        imgui.spacing()

        # Reset
        with apply_ui_color('ERROR'):
            if imgui.button(f'{fa.ICON_FA_DELETE_LEFT} Reset Camera'):
                CameraState().reset()
                self._selected_camera = get_camera_idx(CameraState().current_camera)
                # self._selected_distortion = get_distortion_idx(CameraState().properties.distortion_parameters)
        help_indicator('Reset all camera settings to the dataset default. Does not affect the camera pose.')

    def _render_camera_selection(self):
        """Renders the camera selector."""
        changed, selected_camera = imgui.combo('Camera Model', self._selected_camera,
                                         [cam.name for cam in CAMERA_TYPES])
        if changed and selected_camera != self._selected_camera:
            self._switch_camera(selected_camera)
        help_indicator('Changing the camera model may not be supported by all models.')

    def _render_camera_properties(self):
        """Renders the camera property options."""
        current_camera = CameraState().current_camera

        # Background Color
        changed, color = imgui.color_edit3(
            'Background Color',
            _tensor_to_list_cached(current_camera.background_color),
        )
        if changed:
            current_camera.background_color = torch.tensor(color, dtype=torch.float32, device='cpu')
        help_indicator('Changing the background color may not be supported by all models.')

        # Near and Far Plane
        changed, new_near_far = epsilon_range_input(
            'Near / Far Plane', [current_camera.near_plane, current_camera.far_plane],
            speed=0.005, min_v=1e-2, max_v=10000.0,
            epsilon=1e-2, fmt='%.2f'
        )
        if changed:
            current_camera.near_plane, current_camera.far_plane = new_near_far

        # General Camera Properties
        for prop in CAMERA_TYPES[self._selected_camera].configurable_properties:
            if prop.key in ('near_plane', 'far_plane'):
                continue  # Custom handler, see above
            self._property_input(current_camera, prop)

        self._render_projection_center()  # Projection Center
        self._render_focal_length()  # Focal Length, TODO: Hide for e.g. omnidirectional cameras

    @staticmethod
    def _render_projection_center():
        camera_state = CameraState()
        if not camera_state.has_center:
            return

        min_center, max_center = 0, max(camera_state.width, camera_state.height)
        changed, (center_x, center_y) = imgui.drag_float2(
            'Projection Center',
            [camera_state.center_x, camera_state.center_y],
            0.5, min_center, max_center,
            format='%.1f px',
        )
        if changed:
            camera_state.center_x = max(min_center, min(camera_state.width, center_x))
            camera_state.center_y = max(min_center, min(camera_state.height, center_y))

    def _render_focal_length(self):
        camera_state = CameraState()
        if not camera_state.has_focal:
            return

        # Alignment to make sure the lock button covers the height of both focal length inputs
        cursor_pos = imgui.get_cursor_pos()
        item_height = imgui.get_frame_height()
        line_height = imgui.get_frame_height_with_spacing()
        self._lock_aspect ^= imgui.button(
            fa.ICON_FA_LINK if self._lock_aspect else fa.ICON_FA_LINK_SLASH,
            (0, line_height + item_height)
        )
        button_size = imgui.get_item_rect_size()

        if imgui.is_item_hovered():
            imgui.begin_tooltip()
            imgui.text('Unlock aspect ratio' if self._lock_aspect else 'Lock aspect ratio')
            imgui.end_tooltip()

        cursor_pos.x += button_size.x + imgui.get_style().item_inner_spacing.x
        imgui.push_item_width(imgui.calc_item_width() - button_size.x - imgui.get_style().item_spacing.x)
        imgui.set_cursor_pos(cursor_pos)

        changed, new_fx = focal_input('Focal Length (x)', camera_state.focal_x, camera_state.width)
        if changed:
            if self._lock_aspect:
                # TODO: Fix the fact that max / min can be broken by modifying the lower / higher focal length
                percent_change = new_fx / camera_state.focal_x
                # Apply percent change to focal_y
                camera_state.focal_y *= percent_change
            camera_state.focal_x = new_fx

        # Manually move the cursor position down to the next line
        cursor_pos.y += line_height
        imgui.set_cursor_pos(cursor_pos)

        changed, new_fy = focal_input('Focal Length (y)', camera_state.focal_y, camera_state.height)
        if changed:
            if self._lock_aspect:
                # TODO: Fix the fact that max / min can be broken by modifying the lower / higher focal length
                percent_change = new_fy / camera_state.focal_y
                # Apply percent change to focal_x
                camera_state.focal_x *= percent_change
            camera_state.focal_y = new_fy

        imgui.pop_item_width()  # Remove changes made for the link button

    def _render_distortion_selection(self):
        """Renders the camera selector."""
        changed, selected_distortion = imgui.combo('Distortion Type', self._selected_distortion + 1,
                                                   ['None', *[dist.name for dist in DISTORTION_TYPES]])
        selected_distortion -= 1  # Subtract 1, because None is not a real distortion type

        if changed and selected_distortion != self._selected_distortion:
            self._switch_distortion(selected_distortion)
        help_indicator('Applies distortion parameters to the current camera. '
                       'May not be supported by all cameras / models.')

    def _render_distortion_properties(self):
        if self._selected_distortion < 0:
            return
        distortion_params = getattr(CameraState().current_camera, 'distortion', None)
        if distortion_params is None:
            return
        for prop in DISTORTION_TYPES[self._selected_distortion].configurable_properties:
            self._property_input(distortion_params, prop)

    def _switch_camera(self, camera_idx: int):
        """Switches the camera to the one specified by the index."""
        if camera_idx < 0 or camera_idx >= len(CAMERA_TYPES):
            return

        camera_type = CAMERA_TYPES[camera_idx]
        if camera_type.name not in self._camera_cache:
            current_camera = CameraState().current_camera

            # Initialize camera with previous camera settings where possible
            shared_camera_args = {
                prop.key: getattr(current_camera.shared_settings, prop.key, prop.default)
                for prop in VIRTUAL_CAMERA_SETTINGS
            }
            custom_camera_args = {
                prop.key: getattr(current_camera, prop.key, prop.default)
                for prop in CAMERA_TYPES[camera_idx].configurable_properties
            }
            new_camera = CAMERA_TYPES[camera_idx].camera_class(
                shared_settings=SharedCameraSettings(**shared_camera_args),
                width=current_camera.width, height=current_camera.height,
                **custom_camera_args,
            )
            if hasattr(new_camera, 'distortion') and hasattr(current_camera, 'distortion'):
                setattr(new_camera, 'distortion', getattr(current_camera, 'distortion'))

            # Store camera in cache
            self._camera_cache[camera_type.name] = new_camera

        CameraState().current_camera = self._camera_cache[camera_type.name]
        self._selected_camera = camera_idx
        if not hasattr(CameraState().current_camera, 'distortion'):
            self._selected_distortion = -1

    def _switch_distortion(self, distortion_idx: int):
        """Switches the distortion parameters to the one specified by the index."""
        if distortion_idx >= len(DISTORTION_TYPES):
            return

        if hasattr(CameraState().current_camera, 'distortion'):
            if distortion_idx < 0:
                # Disable distortion
                setattr(CameraState().current_camera, 'distortion', None)
                self._selected_distortion = -1
                return

            # Set selected distortion
            distortion_type = DISTORTION_TYPES[distortion_idx]
            if distortion_type.name not in self._distortion_cache:
                # Initialize distortion parameters with default arguments
                default_args = {prop.key: prop.default for prop in DISTORTION_TYPES[distortion_idx].configurable_properties}
                new_distortion_params = distortion_type.distortion_class(**default_args)

                # Store distortion parameters in cache
                self._distortion_cache[distortion_type.name] = new_distortion_params

            setattr(CameraState().current_camera, 'distortion', self._distortion_cache[distortion_type.name])
            self._selected_distortion = distortion_idx

    @staticmethod
    def _property_input(obj, prop: ConfigurableProperty):
        """Renders a single property input field for the given object."""
        optional = {}
        if prop.format is not None:
            optional['format'] = prop.format

        if prop.property_type is float:
            changed, new_val = imgui.drag_float(
                prop.name, getattr(obj, prop.key),
                prop.change_speed, prop.min_value, prop.max_value,
                **optional,
            )
            if changed:
                setattr(obj, prop.key, max(prop.min_value, min(prop.max_value, new_val)))
        elif prop.property_type is int:
            changed, new_val = imgui.drag_int(
                prop.name, getattr(obj, prop.key),
                prop.change_speed, prop.min_value, prop.max_value,
                **optional,
            )
            if changed:
                setattr(obj, prop.key, max(prop.min_value, min(prop.max_value, new_val)))
        else:
            raise NotImplementedError(f'Unsupported data type {prop.property_type} for camera configuration.'
                                      f'Please implement a custom input method for {prop.name} of type {prop.property_type}'
                                      f'in file {Path(__file__).absolute()}.')


@lru_cache
def _tensor_to_list_cached(tensor: torch.Tensor) -> list[float]:
    """Returns the background color of the current camera."""
    return tensor.numpy().tolist()
