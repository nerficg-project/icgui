"""Components/ConfigSections/CameraSection.py: Camera projection configuration section for the config window."""

from bisect import bisect
from dataclasses import dataclass, field
from pathlib import Path
from copy import deepcopy

import numpy as np
from imgui_bundle import imgui, icons_fontawesome_6 as fa

from Cameras.Base import BaseCamera
from Cameras.utils import BaseDistortion
from ICGui.Components.EpsilonRangeInput import epsilon_range_input
from ICGui.Components.FocalInput import focal_input
from ICGui.Components.HelpIndicator import help_indicator
from ICGui.Components.StyledToggle import styled_toggle
from ICGui.State.Volatile import CameraState
from ICGui.util.Cameras import CAMERA_TYPES, DISTORTION_TYPES, ConfigurableProperty, get_camera_idx, get_distortion_idx
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

    # Presets received from the runner / added by the user: name -> camera
    _preset_map: dict[str, BaseCamera] = field(default_factory=dict, init=False)
    _preset_keys_dataset: list[str] = field(default_factory=list, init=False)
    _preset_keys_custom: list[str] = field(default_factory=list, init=False)
    _selected_preset: int = 0
    _new_preset_name: str = 'New Preset Name'

    def __post_init__(self):
        super().__post_init__()

        self.sync_with_current_camera()

        # Populate presets from the dataset cameras provided to CameraState on GUI launch.
        self._preset_map['Default'] = deepcopy(CameraState().current_camera)
        dataset_cameras = CameraState().dataset_cameras
        if dataset_cameras is not None:
            for i, cam in enumerate(dataset_cameras):
                name = f'{i}-{cam.__class__.__name__}'
                self._preset_keys_dataset.append(name)
                self._preset_map[name] = deepcopy(cam)

    def sync_with_current_camera(self):
        """Updates the selection and caches from the current camera. Must be called after replacing the current camera
        from outside this section, e.g. by trajectory previews."""
        camera = CameraState().current_camera
        self._selected_camera = get_camera_idx(camera)
        self._selected_distortion = get_distortion_idx(getattr(camera, 'distortion', None))
        self._camera_cache[CAMERA_TYPES[self._selected_camera].name] = camera
        if self._selected_distortion > -1:
            self._distortion_cache[DISTORTION_TYPES[self._selected_distortion].name] = camera.distortion

    def _render(self):
        # Render preset controls (load / add)
        self._render_preset_selection()

        # Custom configuration
        imgui.separator_text('Custom Camera Configuration')
        self._render_camera_selection()
        changed = self.render_camera_settings(CameraState().current_camera)
        if hasattr(CameraState().current_camera, 'distortion'):
            self._render_distortion_selection()
            changed |= self.render_distortion_settings(CameraState().current_camera)
        if changed:
            CameraState().set_camera(CameraState().current_camera)  # Derives the unscaled camera from the modified values
            self._selected_preset = -1  # Deselect preset
        imgui.spacing()

        _, CameraState().constant_fov = styled_toggle(
            'Constant FOV on Window Resize',
            CameraState().constant_fov
        )
        help_indicator('Maintain the same field of view when resizing the image. '
                       'Intuitively, this means increasing the window size "zooms" '
                       'the model, instead of expanding the visible area.')
        imgui.spacing()

    def _render_camera_selection(self):
        """Renders the camera selector."""
        changed, selected_camera = imgui.combo('Camera Model', self._selected_camera,
                                         [cam.name for cam in CAMERA_TYPES])
        if changed and selected_camera != self._selected_camera:
            self._switch_camera(selected_camera)
            self._selected_preset = -1  # Deselect preset
        help_indicator('Changing the camera model may not be supported by all models.')

    def _render_preset_selection(self):
        """Renders the camera preset dropdown and add/load controls."""
        if self._selected_preset == -1:
            label = 'Select Preset...'
        elif self._selected_preset == 0:
            label = f'{fa.ICON_FA_STAR} Default'
        elif self._selected_preset <= len(self._preset_keys_dataset):
            label = self._preset_keys_dataset[self._selected_preset - 1]  # -1 to account for default preset at index 0
        elif self._selected_preset <= len(self._preset_keys_dataset) + len(self._preset_keys_custom):
            label = self._preset_keys_custom[self._selected_preset - 1 - len(self._preset_keys_dataset)]  # -1 for default, -dataset presets
        else:
            label = 'Select Preset...'

        if imgui.begin_combo('Load Preset##CameraPresetSelector', label):
            if imgui.selectable(f'{fa.ICON_FA_STAR} Default', self._selected_preset == 0)[0]:
                self._selected_preset = 0
                self._load_camera_from_preset(self._preset_map['Default'])

            if len(self._preset_keys_dataset) > 0:
                imgui.separator_text(f'{fa.ICON_FA_BOOKMARK} Dataset Cameras')
                for i, preset_name in enumerate(self._preset_keys_dataset):
                    index = i + 1  # +1 to account for the default preset at index 0
                    if imgui.selectable(preset_name, index == self._selected_preset)[0]:
                        self._selected_preset = index
                        self._load_camera_from_preset(self._preset_map[preset_name])
            if len(self._preset_keys_custom) > 0:
                imgui.separator_text(f'{fa.ICON_FA_WRENCH} Custom Presets')
                for i, preset_name in enumerate(self._preset_keys_custom):
                    index = i + 1 + len(self._preset_keys_dataset)  # +1 for default, +dataset presets
                    if imgui.selectable(preset_name, index == self._selected_preset)[0]:
                        self._selected_preset = index
                        self._load_camera_from_preset(self._preset_map[preset_name])
            imgui.end_combo()
        help_indicator('Load a preset camera. Presets are provided by the dataset or added by the user.')

        # Add current camera as preset
        _, self._new_preset_name = imgui.input_text('##New Preset Name', self._new_preset_name)
        imgui.same_line()

        disabled = False
        if self._new_preset_name.strip() == '':
            imgui.begin_disabled()
            disabled = True
        if imgui.button(f'{fa.ICON_FA_PLUS} Add Preset'):
            name = self._new_preset_name.strip()
            # Append a trailing number if the name is already in use, e.g. 'cam1' -> 'cam1 (2)'
            preset_name = name
            suffix = 2
            while preset_name in self._preset_map:
                preset_name = f'{name} ({suffix})'
                suffix += 1

            insertion_index = bisect(self._preset_keys_custom, preset_name)
            self._selected_preset = insertion_index + 1 + len(self._preset_keys_dataset)  # +1 for default, +dataset presets
            self._preset_keys_custom.insert(insertion_index, preset_name)
            self._preset_map[preset_name] = CameraState().current_camera
        if disabled:
            imgui.end_disabled()
        help_indicator('Create a new preset from the current camera configuration.')

    def render_camera_settings(self, camera: BaseCamera) -> bool:
        """Renders the settings of the given camera, excluding model and distortion, and modifies it in place.
        Returns whether any setting was modified."""
        # Background Color
        changed, color = imgui.color_edit3(
            'Background Color',
            camera.shared_settings.background_color_numpy.tolist(),
        )
        if changed:
            camera.background_color = np.array(color, dtype=np.float32)
        any_changed = changed
        help_indicator('Changing the background color may not be supported by all models.')

        # Near and Far Plane
        changed, new_near_far = epsilon_range_input(
            'Near / Far Plane', [camera.near_plane, camera.far_plane],
            speed=0.005, min_v=1e-2, max_v=10000.0,
            epsilon=1e-2, fmt='%.2f'
        )
        if changed:
            camera.near_plane, camera.far_plane = new_near_far
        any_changed |= changed

        # General Camera Properties
        for prop in CAMERA_TYPES[get_camera_idx(camera)].configurable_properties:
            if prop.key in ('near_plane', 'far_plane'):
                continue  # Custom handler, see above
            any_changed |= self._property_input(camera, prop)

        if hasattr(camera, 'center_x') and hasattr(camera, 'center_y'):
            any_changed |= self._render_projection_center(camera)
        if hasattr(camera, 'focal_x') and hasattr(camera, 'focal_y'):
            any_changed |= self._render_focal_length(camera)  # TODO: Hide for e.g. omnidirectional cameras
        return any_changed

    @staticmethod
    def _render_projection_center(camera: BaseCamera) -> bool:
        min_center, max_center = 0, max(camera.width, camera.height)
        changed, (center_x, center_y) = imgui.drag_float2(
            'Projection Center',
            [camera.center_x, camera.center_y],
            0.5, min_center, max_center,
            format='%.1f px',
        )
        if changed:
            camera.center_x = max(min_center, min(camera.width, center_x))
            camera.center_y = max(min_center, min(camera.height, center_y))
        return changed

    def _render_focal_length(self, camera: BaseCamera) -> bool:
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

        changed_x, new_fx = focal_input('Focal Length (x)', camera.focal_x, camera.width)
        if changed_x:
            if self._lock_aspect:
                # TODO: Fix the fact that max / min can be broken by modifying the lower / higher focal length
                percent_change = new_fx / camera.focal_x
                # Apply percent change to focal_y
                camera.focal_y *= percent_change
            camera.focal_x = new_fx

        # Manually move the cursor position down to the next line
        cursor_pos.y += line_height
        imgui.set_cursor_pos(cursor_pos)

        changed_y, new_fy = focal_input('Focal Length (y)', camera.focal_y, camera.height)
        if changed_y:
            if self._lock_aspect:
                # TODO: Fix the fact that max / min can be broken by modifying the lower / higher focal length
                percent_change = new_fy / camera.focal_y
                # Apply percent change to focal_x
                camera.focal_x *= percent_change
            camera.focal_y = new_fy

        imgui.pop_item_width()  # Remove changes made for the link button
        return changed_x or changed_y

    def _render_distortion_selection(self):
        """Renders the distortion selector."""
        changed, selected_distortion = imgui.combo('Distortion Type', self._selected_distortion + 1,
                                                   ['None', *[dist.name for dist in DISTORTION_TYPES]])
        selected_distortion -= 1  # Subtract 1, because None is not a real distortion type

        if changed and selected_distortion != self._selected_distortion:
            self._switch_distortion(selected_distortion)
            self._selected_preset = -1  # Deselect preset
        help_indicator('Applies distortion parameters to the current camera. '
                       'May not be supported by all cameras / models.')

    def render_distortion_settings(self, camera: BaseCamera) -> bool:
        """Renders the distortion parameters of the given camera, if any, and modifies them in place.
        Returns whether any parameter was modified."""
        distortion_params = getattr(camera, 'distortion', None)
        if distortion_params is None:
            return False
        changed = False
        for prop in DISTORTION_TYPES[get_distortion_idx(distortion_params)].configurable_properties:
            changed |= self._property_input(distortion_params, prop)
        return changed

    def _switch_camera(self, camera_idx: int):
        """Switches the camera to the one specified by the index."""
        if camera_idx < 0 or camera_idx >= len(CAMERA_TYPES):
            return

        camera_type = CAMERA_TYPES[camera_idx]
        if camera_type.name not in self._camera_cache:
            self._camera_cache[camera_type.name] = self.convert_camera(CameraState().current_camera, camera_idx)

        CameraState().current_camera = self._camera_cache[camera_type.name]
        self._selected_camera = camera_idx
        if not hasattr(CameraState().current_camera, 'distortion'):
            self._selected_distortion = -1

    @staticmethod
    def convert_camera(camera: BaseCamera, camera_idx: int) -> BaseCamera:
        """Creates a camera of the model specified by the index, initialized with the given camera's settings where possible."""
        custom_camera_args = {
            prop.key: getattr(camera, prop.key, prop.default)
            for prop in CAMERA_TYPES[camera_idx].configurable_properties
        }
        new_camera = CAMERA_TYPES[camera_idx].camera_class(
            shared_settings=deepcopy(camera.shared_settings),
            width=camera.width, height=camera.height,
            **custom_camera_args,
        )
        if hasattr(new_camera, 'distortion') and hasattr(camera, 'distortion'):
            setattr(new_camera, 'distortion', getattr(camera, 'distortion'))
        return new_camera

    @staticmethod
    def create_distortion(distortion_idx: int) -> BaseDistortion:
        """Creates distortion parameters of the type specified by the index, initialized with default arguments."""
        default_args = {prop.key: prop.default for prop in DISTORTION_TYPES[distortion_idx].configurable_properties}
        return DISTORTION_TYPES[distortion_idx].distortion_class(**default_args)

    def _load_camera_from_preset(self, camera: BaseCamera):
        camera = deepcopy(camera)  # Avoid modifying the preset camera when changing properties
        # camera.width = CameraState().width  # Preserve current resolution
        # camera.height = CameraState().height
        CameraState().unscaled_camera = camera
        CameraState().rescale()

        # Populate custom camera config fields
        self._selected_camera = get_camera_idx(camera)
        self._selected_distortion = get_distortion_idx(getattr(camera, 'distortion', None))

        # Override / store in cache
        self._camera_cache[CAMERA_TYPES[self._selected_camera].name] = camera
        if self._selected_distortion > -1:
            self._distortion_cache[DISTORTION_TYPES[self._selected_distortion].name] = getattr(camera, 'distortion', None)

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
                self._distortion_cache[distortion_type.name] = self.create_distortion(distortion_idx)

            setattr(CameraState().current_camera, 'distortion', self._distortion_cache[distortion_type.name])
            self._selected_distortion = distortion_idx

    @staticmethod
    def _property_input(obj, prop: ConfigurableProperty) -> bool:
        """Renders a single property input field for the given object. Returns true value was modified."""
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
        return changed
