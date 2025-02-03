# -- coding: utf-8 --

"""GuiComponents/CameraConfiguration.py: Allows configuring the Camera from within the GUI."""

import math
import re
from typing import Any

import numpy as np
import imgui
import torch
from sdl2 import SDL_SCANCODE_F4, SDL_SCANCODE_C, SDL_SCANCODE_G

from ICGui.GuiComponents.Base import GuiComponent
from ICGui.GuiConfig import NeRFConfig
from Cameras.utils import DistortionParameters, RadialTangentialDistortion, IdentityDistortion
from Cameras.Base import BaseCamera
from Cameras.Perspective import PerspectiveCamera
from Cameras.PerspectiveStereo import PerspectiveStereoCamera
from Cameras.Equirectangular import EquirectangularCamera
from Cameras.NDC import NDCCamera
from Cameras.ODS import ODSCamera


class CameraConfiguration(GuiComponent):
    """Informational component for the GUI, showing FPS and hotkeys."""
    _HOTKEYS = [SDL_SCANCODE_F4, SDL_SCANCODE_C]
    _HOTKEY_SHOW_GT = SDL_SCANCODE_G
    _CAMERA_PROPERTIES_ALL = {'near_plane': ('Near Plane', 0.1, float, 0.005, 0.0, 1000.0),
                              'far_plane': ('Far Plane', 5.0, float, 0.005, 0.0, 1000.0)}
    _CAMERAS: list[tuple[str, Any, dict[str, tuple[str, Any, type, ...]]]] = [
        # (Name, Class, {Property: (Label, Default Value, Data Type, Change Speed, Min Value, Max Value)})
        ('Perspective', PerspectiveCamera, _CAMERA_PROPERTIES_ALL),
        ('Perspective Stereo', PerspectiveStereoCamera, _CAMERA_PROPERTIES_ALL |
                                                        {'baseline': ('Baseline', 0.062, float, 0.005, 0.0, 50.0)}),
        ('Equirectangular', EquirectangularCamera, _CAMERA_PROPERTIES_ALL),
        ('Normalized Device Space', NDCCamera, {'cube_scale': ('Cube Scale', 1.0, float, 0.005, 0.0, 10.0)}),
        ('Omni-Directional Stereo Panorama', ODSCamera, _CAMERA_PROPERTIES_ALL |
                                                        {'baseline': ('Baseline', 0.065, float, 0.005, 0.0, 50.0)}),
    ]
    _TIMEMODE_STRINGS: list[str] = [
        # See NeRFConfig.py
        'Sinusoidal',
        'Linear Forward',
        'Linear Backward',
        'Linear',
    ]
    _DISTORTION_TYPES: list[tuple[str, Any, dict[str, tuple[str, Any, type, ...]]]] = [
        ('Identity', IdentityDistortion, {}),
        ('Radial-Tangential Distortion', RadialTangentialDistortion,
             {'k1': ('K1', 0.0, float, 0.005, -1.0, 1.0),
              'k2': ('K2', 0.0, float, 0.005, -1.0, 1.0),
              'k3': ('K3', 0.0, float, 0.005, -1.0, 1.0),
              # 'k4': ('K4', 0.0, float, 0.005, -1.0, 1.0),
              'p1': ('P1', 0.0, float, 0.005, -1.0, 1.0),
              'p2': ('P2', 0.0, float, 0.005, -1.0, 1.0),
              'num_iter': ('Iterations', 10, int, 1, 1, 100)}),
    ]
    _TENSOR_REGEX = re.compile(r'\s*(?:tensor)?\(?\s*\[?\s*\[\s*(-?\d+.?\d*(?:e[-+]\d+)?)\s*,\s*'
                               r'(-?\d+.?\d*(?:e[-+]\d+)?)\s*,\s*(-?\d+.?\d*(?:e[-+]\d+)?)\s*,\s*'
                               r'(-?\d+.?\d*(?:e[-+]\d+)?)\s*]\s*,\s*\[\s*(-?\d+.?\d*(?:e[-+]\d+)?)\s*,\s*'
                               r'(-?\d+.?\d*(?:e[-+]\d+)?)\s*,\s*(-?\d+.?\d*(?:e[-+]\d+)?)\s*,\s*'
                               r'(-?\d+.?\d*(?:e[-+]\d+)?)\s*]\s*,\s*\[\s*(-?\d+.?\d*(?:e[-+]\d+)?)\s*,\s*'
                               r'(-?\d+.?\d*(?:e[-+]\d+)?)\s*,\s*(-?\d+.?\d*(?:e[-+]\d+)?)\s*,\s*'
                               r'(-?\d+.?\d*(?:e[-+]\d+)?)\s*]\s*,\s*\[\s*(-?\d+.?\d*(?:e[-+]\d+)?)\s*,\s*'
                               r'(-?\d+.?\d*(?:e[-+]\d+)?)\s*,\s*(-?\d+.?\d*(?:e[-+]\d+)?)\s*,\s*'
                               r'(-?\d+.?\d*(?:e[-+]\d+)?)\s*]\s*,?\s*\]?\s*,\s*(?:device=\'\w+\')?\)?\s*',
                               re.IGNORECASE)

    def __init__(self, config: NeRFConfig):
        self._config: NeRFConfig
        super().__init__('Camera Config', CameraConfiguration._HOTKEYS, config=config,
                         closable=True, default_open=False)
        self._registerHotkey(
            'Show / Hide Ground Truth',
            lambda: self._updateCamera(show_gt=not self._config.camera.gt_idx != -1),
            (CameraConfiguration._HOTKEY_SHOW_GT,),
            False
        )

        self._camera_cache: dict[str, BaseCamera] = {}
        self._selected_camera: int = 0
        self._distortion_cache: dict[str, DistortionParameters] = {}
        self._last_dataset_camera_w2c: torch.Tensor | None = None
        self._last_dataset_camera_idx: int = -1
        self._selected_distortion: int = 0
        self._show_advanced = False
        self._matrices_as_text = False

        # Try to determine which camera type is used
        for idx, (name, camera_type, _) in enumerate(CameraConfiguration._CAMERAS):
            if isinstance(self._config.camera.current, camera_type):
                self._selected_camera = idx
                self._camera_cache[name] = self._config.camera.current
                break
        else:
            raise NotImplementedError(f'Camera type {type(self._config.camera.current)} not supported by GUI yet')

        # Try to determine which distortion type is used
        distortion = self._config.camera.properties.distortion_parameters
        if distortion is not None:
            for idx, (name, distortion_type, _) in enumerate(CameraConfiguration._DISTORTION_TYPES):
                if isinstance(distortion, distortion_type):
                    self._selected_distortion = idx
                    self._distortion_cache[name] = distortion
                    break
            else:
                raise NotImplementedError(f'Distortion type {type(distortion)} not supported by GUI yet')

        self._config.camera.reset()

    @staticmethod
    def calculateSimilarity(matrix1, matrix2):
        """Calculates the similarity between two projection matrices using the Frobenius norm.

        Args:
          matrix1: A numpy array representing the first projection matrix.
          matrix2: A numpy array representing the second projection matrix.

        Returns:
          A float representing the similarity between the two projection matrices.
        """

        difference = matrix1 - matrix2
        frobenius_norm = np.linalg.norm(difference, 'fro')
        similarity_score = 1 - frobenius_norm / np.linalg.norm(matrix1, 'fro')
        return similarity_score

    @staticmethod
    def focalToFov(focal: float) -> float:
        """Converts a focal length to a field of view."""
        return 180.0 - math.atan(2.0 * focal) / math.pi * 360.0

    @staticmethod
    def fovToFocal(fov: float) -> float:
        """Converts a field of view to a focal length."""
        return math.tan((180.0 - fov) / 360.0 * math.pi) / 2.0

    @staticmethod
    def _renderPropertyInput(obj: Any, key: str, label: str, data_type: type, args: tuple[Any]):
        """Renders a single property input field for the given object."""
        if data_type is float:
            min_val, max_val = args[1], args[2]
            changed, new_val = imgui.drag_float(label, getattr(obj, key), *args)

            if changed:
                setattr(obj, key, max(min_val, min(max_val, new_val)))
        elif data_type is int:
            min_val, max_val = args[1], args[2]
            changed, new_val = imgui.drag_int(label, getattr(obj, key), *args)
            if changed:
                setattr(obj, key, max(min_val, min(max_val, new_val)))
        else:
            raise NotImplementedError(f'Unsupported data type {data_type} for camera configuration')

    # pylint: disable=arguments-differ
    def _render(self):
        """Renders the GUI component."""
        self._renderTime()

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        self._renderControls()

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        self._renderCameraSelector()
        self._renderCameraProperties()

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        self._renderDatasetCameraSelector()

    def _renderTime(self):
        """Renders the time controls."""
        changed, self._config.camera.discrete_time = imgui.checkbox('Use Discrete Timesteps',
                                                                    self._config.camera.discrete_time)
        if imgui.is_item_hovered():
            imgui.set_tooltip('Automatically round the time to the nearest discrete timestep from the dataset')
        if changed:
            self._config.camera.incrementTime(0.0)  # Immediately update the timestamp

        _, self._config.camera.paused = imgui.checkbox('Pause', self._config.camera.paused)
        old_time = self._config.camera.time
        imgui.same_line()

        min_time, max_time = 0.0, 1.0
        changed, new_time = imgui.slider_float('Time', self._config.camera.properties.timestamp,
                                               min_time, max_time)
        if changed:
            self._config.camera.time = max(min_time, min(max_time, new_time))

        min_speed, max_speed = 1e-6, 64.0
        changed, new_speed = imgui.drag_float('Speed', self._config.camera.time_speed, 0.005,
                                              0.0, 64.0)
        if changed:
            self._config.camera.time_speed = max(min_speed, min(max_speed, new_speed))
            self._config.camera.time = old_time

        _, self._config.camera.time_mode = imgui.combo('Time Scaling', self._config.camera.time_mode,
                                                       CameraConfiguration._TIMEMODE_STRINGS)

    def _renderCameraSelector(self):
        """Renders the camera selector."""
        _, selected_camera = imgui.combo('Camera Type', self._selected_camera,
                                         [name for name, _, _ in CameraConfiguration._CAMERAS])
        if selected_camera != self._selected_camera:
            self._switchCamera(selected_camera)
            self._selected_camera = selected_camera

    def _renderCameraProperties(self):
        """Renders the camera property options."""
        # Background Color
        changed, color = imgui.color_edit3('Background Color',
                                           *self._config.camera.current.background_color.cpu().numpy())
        if changed:
            self._config.camera.current.background_color = torch.tensor(color, dtype=torch.float32, device='cpu')

        # General Camera Properties
        for key, (label, _, data_type, *args) in \
                CameraConfiguration._CAMERAS[self._selected_camera][2].items():
            self._renderPropertyInput(self._config.camera.current, key, label, data_type, args)

        self._renderPrincipalPoint()  # Principal Point
        self._renderFocal()  # Focal Length

        imgui.spacing()
        self._renderDistortionProperties()  # Distortion
        imgui.spacing()

        # Reset
        if imgui.button('Reset Camera'):
            self._config.camera.reset()

        self._renderAvanced()  # Advanced (Direct Matrix Input)

    def _renderPrincipalPoint(self):
        min_principal, max_principal = -(2**10), 2**10
        changed, (principal_x, principal_y) = imgui.drag_float2(
            'Principal Point',
            self._config.camera.properties.principal_offset_x,
            self._config.camera.properties.principal_offset_y,
            0.5, min_principal, max_principal
        )
        if changed:
            self._config.camera.properties.principal_offset_x = (
                max(min_principal, min(max_principal, principal_x)))
            self._config.camera.properties.principal_offset_y = (
                max(min_principal, min(max_principal, principal_y)))

    def _renderFocal(self):
        converter = CameraConfiguration.focalToFov if self._config.camera.focal_degrees else lambda x: x
        limit_converter = (lambda x: 180.0 - CameraConfiguration.focalToFov(x)) if self._config.camera.focal_degrees \
            else lambda x: x
        reverse_converter = CameraConfiguration.fovToFocal if self._config.camera.focal_degrees else lambda x: x

        min_focal, max_focal, change_speed = limit_converter(1e-6), limit_converter(100.0), limit_converter(0.002)
        changed, (focal_x, focal_y) = imgui.drag_float2(
            'Focal Length' if not self._config.camera.focal_degrees else 'Field of View',
            converter(self._config.camera.properties.focal_x / self._config.camera.properties.width),
            converter(self._config.camera.properties.focal_y / self._config.camera.properties.height),
            change_speed, min_focal, max_focal,
            format='%.3f' if not self._config.camera.focal_degrees else '%.1f°'
        )
        if changed:
            focal_x = reverse_converter(max(min_focal, min(max_focal, focal_x)))
            focal_y = reverse_converter(max(min_focal, min(max_focal, focal_y)))
            self._config.camera.base_focal = (
                focal_x * self._config.camera.properties.width / self._config.resolution_scaling.factor,
                focal_y * self._config.camera.properties.height / self._config.resolution_scaling.factor
            )
            self._config.camera.properties.focal_x = focal_x * self._config.camera.properties.width
            self._config.camera.properties.focal_y = focal_y * self._config.camera.properties.height

        _, self._config.camera.focal_degrees = imgui.checkbox('Show as FOV [°]',
                                                              self._config.camera.focal_degrees)
        _, self._config.camera.constant_fov = imgui.checkbox('Constant FOV when resizing',
                                                             self._config.camera.constant_fov)

        if imgui.is_item_hovered():
            imgui.set_tooltip('Maintain current fov when resizing the window.\n'
                              'This effectively makes resizing the window equivalent to zooming in or out.')

    # pylint: disable=too-many-branches  # Complex GUI rendering
    def _renderAvanced(self):
        imgui.separator()

        if not self._show_advanced:
            if imgui.button('Show Advanced'):
                self._show_advanced = not self._show_advanced
            return
        _, self._matrices_as_text = imgui.checkbox('Show Matrices as Text', self._matrices_as_text)

        imgui.spacing()

        for matrix, matrix_text in [('w2c', 'World to Camera Matrix'), ('c2w', 'Camera to World Matrix')]:
            imgui.text(matrix_text)
            original_mat: torch.Tensor = getattr(self._config.gui_camera, matrix)
            if self._matrices_as_text:
                changed, mat_text = imgui.input_text_multiline(f'##{matrix}', str(original_mat.cpu()))
                if not changed:
                    continue

                match = CameraConfiguration._TENSOR_REGEX.match(mat_text)
                if not match:
                    continue

                mat = torch.tensor([float(match.group(i)) for i in range(1, 17)]).reshape(4, 4)
                if matrix == 'c2w':
                    self._config.gui_camera.c2w = mat.to(device=original_mat.device)
                else:
                    self._config.gui_camera.w2c = mat.to(device=original_mat.device)
            else:
                for i in range(4):
                    mat = original_mat.cpu()
                    changed, values = imgui.input_float4(f'##{matrix}_{i}', mat[i, 0], mat[i, 1],
                                                         mat[i, 2], mat[i, 3], format='%.3f')
                    if not changed:
                        continue

                    mat[i] = torch.tensor(values)
                    if matrix == 'c2w':
                        self._config.gui_camera.c2w = mat.to(device=original_mat.device)
                    else:
                        self._config.gui_camera.w2c = mat.to(device=original_mat.device)

            if matrix != 'c2w':
                imgui.separator()

        if imgui.button('Hide Advanced'):
            self._show_advanced = False

    def _renderControls(self):
        changed, selected_controls = imgui.combo('Controls', self._config.gui_camera.current_camera_controls,
                                                 self._config.gui_camera.camera_controls)
        if changed:
            self._config.gui_camera.chooseControls(selected_controls)

        _, self._config.gui_camera.invert_x = imgui.checkbox('Invert Horizontal Rotation',
                                                             self._config.gui_camera.invert_x)
        _, self._config.gui_camera.invert_y = imgui.checkbox('Invert Vertical Rotation',
                                                             self._config.gui_camera.invert_y)

        _, self._config.gui_camera.mouse_sensitivity = imgui.drag_float(
            'Mouse Sensitivity',
            self._config.gui_camera.mouse_sensitivity,
            0.0001,
            0.0001, 1.0
        )

        _, self._config.gui_camera.movement_speed = imgui.drag_float(
            'Movement Speed',
            self._config.gui_camera.movement_speed,
            0.01,
            0.01, 10.0
        )

    def _renderDistortionProperties(self):
        changed, use_distortion = imgui.checkbox(
            'Apply camera distortion',
            self._config.camera.properties.distortion_parameters is not None
        )
        if changed and not use_distortion:
            self._config.camera.properties.distortion_parameters = None
        if changed and use_distortion:
            self._enableDistortion()

        if use_distortion:
            _, selected_distortion = imgui.combo(
                'Distortion Type',
                self._selected_distortion,
                [name for name, _, _ in CameraConfiguration._DISTORTION_TYPES]
            )
            if (selected_distortion != self._selected_distortion
                    and 0 <= selected_distortion < len(CameraConfiguration._DISTORTION_TYPES)):
                self._selected_distortion = selected_distortion
                self._enableDistortion()

            # noinspection PyUnresolvedReferences
            for key, (label, _, data_type, *args) in \
                    CameraConfiguration._DISTORTION_TYPES[self._selected_distortion][2].items():
                self._renderPropertyInput(
                    self._config.camera.properties.distortion_parameters,
                    key, label, data_type,
                    args
                )

    def _renderDatasetCameraSelector(self):
        """Renders the selector allowing to jump to the closest / any camera in the dataset and show GT images."""
        with imgui.begin_combo('Dataset Split', self._config.camera.dataset_split) as split_selector_opened:
            if split_selector_opened:
                for split in self._config.state.splits:
                    if imgui.selectable(split, split == self._config.camera.dataset_split)[0]:
                        self._config.camera.dataset_split = split
                        self._config.gui_camera.w2c = self._config.dataset_poses[split][0].w2c.to(device='cpu')
                        self._last_dataset_camera_idx = -1
                        self._last_dataset_camera_w2c = None

        with (imgui.begin_combo('##DatasetCameraSelector',
                                'Choose camera pose from dataset' if self._last_dataset_camera_idx == -1
                                else f'Camera {self._last_dataset_camera_idx}')
              as camera_selector_opened):
            if camera_selector_opened:
                for i, cam in enumerate(self._config.dataset_poses[self._config.camera.dataset_split]):
                    if imgui.selectable(f'Camera {i}', False)[0]:
                        self._config.gui_camera.easeTo(cam.w2c.to(device='cpu'))
                        self._last_dataset_camera_idx = i
                        self._last_dataset_camera_w2c = cam.w2c.to(device='cpu')

        jump_to_closest = imgui.button('Jump to closest dataset camera')
        _, show_gt = imgui.checkbox('Show ground truth image', self._config.camera.gt_idx != -1)

        self._updateCamera(show_gt, jump_to_closest)

    def _checkCameraChanged(self):
        if self._last_dataset_camera_idx == -1:
            return False
        if self._last_dataset_camera_w2c is None:
            return False
        if torch.allclose(self._last_dataset_camera_w2c, self._config.gui_camera.w2c):
            return False
        if self._config.gui_camera.easing_target is None:
            return True

        return not torch.allclose(self._last_dataset_camera_w2c, self._config.gui_camera.easing_target)

    def _updateCamera(self, show_gt: bool = False, jump_to_closest: bool = False):
        if not show_gt:
            self._config.camera.gt_idx = -1

        if self._checkCameraChanged():
            self._last_dataset_camera_idx = -1
            self._last_dataset_camera_w2c = None

        if not jump_to_closest and not show_gt:
            return

        # Update camera to closest dataset camera
        max_sim, max_sim_idx = 0, 0
        for i, cam in enumerate(self._config.dataset_poses[self._config.camera.dataset_split]):
            sim = self.calculateSimilarity(self._config.gui_camera.c2w, cam.c2w.to(device='cpu'))
            if sim > max_sim:
                max_sim = sim
                max_sim_idx = i
        self._config.gui_camera.easeTo(self._config.dataset_poses[self._config.camera.dataset_split][max_sim_idx]
                                       .w2c.to(device='cpu'))
        self._last_dataset_camera_w2c = (self._config.dataset_poses[self._config.camera.dataset_split][max_sim_idx]
                                         .w2c.to(device='cpu'))
        self._last_dataset_camera_idx = max_sim_idx

        if show_gt:
            self._config.camera.gt_idx = max_sim_idx
            self._config.camera.paused = True
            self._config.camera.time = self._config.dataset_poses[
                self._config.camera.dataset_split][max_sim_idx].timestamp

    def _switchCamera(self, idx):
        """Switches the camera type to the given index in _CAMERAS."""
        if CameraConfiguration._CAMERAS[idx][0] in self._camera_cache:
            self._config.camera.current = self._camera_cache[CameraConfiguration._CAMERAS[idx][0]]
            return

        # Initialize camera with default arguments, then copy over the properties from the previous camera
        # noinspection PyUnresolvedReferences
        default_args = {k: v[1] for k, v in CameraConfiguration._CAMERAS[idx][2].items()}
        # noinspection PyCallingNonCallable
        new_camera = CameraConfiguration._CAMERAS[idx][1](**default_args)
        new_camera.properties = self._config.camera.properties
        new_camera.near_plane = self._config.camera.current.near_plane
        new_camera.far_plane = self._config.camera.current.far_plane
        new_camera.background_color = self._config.camera.current.background_color
        self._camera_cache[CameraConfiguration._CAMERAS[idx][0]] = new_camera
        self._config.camera.current = new_camera

    def _enableDistortion(self):
        distortion_name, distortion_cls, distortion_params = \
                CameraConfiguration._DISTORTION_TYPES[self._selected_distortion]
        if distortion_name not in self._distortion_cache:
            self._distortion_cache[distortion_name] = distortion_cls(**{k: v[1] for k, v in
                                                                        distortion_params.items()})
        self._config.camera.properties.distortion_parameters = self._distortion_cache[distortion_name]
