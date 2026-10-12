"""State/Volatile.py: Dataclasses to provide global configuration access to volatile (non-saved) state."""

import inspect
import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import torch
from OpenGL.GL import GL_NEAREST, GL_LINEAR

from Cameras.Base import BaseCamera
from Cameras.Perspective import PerspectiveCamera
from Datasets.utils import View
from Visual.ColorMap import ColorMap
from ICGui.util.Cameras import argmax_temporal_similarity
from ICGui.util.Singleton import Singleton
from .Shared import SharedState
from .LaunchConfig import LaunchConfig


@dataclass(slots=True)
class GlobalState(metaclass=Singleton):
    """Dataclass to store shared configuration for the NeRF renderer"""
    shared: SharedState | None = None
    launch_config: LaunchConfig | None = None
    input_manager: 'InputHandler | None' = None
    backend_window: 'SDL3Window | None' = None  # Added to the state after initialization
    viewer: 'ModelViewer | None' = None  # Added to the state after initialization

    model_framerate: float = np.nan
    model_framerate_std: float = np.nan
    model_frametime: str = 'NaN ms'
    model_frametime_std: str = 'NaN ms'
    render_to_window: bool = True
    window_size: tuple[int, int] = (-1, -1)  # Set in __post_init__

    skip_animations: bool = False

    def __post_init__(self):
        if self.shared is None:
            raise ValueError('NeRFConfig must be initialized with a SharedState instance.')
        if self.launch_config is None:
            raise ValueError('NeRFConfig must be initialized with a LaunchConfig instance.')
        self.window_size = tuple(self.launch_config.initial_resolution)

    def resize(self, width: int, height: int):
        """Resizes the window to the given width and height."""
        self.window_size = width, height
        CameraState().resize(width, height)


@dataclass(slots=True)
class CameraState(metaclass=Singleton):
    """Dataclass to store shared configuration for the NeRF camera"""
    dataset_camera: BaseCamera | None = None
    dataset_cameras: list[BaseCamera] | None = None
    dataset_view: View | None = None
    current_view: View | None = field(init=False, default=None)  # Added automatically in __post_init__
    dataset_poses: dict[str, list[View]] | None = None
    dataset_split: str | None = field(default_factory=lambda: SharedState().splits[0] if len(SharedState().splits) > 0 else None)
    bbox_size: float = 1.0
    window_size: tuple[int, ...] = field(init=False, default_factory=lambda: (-1, -1))  # Added after initialization

    # GUI toggles
    focal_degrees: bool = field(init=False, default=False)
    constant_fov: bool = field(init=False, default=False)
    prefer_temporal_similarity: bool = field(init=False, default=True)

    # Whether a ground truth camera is rendered
    render_gt: bool = field(init=False, default=False)
    # Whether to snap to the closest ground truth pose when not rendering it
    snap_to_gt: bool = field(init=False, default=False)

    # Maintain an unscaled version of the camera to avoid accumulating numerical errors
    unscaled_camera: BaseCamera | None = field(init=False, default=None)  # Added automatically in __post_init__

    def __post_init__(self):
        if self.dataset_camera is None:
            raise ValueError('CameraConfig must be initialized with a dataset default camera.')
        if self.dataset_view is None:
            raise ValueError('CameraConfig must be initialized with a dataset default view.')
        if self.dataset_poses is None:
            raise ValueError('CameraConfig must be initialized with dataset poses.')

        if self.window_size == (-1, -1):
            self.window_size = tuple(GlobalState().window_size)
        self.reset()

    def reset(self):
        """Resets the camera to the default dataset camera, preserving the view."""
        self.current_view = deepcopy(self.dataset_view)
        self.unscaled_camera = deepcopy(self.dataset_camera)
        self.rescale()

    def reset_view(self):
        """Resets the current view to the dataset default view, preserving the current camera."""
        current_camera = self.current_camera
        self.current_view.camera = deepcopy(self.dataset_camera)
        self.current_camera = current_camera

    def rescale(self):
        """Updates the camera based on the current window size and resolution factor."""
        resolution_factor = ResolutionScaleState().factor

        # Rescale width / height based on the current resolution factor and window size
        width = self.current_camera.width = int(self.window_size[0] * resolution_factor)
        height = self.current_camera.height = int(self.window_size[1] * resolution_factor)

        # Resolution factors, rounded to the nearest pixel
        resolution_factor_x = width / GlobalState().window_size[0]
        resolution_factor_y = height / GlobalState().window_size[1]

        # Scaling factors relative to the unscaled camera
        factor_x = width / self.unscaled_camera.width
        factor_y = height / self.unscaled_camera.height

        # Rescale focal / center
        if self.has_focal:
            unscaled_cam: PerspectiveCamera = self.unscaled_camera  # type: ignore
            current_cam: PerspectiveCamera = self.current_camera  # type: ignore
            if self.constant_fov:
                current_cam.focal_x = current_cam.width / unscaled_cam.width * unscaled_cam.focal_x
                current_cam.focal_y = current_cam.height / unscaled_cam.height * unscaled_cam.focal_y
            else:
                current_cam.focal_x = resolution_factor_x * unscaled_cam.focal_x
                current_cam.focal_y = resolution_factor_y * unscaled_cam.focal_y
        if self.has_center:
            unscaled_cam: PerspectiveCamera = self.unscaled_camera  # type: ignore
            current_cam: PerspectiveCamera = self.current_camera  # type: ignore
            current_cam.center_x = (unscaled_cam.center_x - unscaled_cam.width / 2) * factor_x + width / 2
            current_cam.center_y = (unscaled_cam.center_y - unscaled_cam.height / 2) * factor_y + height / 2

    def resize(self, width: int, height: int):
        """Resizes the camera to the given width and height."""
        self.window_size = width, height
        self.rescale()

    def set_camera(self, camera: BaseCamera):
        """Sets the current camera, which must match the current resolution, and derives the unscaled camera such
        that rescaling reproduces it."""
        unscaled_size = self.unscaled_camera.width, self.unscaled_camera.height
        self.current_camera = camera
        self.unscaled_camera = deepcopy(camera)
        self.unscaled_camera.width, self.unscaled_camera.height = unscaled_size
        if self.has_focal:
            # The setters convert to unscaled values and rescale, capture all values first
            focal_x, focal_y, center_x, center_y = camera.focal_x, camera.focal_y, camera.center_x, camera.center_y
            self.focal_x, self.focal_y, self.center_x, self.center_y = focal_x, focal_y, center_x, center_y

    def update(self, c2w: np.ndarray):
        """Updates the current view with the given c2w matrix, and updates the timestamp."""
        if (gt_idx := self.gt_idx) == -1:
            if self.snap_to_gt:
                self.current_view.c2w = self.dataset_poses[self.dataset_split][self._nearest_gt_idx].c2w_numpy
            else:
                self.current_view.c2w = c2w
        else:
            self.current_view.c2w = self.dataset_poses[self.dataset_split][gt_idx].c2w_numpy
        self.current_view.timestamp = TimeState().timestamp

        # Reset global frame index if moved
        if GlobalState().input_manager.control_scheme.has_moved:
            self.current_view.global_frame_idx = -1
            self.current_view.frame_idx = -1

    @property
    def current_camera(self) -> BaseCamera:
        """Returns the current camera instance."""
        return self.current_view.camera

    @current_camera.setter
    def current_camera(self, camera: BaseCamera):
        """Sets the current camera instance."""
        self.current_view.camera = camera

    @property
    def width(self) -> int:
        """Returns the current camera width."""
        return self.current_camera.width

    @property
    def height(self) -> int:
        """Returns the current camera height."""
        return self.current_camera.height

    @property
    def has_focal(self):
        """Returns whether the current camera has a focal length."""
        return isinstance(self.current_camera, PerspectiveCamera)

    @property
    def focal_x(self):
        """Returns the current camera focal length in x direction, or 0 if the camera has no focal length."""
        if self.has_focal:
            cam: PerspectiveCamera = self.current_camera  # type: ignore
            return cam.focal_x
        return 0.0

    @focal_x.setter
    def focal_x(self, value: float):
        """Sets the current camera focal length in x direction, if the camera has a focal length."""
        if self.has_focal:
            cam: PerspectiveCamera = self.unscaled_camera  # type: ignore
            # Write unscaled value
            if self.constant_fov:
                cam.focal_x = value / self.current_camera.width * self.unscaled_camera.width
            else:
                # Resolution factors, rounded to the nearest pixel
                resolution_factor = int(self.window_size[0] * ResolutionScaleState().factor) / GlobalState().window_size[0]
                cam.focal_x = value / resolution_factor
            self.rescale()

    @property
    def focal_y(self):
        """Returns the current camera focal length in y direction, or 0 if the camera has no focal length."""
        if self.has_focal:
            cam: PerspectiveCamera = self.current_camera  # type: ignore
            return cam.focal_y
        return 0.0

    @focal_y.setter
    def focal_y(self, value: float):
        """Sets the current camera focal length in y direction, if the camera has a focal length."""
        if self.has_focal:
            cam: PerspectiveCamera = self.unscaled_camera  # type: ignore
            # Write unscaled value
            if self.constant_fov:
                cam.focal_y = value / self.current_camera.height * self.unscaled_camera.height
            else:
                # Resolution factors, rounded to the nearest pixel
                resolution_factor = int(self.window_size[1] * ResolutionScaleState().factor) / GlobalState().window_size[1]
                cam.focal_y = value / resolution_factor
            self.rescale()

    @property
    def has_center(self):
        """Returns whether the current camera has a (modifiable) projection center."""
        return isinstance(self.current_camera, PerspectiveCamera)

    @property
    def center_x(self):
        """Returns the current projection center in x direction, or width/2 if the camera has no projection center."""
        if self.has_center:
            cam: PerspectiveCamera = self.current_camera  # type: ignore
            return cam.center_x
        return self.current_camera.width / 2

    @center_x.setter
    def center_x(self, value: float):
        """Sets the current projection center in x direction, if the camera has a projection center."""
        if self.has_center:
            cam: PerspectiveCamera = self.unscaled_camera  # type: ignore

            # Write unscaled value, inverting the scaling in rescale()
            width_orig = self.unscaled_camera.width
            width_scaled = self.current_camera.width
            cam.center_x = (value - width_scaled / 2) / width_scaled * width_orig + width_orig / 2
            self.rescale()

    @property
    def center_y(self):
        """Returns the current projection center in y direction, or height/2 if the camera has no projection center."""
        if self.has_center:
            cam: PerspectiveCamera = self.current_camera  # type: ignore
            return cam.center_y
        return self.current_camera.height / 2

    @center_y.setter
    def center_y(self, value: float):
        """Sets the current projection center in y direction, if the camera has a projection center."""
        if self.has_center:
            cam: PerspectiveCamera = self.unscaled_camera  # type: ignore

            # Write unscaled value, inverting the scaling in rescale()
            height_orig = self.unscaled_camera.height
            height_scaled = self.current_camera.height
            cam.center_y = (value - height_scaled / 2) / height_scaled * height_orig + height_orig / 2
            self.rescale()

    @property
    def _nearest_gt_idx(self) -> int:
        return argmax_temporal_similarity(
            GlobalState().input_manager.control_scheme.c2w,
            TimeState().timestamp,
            [(pose.c2w_numpy, pose.timestamp) for pose in self.dataset_poses[self.dataset_split]],
            prefer_time=self.prefer_temporal_similarity,
        )

    @property
    def gt_idx(self) -> int:
        """Index of the GT pose to render, -1 means no GT, i.e. shows the current camera render"""
        if self.render_gt:
            return self._nearest_gt_idx
        return -1

    @property
    def closest_timestep_indices(self):
        """Returns a list of indices of views with the minimum distance to the current timestamp."""
        timestamps = np.array([pose.timestamp for pose in self.dataset_poses[self.dataset_split]])
        min_distance = np.min(np.abs(timestamps - TimeState().timestamp))
        closest_indices = np.nonzero(np.abs(timestamps - TimeState().timestamp) == min_distance)[0]
        return closest_indices

    @property
    def poses(self) -> list[View]:
        """Returns the list of camera poses for the current dataset split."""
        if self.dataset_split in self.dataset_poses:
            return self.dataset_poses[self.dataset_split]
        else:
            return []

    @property
    def texture_size(self) -> tuple[int, int]:
        """Returns the texture size of the current camera."""
        if ResolutionScaleState().adaptive:
            # Always render to full-res texture to avoid flickering
            return self.window_size[0], self.window_size[1]
        return self.current_view.camera.width, self.current_view.camera.height


@dataclass(slots=True)
class ViewerOutputState(metaclass=Singleton):
    """Dataclass to store shared configuration for the viewer output"""
    selected_idx: int = 0
    output_choices: list[str] = field(default_factory=lambda: ['rgb'])

    _IGNORED_KEYS: ClassVar[list[str]] = ['total_samples', 'fps', 'fps_std', 'type']

    def update_output_choices(self, frame: dict[str, torch.Tensor]):
        """Updates the choices for the model output, e.g. ['rgb', 'depth']."""
        current_output = self.selected_output
        self.output_choices = list(map(lambda x: x[0], filter(self.is_key_valid, frame.items())))
        try:
            self.selected_output = current_output
        except ValueError:
            # If the current output is not in the new choices, reset to the first choice
            self.selected_idx = 0

    @property
    def selected_output(self) -> str:
        """Returns the currently selected model output."""
        if not self.output_choices:
            return 'rgb'
        return self.output_choices[self.selected_idx]

    @selected_output.setter
    def selected_output(self, value: str):
        """Sets the currently selected model output."""
        if value not in self.output_choices:
            raise ValueError(f'Output {value} is not in the current output choices: {self.output_choices}')
        self.selected_idx = self.output_choices.index(value)

    @classmethod
    def is_key_valid(cls, item: tuple[str, Any]) -> bool:
        """Returns whether the given key is a valid choice for the viewer."""
        key, value = item
        if key in cls._IGNORED_KEYS:
            return False
        if not isinstance(value, torch.Tensor):
            return False  # Only tensors are supported
        if not 2 <= len(value.shape) <= 3:
            return False  # Only 2D or 3D tensors are supported
        if len(value.shape) == 3 and value.shape[2] > 4:
            return False  # More than 4 channels is not supported
        return True


@dataclass(slots=True)
class ColorMapState(metaclass=Singleton):
    """Dataclass to store shared configuration for the color map"""
    COLORMAP_CHOICES: ClassVar[list[str]] = ['Grayscale'] + [
        member[0]
        for member in inspect.getmembers(ColorMap, predicate=inspect.isfunction)
        if member[0].isupper()
    ]
    selected_idx: int = 0

    show_settings: bool = False
    interpolate: bool = False
    logscale: bool = False
    invert: bool = False
    custom_min_max: bool = False
    reset_bounds_on_next_update: bool = False

    nth_percentile: float = 100.0
    min_max: tuple[float, float] = (0.0, 1.0)
    min_max_change_speed: float = 0.0025

    def update(self, current_output: torch.Tensor):
        # When an output is rendered that has 1 channel, show the color map settings
        if (len(current_output.shape) > 2 and current_output.shape[-1] == 1) or len(current_output.shape) == 2:
            self.show_settings = True
        else:
            self.show_settings = False

        if self.custom_min_max and not self.reset_bounds_on_next_update:
            return

        # TODO: Implement temporal smoothing
        if self.nth_percentile < 100.0 - 1e-4:
            self.min_max = (current_output.min().item(),
                            current_output.quantile(self.nth_percentile / 100.0).item())
        else:
            self.min_max = (current_output.min().item(),
                            current_output.max().item())
        self.min_max_change_speed = (self.min_max[1] - self.min_max[0]) * 0.0025
        self.reset_bounds_on_next_update = False

    @property
    def selected_colormap(self) -> str:
        """Returns the currently selected depth color map."""
        return self.COLORMAP_CHOICES[self.selected_idx]

    @property
    def dict(self) -> dict[str, Any]:
        """Returns the currently selected color map configuration as a kwargs dict
           to pass to ICGui.util.ColorChannels.to_rgba."""
        min_max = None
        if self.custom_min_max or self.nth_percentile < 100.0 - 1e-4:
            min_max = self.min_max

        return {
            'color_map': self.selected_colormap,
            'interpolate': self.interpolate,
            'logscale': self.logscale,
            'invert': self.invert,
            'min_max': min_max,
        }


@dataclass(slots=True)
class ResolutionScaleState(metaclass=Singleton):
    """Dataclass to store shared configuration for (adaptive) resolution scaling"""
    min_scale: ClassVar[float] = 0.05
    max_scale: ClassVar[float] = 2.0

    _factor: float | None = None
    adaptive: bool = False
    max_adaptive_scale: float = 1.0
    target_fps: float = 30.0
    dampening: float = 0.4

    MAGNIFICATION_FILTERS: ClassVar[list[int]] = [GL_NEAREST, GL_LINEAR]
    MINIFICATION_FILTERS: ClassVar[list[int]] = [GL_NEAREST, GL_LINEAR]
    MAGNIFICATION_FILTER_NAMES: ClassVar[list[str]] = ['Nearest', 'Linear']
    MINIFICATION_FILTER_NAMES: ClassVar[list[str]] = ['Nearest', 'Linear']
    _magnification_filter: int = 1
    _minification_filter: int = 1

    def __post_init__(self):
        if self._factor is None:
            self._factor = GlobalState().launch_config.resolution_factor

    def adapt(self, last_frame_fps: float):
        """Updates the resolution factor based on the target FPS and the framerate model."""
        if not self.adaptive or math.isnan(last_frame_fps) or last_frame_fps == 0.0:
            return

        # This assumes that to increase the FPS by a factor of x, there have to be 1/x as many pixels rendered.
        # This then reduces to a factor of sqrt(x) for width and height.
        ratio = math.sqrt(last_frame_fps / self.target_fps)
        self.factor = max(self.min_scale, min(self.max_adaptive_scale, self.factor * ratio ** self.dampening))

    @property
    def factor(self) -> float:
        """Returns the current resolution factor."""
        return self._factor

    @factor.setter
    def factor(self, value: float):
        """Sets the resolution factor, clamping it to the min and max scale."""
        if not self.min_scale <= value <= self.max_scale:
            raise ValueError(f'Resolution factor must be in range [{self.min_scale}, {self.max_scale}], got {value}')
        self._factor = value
        CameraState().rescale()

    @property
    def magnification_filter(self) -> int:
        """Returns the current magnification filter."""
        return self._magnification_filter

    @magnification_filter.setter
    def magnification_filter(self, idx: int):
        """Sets the magnification filter."""
        if idx >= len(self.MAGNIFICATION_FILTERS) or idx < 0:
            raise ValueError(f'Invalid magnification filter index {idx}')
        self._magnification_filter = idx
        GlobalState().viewer.set_output_texture_filtering(magnify=self.MAGNIFICATION_FILTERS[idx])

    @property
    def minification_filter(self) -> int:
        """Returns the current minification filter."""
        return self._minification_filter

    @minification_filter.setter
    def minification_filter(self, idx: int):
        """Sets the minification filter."""
        if idx >= len(self.MINIFICATION_FILTERS) or idx < 0:
            raise ValueError(f'Invalid minification filter index {idx}')
        self._minification_filter = idx
        GlobalState().viewer.set_output_texture_filtering(minify=self.MINIFICATION_FILTERS[idx])


@dataclass(slots=True)
class TimeState(metaclass=Singleton):
    """Manages time state for dynamic scenes with direct dataset timestamp values."""
    timestamp: float = 0.0  # Derived from internal time, including discretization
    paused: bool = False
    discrete_time: bool = True
    disabled: bool = False  # Whether time should be disabled entirely, i.e. always set to 0.0 and not updated

    speed: float = 0.2
    is_reverse: bool = False  # Whether to move backward instead of forward
    enable_bounce: bool = False  # Whether to reverse at boundaries (instead of wrapping)

    # Dynamic timestamp range determined from dataset
    min_timestamp: float = field(init=False, default=0.0)
    max_timestamp: float = field(init=False, default=1.0)
    _timestamp_list: np.ndarray = field(init=False)  # List of all timestamps in the dataset, used for discretization
    _time: float = field(init=False, default=0.0)  # Internal time position used for animation

    def __post_init__(self):
        self._timestamp_list = np.array([c.timestamp for c in CameraState().poses])
        if len(self._timestamp_list) > 0:
            self.min_timestamp = np.min(self._timestamp_list)
            self.max_timestamp = np.max(self._timestamp_list)
            self._time = self.min_timestamp

    @property
    def time(self) -> float:
        """Returns the current camera timestamp."""
        if self.disabled:
            return self.min_timestamp
        return self.timestamp

    @time.setter
    def time(self, value: float) -> None:
        """Sets the current timestamp to a value in the range [min_timestamp, max_timestamp]."""
        if not self.min_timestamp <= value <= self.max_timestamp:
            raise ValueError(f'Invalid value {value} for time, must be in range [{self.min_timestamp}, {self.max_timestamp}]')
        self._time = value

        # Set timestamp based on discretization setting
        if self.discrete_time and len(self._timestamp_list) > 0:
            # Discretize time to ensure only dataset timestamps are used
            idx = np.argmin(np.abs(self._timestamp_list - self._time))
            self.timestamp = self._timestamp_list[idx]
        else:
            self.timestamp = self._time

    def update(self, dt: float):
        """Increments time by dt and updates the camera timestamp."""
        if self.paused or self.disabled:
            return
        if self.max_timestamp - self.min_timestamp == 0:
            return

        # Update position based on direction and bounce settings
        direction = -1 if self.is_reverse else 1
        self._time += dt * self.speed * direction

        if self._time < self.min_timestamp:
            if self.enable_bounce:
                self._time = self.min_timestamp + (self.min_timestamp - self._time)
                self.is_reverse = not self.is_reverse
            else:
                self._time = self.max_timestamp - (self.min_timestamp - self._time) % (self.max_timestamp - self.min_timestamp)
        elif self._time > self.max_timestamp:
            if self.enable_bounce:
                self._time = self.max_timestamp - (self._time - self.max_timestamp)
                self.is_reverse = not self.is_reverse
            else:
                self._time = self.min_timestamp + (self._time - self.max_timestamp) % (self.max_timestamp - self.min_timestamp)

        # Clamp to valid range in case we overshoot due to large dt or speed
        self.time = max(self.min_timestamp, min(self.max_timestamp, self._time))


@dataclass(slots=True)
class OverlayState(metaclass=Singleton):
    """Dataclass to store shared configuration for OpenGL extras overlays"""
    enabled: dict[str, bool] = field(default_factory=dict)
    available_overlays: list['BaseOverlay'] = field(default_factory=list)
    additional_overlay_data: dict[str, dict] = field(default_factory=dict)
