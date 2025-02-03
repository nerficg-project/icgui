# -- coding: utf-8 --

"""GuiComponents/NeRFConfig.py: Dataclass to provide global configuration access."""
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, ClassVar

import numpy as np
import torch

from ICGui.ModelRunners import ModelState
from Cameras.utils import CameraProperties
from Cameras.Base import BaseCamera


# pylint: disable=too-few-public-methods  # Simple container data class
@dataclass(slots=True)
class NeRFConfig:
    """Dataclass to store shared configuration for the NeRF renderer"""
    @dataclass(slots=True)
    class CameraConfig:
        """Dataclass to store shared configuration for the NeRF camera"""
        class TimeModes:
            """Enum for different time modes"""
            SINUSOIDAL = 0
            LINEAR_FORWARD = 1
            LINEAR_REVERSE = 2
            LINEAR = 3

        parent: 'NeRFConfig'
        current: BaseCamera
        base_focal: tuple[float, float] = (0.0, 0.0)
        _time: float = 0.0
        paused: bool = False
        discrete_time: bool = True
        time_mode: int = TimeModes.SINUSOIDAL
        time_speed: float = 0.5
        focal_degrees: bool = False
        constant_fov: bool = False
        gt_idx: int = -1
        dataset_split: str = 'test'
        reset_to_dataset_nearfar: bool = True

        def __post_init__(self):
            self.reset()

        def reset(self):
            """Resets the camera to the first pose in the dataset."""
            camera_properties = deepcopy(self.parent.dataset_poses[self.parent.state.gt_split][0])
            camera_properties.c2w = camera_properties.c2w.to(device='cpu')
            camera_properties.rgb = None
            camera_properties.alpha = None
            camera_properties.depth = None

            self.current.setProperties(camera_properties)
            self.current.background_color = self.current.background_color.to(device='cpu')
            self.base_focal = camera_properties.focal_x, camera_properties.focal_y

            if not self.reset_to_dataset_nearfar:
                # Default to extreme near and far plane, so zooming does not cause the scene to disappear
                self.current.near_plane = min(self.current.near_plane, 0.01)
                self.current.far_plane = max(self.current.far_plane, 100.0)

            self.recalculate()

        def recalculate(self):
            """Updates the camera properties based on the current window size and resolution factor."""
            camera_properties = self.current.properties
            camera_properties.width = int(self.parent.window_size[0] * self.parent.resolution_scaling.factor)
            camera_properties.height = int(self.parent.window_size[1] * self.parent.resolution_scaling.factor)
            camera_properties.focal_x = self.parent.resolution_scaling.factor * self.base_focal[0]
            camera_properties.focal_y = self.parent.resolution_scaling.factor * self.base_focal[1]
            self.current.setProperties(camera_properties)

        @property
        def time(self) -> float:
            """Returns the current camera timestamp in range [-1, 1], with the sign indicating the
            current direction of time."""
            if self.time_mode == self.TimeModes.LINEAR_FORWARD:
                return self.current.properties.timestamp
            if self.time_mode == self.TimeModes.LINEAR_REVERSE:
                return -self.current.properties.timestamp
            if self.time_mode == self.TimeModes.LINEAR:
                return math.copysign(self.current.properties.timestamp,
                                     self._time - 1.0)
            if self.time_mode == self.TimeModes.SINUSOIDAL:
                return math.copysign(self.current.properties.timestamp,  # ~ math.sin(self._time * self.time_speed)
                                     math.cos(self._time * self.time_speed))

            raise ValueError(f'Invalid time mode {self.time_mode}')

        @time.setter
        def time(self, value: float) -> None:
            """Sets the current time to a value in the range [-1, 1]."""
            if not -1.0 <= value <= 1.0:
                raise ValueError(f'Invalid value {value} for time, must be in range [-1, 1]')

            if self.time_mode == self.TimeModes.LINEAR_FORWARD:
                self._time = self.current.properties.timestamp = abs(value)
            elif self.time_mode == self.TimeModes.LINEAR_REVERSE:
                self._time = 1.0 - abs(value)
                self.current.properties.timestamp = 1.0 - self._time
            elif self.time_mode == self.TimeModes.LINEAR:
                self._time = value + 1.0
            elif self.time_mode == self.TimeModes.SINUSOIDAL:
                self._time = math.asin(2 * abs(value) - 1) / self.time_speed if value >= 0 else \
                    (-math.asin(2 * abs(value) - 1) + math.pi) / self.time_speed
                self.current.properties.timestamp = (math.sin(self._time * self.time_speed) + 1) / 2

        def incrementTime(self, dt: float):
            """Increments the current time by dt and updates the camera timestamp."""
            if self.time_mode == self.TimeModes.LINEAR_FORWARD:
                self._time += dt * self.time_speed if not self.paused else 0.0
                self._time %= 1.0
                timestamp = self._time
            elif self.time_mode == self.TimeModes.LINEAR_REVERSE:
                self._time += dt * self.time_speed if not self.paused else 0.0
                self._time %= 1.0
                timestamp = 1.0 - self._time
            elif self.time_mode == self.TimeModes.LINEAR:
                self._time += dt * self.time_speed if not self.paused else 0.0
                self._time %= 2.0
                timestamp = abs(self._time - 1.0)
            elif self.time_mode == self.TimeModes.SINUSOIDAL:
                self._time += dt if not self.paused else 0.0
                timestamp = (math.sin(self._time * self.time_speed) + 1) / 2
            else:
                raise ValueError(f'Invalid time mode {self.time_mode}')

            if self.discrete_time:
                timestamps = [c.timestamp for c in self.parent.dataset_poses[self.parent.state.gt_split]]
                idx = np.argmin(np.abs(np.array(timestamps) - timestamp))
                self.current.properties.timestamp = timestamps[idx]
            else:
                self.current.properties.timestamp = timestamp

        def update(self, new_c2w: torch.Tensor, dt: float):
            """Updates the camera with the given c2w matrix and increments the timestamp by dt."""
            self.current.properties.c2w = new_c2w
            self.incrementTime(dt)

        @property
        def properties(self) -> CameraProperties:
            """Returns the currently selected camera properties."""
            return self.current.properties

        @property
        def render_cam(self) -> BaseCamera:
            """Returns the currently selected camera, or an unscaled camera if GT rendering is enabled."""
            if self.gt_idx < 0:
                return self.current

            cam = deepcopy(self.current)

            camera_properties = cam.properties
            camera_properties.width = self.parent.window_size[0]
            camera_properties.height = self.parent.window_size[1]
            camera_properties.focal_x = self.base_focal[0]
            camera_properties.focal_y = self.base_focal[1]
            cam.setProperties(camera_properties)

            return cam

        @property
        def screenshot_cam(self) -> BaseCamera:
            """Returns the camera used for screenshots, which is the same as the current camera unless a resolution
            override is set."""
            if self.parent.screenshot.resolution_override is None:
                return self.current

            cam = deepcopy(self.current)
            camera_properties = cam.properties
            camera_properties.focal_x /= camera_properties.width
            camera_properties.focal_y /= camera_properties.height
            camera_properties.width = self.parent.screenshot.resolution_override[0]
            camera_properties.height = self.parent.screenshot.resolution_override[1]
            camera_properties.focal_x *= camera_properties.width
            camera_properties.focal_y *= camera_properties.height
            cam.setProperties(camera_properties)

            return cam

        @property
        def cam_idx_current_timestamp(self):
            """Returns the index of the camera that is closest to the current timestamp."""
            timestamps = [c.timestamp for c in self.parent.dataset_poses[self.parent.state.gt_split]]
            return np.argmin(np.abs(np.array(timestamps) - self.current.properties.timestamp))

    def resize(self, width: int, height: int):
        """Resizes the camera to the given width and height."""
        if self.camera.constant_fov:
            self.camera.current.properties.focal_x *= width / self.window_size[0]
            self.camera.current.properties.focal_y *= height / self.window_size[1]

        self.camera.properties.width = int(width * self.resolution_scaling.factor)
        self.camera.properties.height = int(height * self.resolution_scaling.factor)
        self.window_size = width, height

    @dataclass(slots=True)
    class ScreenshotConfig:
        """Dataclass to store shared configuration for screenshots"""
        _take_screenshot: bool = False
        resolution_override: tuple[int, int] | None = None
        include_extras: bool = False

        @property
        def should_take(self) -> bool:
            """Returns whether a screenshot should be taken."""
            take_screenshot, self._take_screenshot = self._take_screenshot, False
            return take_screenshot

        def take(self):
            """Takes a screenshot."""
            self._take_screenshot = True

    @dataclass(slots=True)
    class ResolutionScaleConfig:
        """Dataclass to store shared configuration for (adaptive) resolution scaling"""
        min_scale: ClassVar[float] = 0.05
        max_scale: ClassVar[float] = 2.0

        parent: 'NeRFConfig'
        factor: float
        max_adaptive_scale: float
        adaptive: bool = False
        target_fps: float = 30.0
        dampening: float = 0.4

        def update(self, last_frame_fps: float):
            """Updates the resolution factor based on the target FPS and the framerate model."""
            if not self.adaptive or last_frame_fps == math.nan or last_frame_fps == 0.0:
                return

            # To double the fps, width and height have to be smaller by a factor of sqrt(2)
            ratio = math.sqrt(last_frame_fps / self.target_fps)
            self.factor = max(self.min_scale, min(self.max_adaptive_scale, self.factor * ratio ** self.dampening))

            self.parent.camera.recalculate()

    state: ModelState
    window_size: tuple[int, int]
    dataset_poses: dict[str, list[CameraProperties]]
    dataset_camera: BaseCamera

    # Since these are functions, we're using camelCase to match method naming rather than variable naming
    # pylint: disable=invalid-name
    getTrueResolution: Callable[[], tuple[int, int]]
    resizeWindowCallback: Callable[[int, int], None]

    camera: CameraConfig | None = None
    gui_camera: 'GuiCamera | None' = None
    screenshot: ScreenshotConfig | None = None
    resolution_scaling: ResolutionScaleConfig | None = None
    render_to_window: bool = False
