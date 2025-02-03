# -- coding: utf-8 --

"""ModelState.py: Handles exchanging shared state between the GUI and render processes"""

import queue
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.multiprocessing as mp

import Framework
from Cameras.Base import BaseCamera


_FRAME_QUEUE_SIZE = 3
_CAMERA_QUEUE_SIZE = 2
_SCREENSHOT_QUEUE_SIZE = 8
_CONFIG_ADVERTISEMENT_QUEUE_SIZE = 8
_CONFIG_REQUEST_QUEUE_SIZE = 32


@dataclass
class ModelState:
    """Handles exchanging shared state between the GUI and render processes by using queues to send tensors."""
    _frame_queue: 'mp.Queue[dict[str, torch.Tensor | float | BaseCamera]]' = \
        field(default_factory=lambda: mp.Queue(_FRAME_QUEUE_SIZE))
    _config_advertisements: 'mp.Queue[list[tuple[str, Any]]]' \
        = field(default_factory=lambda: mp.Queue(_CONFIG_ADVERTISEMENT_QUEUE_SIZE))
    _config_request: 'mp.Queue[dict[str, Any]]' = field(default_factory=lambda: mp.Queue(_CONFIG_REQUEST_QUEUE_SIZE))
    _camera: 'mp.Queue[BaseCamera]' = field(default_factory=lambda: mp.Queue(_CAMERA_QUEUE_SIZE))
    _screenshot_queue: 'mp.Queue[tuple[BaseCamera, str, str]]' = \
        field(default_factory=lambda: mp.Queue(_SCREENSHOT_QUEUE_SIZE))

    splits: list[str] = field(default_factory=lambda: ['train', 'val', 'test'])
    _gt_index: 'mp.Value[int]' = field(default_factory=lambda: mp.get_context('spawn').Value('i', -1))
    _gt_split: 'mp.Value[str]' = field(default_factory=lambda: mp.get_context('spawn').Value('i', 2))
    _window_size: 'mp.Array[int]' = field(default_factory=lambda: mp.Array('i', 2))

    start_as_training: bool = field(default=False, init=True, repr=False)
    _training: 'mp.Value[bool]' = field(default_factory=lambda: mp.get_context('spawn').Value('b', False))
    _training_iteration: 'mp.Value[int]' = field(default_factory=lambda: mp.get_context('spawn').Value('i', -1))
    _terminate_training: 'mp.Value[bool]' = field(default_factory=lambda: mp.get_context('spawn').Value('b', False))

    def __post_init__(self):
        if self.start_as_training:
            self.is_training = True

    @property
    def frame(self) -> torch.Tensor | None:
        """Receives the new frame from the GUI process or None if no new frame has been sent."""
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None
        except FileNotFoundError:
            Framework.Logger.logInfo('Frame Queue does not exist anymore.')
            return None

    @frame.setter
    def frame(self, value: dict[str, torch.Tensor | float]):
        try:
            self._frame_queue.put(value, timeout=1.0)
        except queue.Full:
            Framework.Logger.logDebug('Frame queue has been full for more than 1 second, skipping frame')
        except FileNotFoundError:
            Framework.Logger.logWarning('Frame queue does not exist anymore, discarding frame')

    @property
    def camera(self) -> BaseCamera | None:
        """Receives the new camera from the GUI process or None if no new properties have been sent."""
        try:
            camera = self._camera.get_nowait()
            if camera is not None:
                camera.background_color = camera.background_color.to(Framework.config.GLOBAL.DEFAULT_DEVICE)
                camera.properties = camera.properties.toDefaultDevice()
            return camera
        except queue.Empty:
            return None
        except FileNotFoundError:
            Framework.Logger.logInfo('Camera Queue does not exist anymore.')
            return None

    @camera.setter
    def camera(self, value: BaseCamera):
        """Sends the new camera to the render process, removing any older cameras if they are still in
        the queue."""
        try:
            # Always leave at least 1 element in the queue. This avoids race conditions when the renderer tries to
            #   grab a new camera inbetween an old one being removed from the GUI and a new one being added. These can
            #   otherwise be seen as stutters (due to the model timestamp not getting updated). One downside is that
            #   this may delay the camera update by 1 GUI frame (i.e. at 60 FPS an additional 0.0167 seconds).
            if self._camera.qsize() > 1:
                self._camera.get_nowait()
        except queue.Empty:
            pass
        except FileNotFoundError:
            Framework.Logger.logInfo('Camera Queue does not exist anymore, discarding frame.')
            return

        self._camera.put(value)

    @property
    def screenshot_camera(self) -> tuple[BaseCamera, str, str] | None:
        """Receives the new camera from the GUI process or None if no new properties have been sent."""
        try:
            camera = self._screenshot_queue.get_nowait()
            if camera is not None:
                camera[0].background_color = camera[0].background_color.to(Framework.config.GLOBAL.DEFAULT_DEVICE)
                camera[0].properties = camera[0].properties.toDefaultDevice()
            return camera
        except queue.Empty:
            return None
        except FileNotFoundError:
            Framework.Logger.logInfo('Screenshot Queue does not exist anymore.')
            return None

    def screenshot(self, camera: BaseCamera, color_mode: str, color_map: str):
        """Sends the new camera to the render process, removing any older cameras if they are still in
        the queue."""
        try:
            self._screenshot_queue.put_nowait((camera, color_mode, color_map))
        except queue.Full:
            Framework.Logger.logWarning('Screenshot Queue is full, discarding screenshot request.')
        except FileNotFoundError:
            Framework.Logger.logWarning('Screenshot Queue does not exist anymore.')

    @property
    def gt_index(self) -> int:
        """Receives the index of the GT image to render or -1 if the renderer should be used."""
        return self._gt_index.value

    @property
    def gt_split(self) -> str:
        """Receives the split of the GT image to render."""
        return self.splits[self._gt_split.value]

    def useRenderer(self):
        """Tells the render process to use the renderer instead of the GT image."""
        self._gt_index.value = -1

    def renderGt(self, idx: int, split: str):
        """Tells the render process to render the GT image at the given index."""
        self._gt_index.value = idx
        self._gt_split.value = self.splits.index(split)

    @property
    def window_size(self):
        """Receives the current window size."""
        return tuple(self._window_size)

    @window_size.setter
    def window_size(self, value: tuple[int, int]):
        """Sends the new window size to the render process."""
        self._window_size[0] = value[0]
        self._window_size[1] = value[1]

    @property
    def is_training(self):
        """Receives whether the model is currently training."""
        return self._training.value

    @is_training.setter
    def is_training(self, value: bool):
        """Sends whether the model is currently training."""
        self._training.value = value

    @property
    def training_iteration(self):
        """Receives the current training iteration."""
        return self._training_iteration.value

    @training_iteration.setter
    def training_iteration(self, value: int):
        """Sends the current training iteration."""
        self._training_iteration.value = value

    def terminateTraining(self):
        """Tells the render process to terminate the training."""
        self._terminate_training.value = True

    @property
    def should_terminate_training(self):
        """Receives whether the training should be terminated."""
        return self._terminate_training.value

    @property
    def config_requests(self) -> dict[str, Any]:
        """Receives the current configuration requests."""
        requests = {}
        while not self._config_request.empty():
            request = self._config_request.get_nowait()
            requests.update(request)

        return requests

    def requestConfigChange(self, request: dict[str, Any]):
        """Sends the configuration request to the render process."""
        try:
            self._config_request.put_nowait(request)
        except queue.Full:
            Framework.Logger.logWarning('Config Request Queue is full, discarding request.')
        except FileNotFoundError:
            Framework.Logger.logWarning('Config Request Queue does not exist anymore.')

    @property
    def config_options(self) -> list[tuple[str, Any]]:
        """Receives any advertised configuration options."""
        configs = []
        while not self._config_advertisements.empty():
            config = self._config_advertisements.get_nowait()
            configs.extend(config)

        return configs

    def advertiseConfig(self, configs: list[tuple[str, Any]]):
        """Advertises a list of configuration options to the GUI process."""
        try:
            self._config_advertisements.put_nowait(configs)
        except queue.Full:
            Framework.Logger.logWarning('Config Advertisement Queue is full, discarding request.')
        except FileNotFoundError:
            Framework.Logger.logWarning('Config Advertisement Queue does not exist anymore.')
