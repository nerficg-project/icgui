"""State/Shared.py: Handles exchanging shared IPC state between the GUI and render processes"""

from __future__ import annotations  # Type hinting of multiprocessing Queues and Values fails without this in Python < 3.14

import queue
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import torch
import torch.multiprocessing as mp

import Framework
from Datasets.utils import View


_FRAME_QUEUE_SIZE = 3
_CAMERA_QUEUE_SIZE = 2
_SCREENSHOT_QUEUE_SIZE = 8
_TRAJECTORY_REQUEST_QUEUE_SIZE = 1
_CONFIG_ADVERTISEMENT_QUEUE_SIZE = 8
_CONFIG_REQUEST_QUEUE_SIZE = 32


@dataclass(slots=True)
class SharedState:
    """Handles exchanging shared state between the GUI and render processes by using queues to send tensors."""
    splits: list[str] = field(default_factory=lambda: ['train', 'val', 'test'])
    start_as_training: bool = field(default=False, repr=False)

    # IPC Queues
    _frame_queue: mp.Queue[dict[str, View | torch.Tensor | float]] = \
        field(default_factory=lambda: mp.Queue(_FRAME_QUEUE_SIZE))
    _render_view_queue: mp.Queue[View] = (
        field(default_factory=lambda: mp.Queue(_CAMERA_QUEUE_SIZE)))
    _configurable_advertisement: mp.Queue[list[tuple[str, Any]]] = \
        field(default_factory=lambda: mp.Queue(_CONFIG_ADVERTISEMENT_QUEUE_SIZE))
    _configurable_change: mp.Queue[dict[str, Any]] = (
        field(default_factory=lambda: mp.Queue(_CONFIG_REQUEST_QUEUE_SIZE)))
    _screenshot_requests: mp.Queue[tuple[View, str, dict[str, Any]]] = \
        field(default_factory=lambda: mp.Queue(_SCREENSHOT_QUEUE_SIZE))

    # Trajectory rendering state
    _trajectory_rendering_request: mp.Queue[tuple[list[View], str, dict[str, Any]]] = \
        field(default_factory=lambda: mp.Queue(_TRAJECTORY_REQUEST_QUEUE_SIZE))
    _trajectory_progress_num_frames: mp.Value[int] = \
        field(default_factory=lambda: mp.get_context('spawn').Value('i', 0))
    _trajectory_cancel_requested: mp.Value[bool] = \
        field(default_factory=lambda: mp.get_context('spawn').Value('b', False))
    _current_trajectory_request: tuple[list[View], str, dict[str, Any]] | None = None
    _current_trajectory_dirname: str | None = None

    # Ground truth request: atomic packed value (split index << 32 | GT index + 1).
    # Offset +1 enables storing the sentinel value -1 (use live render instead of GT)
    _gt_request: mp.Value[int] = field(default_factory=lambda: mp.get_context('spawn').Value('q', 0))

    # Training state
    _training: mp.Value[bool] = field(default_factory=lambda: mp.get_context('spawn').Value('b', False))
    _training_iteration: mp.Value[int] = field(default_factory=lambda: mp.get_context('spawn').Value('i', -1))
    _terminate_training: mp.Value[bool] = field(default_factory=lambda: mp.get_context('spawn').Value('b', False))
    _renderer_exited: mp.Value[bool] = field(default_factory=lambda: mp.get_context('spawn').Value('b', False))

    def __post_init__(self):
        if self.start_as_training:
            self.is_training = True

    # Getters / Setters for IPC Queues and Values

    @property
    def frame(self) -> dict[str, View | torch.Tensor | float | str] | None:
        """Receives a frame from the renderer process or None if no new frame has been sent."""
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None
        except FileNotFoundError:
            Framework.Logger.log_info('Frame Queue does not exist anymore.')
            return None

    @frame.setter
    def frame(self, value: dict[str, View | torch.Tensor | float | str]):
        """Sends a frame to the GUI process; if the queue is full, the frame is skipped."""
        try:
            self._frame_queue.put(value, timeout=1.0)
        except queue.Full:
            Framework.Logger.log_debug('Frame queue has been full for more than 1 second, skipping frame')
        except FileNotFoundError:
            Framework.Logger.log_warning('Frame queue does not exist anymore, discarding frame')

    @property
    def view(self) -> View | None:
        """Receives a view to render from the GUI process or None if no new view has been sent."""
        try:
            view = self._render_view_queue.get_nowait()
            return view
        except queue.Empty:
            return None
        except FileNotFoundError:
            Framework.Logger.log_info('Camera Queue does not exist anymore.')
            return None

    @view.setter
    def view(self, value: View):
        """Sends a new view to the render process, removing any older views if they are still in
        the queue."""
        try:
            # Always leave at least 1 element in the queue. This avoids race conditions when the renderer tries to
            #   grab a new camera inbetween an old one being removed from the GUI and a new one being added. These can
            #   otherwise be seen as stutters (due to the model timestamp not getting updated). One downside is that
            #   this may delay the camera update by 1 GUI frame (i.e. at 60 FPS an additional 0.0167 seconds).
            if self._render_view_queue.qsize() > 1:
                self._render_view_queue.get_nowait()
        except queue.Empty:
            pass
        except FileNotFoundError:
            Framework.Logger.log_info('Camera Queue does not exist anymore, discarding frame.')
            return

        self._render_view_queue.put(value)

    @property
    def screenshot_view(self) -> tuple[View, str, dict[str, Any]] | None:
        """Receives a new camera from the GUI process to render a screenshot, or None if no
        new screenshot camera has been sent."""
        try:
            view, color_mode, color_map = self._screenshot_requests.get_nowait()
            return view, color_mode, color_map
        except queue.Empty:
            return None
        except FileNotFoundError:
            Framework.Logger.log_info('Screenshot Queue does not exist anymore.')
            return None

    def screenshot(self, view: View, color_mode: str, color_map: dict[str, Any]):
        """Sends the new camera to the render process, removing any older cameras if they are still in
        the queue."""
        try:
            self._screenshot_requests.put_nowait((view, color_mode, color_map))
        except queue.Full:
            Framework.Logger.log_warning('Screenshot Queue is full, discarding screenshot request.')
        except FileNotFoundError:
            Framework.Logger.log_warning('Screenshot Queue does not exist anymore.')

    @property
    def _trajectory_views(self) -> tuple[list[View], str, dict[str, Any]] | None:
        """Receives the views of the current trajectory rendering request, or None if none is pending."""
        try:
            views, color_mode, color_map = self._trajectory_rendering_request.get_nowait()
            return views, color_mode, color_map
        except queue.Empty:
            return None
        except FileNotFoundError:
            Framework.Logger.log_info('Trajectory Request Queue does not exist anymore.')
            return None

    @property
    def trajectory_view(self) -> tuple[View, str, dict[str, Any], str] | None:
        """
        Returns the next view of the current trajectory rendering request, or None if no trajectory is being rendered.
        Note: The trajectory_progress_num_frames index needs to be updated manually to correctly track the progress
              of the trajectory rendering and to ensure a new frame is grabbed for the next view.
        """
        if self.trajectory_cancel_requested:
            self._current_trajectory_request = None
            self._current_trajectory_dirname = None
            self.trajectory_progress_num_frames = 0
            self.trajectory_cancel_requested = False
            return None

        if self._current_trajectory_request is None:
            self._current_trajectory_request = self._trajectory_views
            if self._current_trajectory_request is not None:
                self.trajectory_progress_num_frames = 0
                self._current_trajectory_dirname = f'trajectory_{datetime.now():%Y-%m-%d-%H-%M-%S}'
        if self._current_trajectory_request is None:
            return None

        views, color_mode, color_map = self._current_trajectory_request
        if self.trajectory_progress_num_frames >= len(views):
            self._current_trajectory_request = None
            return None

        view = views[self.trajectory_progress_num_frames]
        return view, color_mode, color_map, self._current_trajectory_dirname

    def render_trajectory(self, views: list[View], color_mode: str, color_map: dict[str, Any]) -> bool:
        """Sends a trajectory rendering request to the render process."""
        self.trajectory_progress_num_frames = 0
        self.trajectory_cancel_requested = False
        try:
            self._trajectory_rendering_request.put_nowait((views, color_mode, color_map))
            return True
        except queue.Full:
            Framework.Logger.log_warning('Trajectory Request Queue is full, discarding trajectory request.')
        except FileNotFoundError:
            Framework.Logger.log_warning('Trajectory Request Queue does not exist anymore.')
        return False

    @property
    def trajectory_progress_num_frames(self) -> int:
        """Receives the current number of rendered frames for the current trajectory rendering request."""
        return self._trajectory_progress_num_frames.value

    @trajectory_progress_num_frames.setter
    def trajectory_progress_num_frames(self, value: int):
        """Sends the current number of rendered frames for the current trajectory rendering request."""
        self._trajectory_progress_num_frames.value = value

    @property
    def trajectory_cancel_requested(self) -> bool:
        """Returns whether the current trajectory rendering request should be canceled."""
        return self._trajectory_cancel_requested.value

    @trajectory_cancel_requested.setter
    def trajectory_cancel_requested(self, value: bool):
        """Requests cancellation of the current trajectory rendering request."""
        self._trajectory_cancel_requested.value = value

    def cancel_trajectory_rendering(self):
        """Requests cancellation of the current trajectory rendering request."""
        self.trajectory_cancel_requested = True

    @property
    def gt_request(self) -> tuple[int, str]:
        """Receives the (index, split) of the GT image to render as one atomic unit.
        Index -1 means no GT, i.e. the live renderer should be used."""
        packed = self._gt_request.value
        split_idx = packed >> 32
        return (packed & 0xFFFFFFFF) - 1, self.splits[split_idx]

    def render_gt(self, idx: int, split: str):
        """Tells the render process to render the GT image at the given index and split,
        sent as one atomic unit."""
        self._gt_request.value = (self.splits.index(split) << 32) | ((idx + 1) & 0xFFFFFFFF)

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

    @property
    def terminate_training(self):
        """Receives whether the training should be terminated."""
        return self._terminate_training.value

    @terminate_training.setter
    def terminate_training(self, value: bool):
        """Sends whether the training should be terminated."""
        self._terminate_training.value = value

    @property
    def has_renderer_exited(self):
        """Receives whether the renderer exited or not."""
        return self._renderer_exited.value

    def renderer_exited(self):
        """Sends a notification to the GUI that the renderer has exited."""
        self._renderer_exited.value = True

    @property
    def configurable_changes(self) -> dict[str, Any]:
        """Receives the current configuration requests."""
        requests = {}
        while not self._configurable_change.empty():
            request = self._configurable_change.get_nowait()
            requests.update(request)

        return requests

    @configurable_changes.setter
    def configurable_changes(self, request: dict[str, Any]):
        """Sends the configuration request to the render process."""
        try:
            self._configurable_change.put_nowait(request)
        except queue.Full:
            Framework.Logger.log_warning('Config Request Queue is full, discarding request.')
        except FileNotFoundError:
            Framework.Logger.log_warning('Config Request Queue does not exist anymore.')

    @property
    def configurable_advertisements(self) -> list[tuple[str, Any]]:
        """Receives any advertised configuration options."""
        configs = []
        while not self._configurable_advertisement.empty():
            config = self._configurable_advertisement.get_nowait()
            configs.extend(config)

        return configs

    @configurable_advertisements.setter
    def configurable_advertisements(self, configs: list[tuple[str, Any]]):
        """Advertises a list of configuration options to the GUI process."""
        try:
            self._configurable_advertisement.put_nowait(configs)
        except queue.Full:
            Framework.Logger.log_warning('Config Advertisement Queue is full, discarding request.')
        except FileNotFoundError:
            Framework.Logger.log_warning('Config Advertisement Queue does not exist anymore.')
