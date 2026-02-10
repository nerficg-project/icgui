"""Backend/CheckpointRunner.py: Implementation of an inference model runner that
loads a checkpoint from a given path and runs the model."""

import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, NoReturn

import torch.multiprocessing as mp

from ICGui.State.Shared import SharedState
from ICGui.util.FPSRollingAverage import FPSRollingAverage
from ICGui.util.Screenshots import save_screenshot
from ICGui.util.Transforms import transform_gt_image, transform_gt_changed

import Framework
from Cameras.Perspective import PerspectiveCamera
from Datasets.Base import BaseDataset
from Implementations import Methods as MI


class CheckpointRunner:
    """Model runner that loads a checkpoint from a given path and runs the model,
    interacting with the GUI process through the synchronization objects obtained
    from the Launcher."""
    def __init__(self, dataset: BaseDataset, state: SharedState, checkpoint_path: str | Path, *,
                 initial_resolution: list[int] = (1280, 720), initial_resolution_factor: float = 1.0, rolling_average_size: int = 10):
        self.dataset = dataset
        self.model = MI.get_model(Framework.config.GLOBAL.METHOD_TYPE, str(checkpoint_path))
        self.model.eval()
        self.renderer = MI.get_renderer(Framework.config.GLOBAL.METHOD_TYPE, self.model)
        self._state: SharedState = state
        self._fps = FPSRollingAverage(window_size=rolling_average_size)
        self._gt_image_cache = {'idx': -1, 'split': 'None', 'rgb': None}

        self._gui_view = deepcopy(dataset.default_view.to_simple())
        self._gui_view.width = int(initial_resolution[0] * initial_resolution_factor)
        self._gui_view.height = int(initial_resolution[1] * initial_resolution_factor)
        if isinstance(self._gui_view, PerspectiveCamera):
            self._gui_view.focal_x *= initial_resolution_factor
            self._gui_view.focal_y *= initial_resolution_factor
            self._gui_view.center_x *= initial_resolution_factor
            self._gui_view.center_y *= initial_resolution_factor

        self._advertise_renderer_config()

    def _advertise_renderer_config(self):
        """Sends additional configuration options specific to the renderer to the GUI."""
        config_options = []
        for key, value in self.renderer.__dict__.items():
            # Assume configuration options do not start with _ and are fully uppercase
            if key[0] == '_' or not key.isupper():
                continue
            config_options.append((f'Renderer.{key}', value))

        self._state.configurable_advertisements = config_options

    def _update_config_changes(self, config_changes: Mapping[str, Any]):
        """Updates the runner configuration with the given changes."""
        for key, value in config_changes.items():
            match key.split('.'):
                case ['Renderer', attr]:
                    if hasattr(self.renderer, attr):
                        setattr(self.renderer, attr, value)
                    else:
                        Framework.Logger.log_warning(f'Unknown renderer key {key} with value {value}')
                case _:
                    Framework.Logger.log_warning(f'Unknown configuration key {key} with value {value}')

    def model_step(self) -> Mapping[str, Any]:
        """Performs a single model render step and return the results as a dictionary of tensors."""
        # Update configuration if new configuration is available
        config_changes = self._state.configurable_changes
        if len(config_changes) > 0:
            self._update_config_changes(config_changes)

        # Save one screenshot per frame if requested
        screenshot_view = self._state.screenshot_view
        if screenshot_view is not None:
            view, color_mode, color_map = screenshot_view
            output = self.renderer.render_image(view)
            save_screenshot(output, view.timestamp, color_mode, color_map)
            del output

        # Update the requested view before each step
        new_view = self._state.view
        if new_view is not None:
            if isinstance(self._gui_view.camera, PerspectiveCamera) and isinstance(new_view.camera, PerspectiveCamera):
                # Check if the new camera has relevant changes, that affect transformation
                if transform_gt_changed(self._gui_view.camera, new_view.camera):
                    self._gt_image_cache['idx'] = -1  # Invalidate cache
                elif isinstance(self._gui_view.camera, PerspectiveCamera) ^ isinstance(new_view.camera, PerspectiveCamera):
                    self._gt_image_cache['idx'] = -1  # Camera mode changed, invalidate cache

            self._gui_view = new_view

        # TODO: Send as one unit to avoid race conditions
        gt_idx = self._state.gt_index
        gt_split = self._state.gt_split
        if gt_idx < 0 or not isinstance(self._gui_view.camera, PerspectiveCamera):
            self._fps.enable()
            self._fps.start_timer()
            output = {
                **self.renderer.render_image(self._gui_view),
                'view': self._gui_view,
                'type': 'render'
            }
            self._fps.update()
        else:
            self._fps.disable()  # Disable FPS calculation when showing GT
            if self._gt_image_cache['idx'] != gt_idx or self._gt_image_cache['split'] != gt_split:
                self._gt_image_cache['result'] = transform_gt_image(self.dataset.set_mode(gt_split), gt_idx,
                                                                    self._gui_view.camera)
                self._gt_image_cache['idx'] = gt_idx
                self._gt_image_cache['split'] = gt_split
            output = {
                **self._gt_image_cache['result'],
                'view': self._gui_view,
                'type': 'gt'
            }
        return output

    @Framework.catch()
    def model_step_with_framerate(self) -> Mapping[str, Any]:
        """Performs a single model render step and return the results as a dictionary of tensors, including the
        framerate."""
        return {
            **self.model_step(),
            **self._fps.stats,
        }

    def run(self, gui_process: mp.Process) -> NoReturn:
        """Continually runs the model until the GUI process is closed, the exits the process."""
        while gui_process.is_alive():
            self._state.frame = self.model_step_with_framerate()

        sys.exit(0)
