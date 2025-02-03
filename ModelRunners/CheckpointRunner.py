# -- coding: utf-8 --

"""ModelRunners/CheckpointRunner.py: Implementation of the model runner that loads a checkpoint from a given path and
runs the model."""

import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, NoReturn

import torch
import torch.multiprocessing as mp

from ICGui.ModelRunners.Base import BaseModelRunner
from ICGui.ModelRunners.ModelState import ModelState
from ICGui.Viewers.ScreenshotUtils import saveScreenshot
from ICGui.Viewers.utils import transformGtImage

import Framework
from Datasets.Base import BaseDataset
from Implementations import Methods as MI


class CustomModelRunner(BaseModelRunner):
    """Model runner that loads a checkpoint from a given path and runs the model, interacting with the GUI process
    through the synchronization objects obtained from the AsyncLauncher."""
    def __init__(self, dataset: BaseDataset, state: ModelState, checkpoint_path: str | Path, *,
                 initial_resolution: list[int] = (1280, 720), rolling_average_size: int = 10):
        super().__init__(dataset, state, initial_resolution=initial_resolution,
                         rolling_average_size=rolling_average_size)
        checkpoint_path = str(checkpoint_path)

        self.dataset = dataset
        self.model = MI.getModel(Framework.config.GLOBAL.METHOD_TYPE, checkpoint_path)
        self.model.eval()
        self.renderer = MI.getRenderer(Framework.config.GLOBAL.METHOD_TYPE, self.model)
        self._gui_camera = deepcopy(dataset.camera)
        self._gt_image_cache = {
            'idx': -1,
            'rgb': None,
        }

        self._advertiseRendererConfig()

    def _advertiseRendererConfig(self):
        config_options = []
        for key, value in self.renderer.__dict__.items():
            # Assume configuration options do not start with _ and are fully uppercase
            if key[0] == '_' or not key.isupper():
                continue
            config_options.append((f'Renderer.{key}', value))

        self._state.advertiseConfig(config_options)

    def _updateConfig(self, config_changes: Mapping[str, Any]):
        """Updates the runner configuration with the given changes."""
        for key, value in config_changes.items():
            match key.split('.'):
                case ['Renderer', attr]:
                    if hasattr(self.renderer, attr):
                        setattr(self.renderer, attr, value)
                    else:
                        Framework.Logger.logWarning(f'Unknown renderer key {key} with value {value}')
                case _:
                    Framework.Logger.logWarning(f'Unknown configuration key {key} with value {value}')

    def modelStep(self) -> Mapping[str, torch.Tensor | None]:
        """Performs a single model render step and return the results as a dictionary of tensors."""
        # Update configuration if new configuration is available
        config_changes = self._state.config_requests
        if config_changes != {}:
            self._updateConfig(config_changes)

        # Save one screenshot per frame if requested
        screenshot_camera = self._state.screenshot_camera
        if screenshot_camera is not None:
            camera = screenshot_camera[0]
            color_mode = screenshot_camera[1]
            color_map = screenshot_camera[2]
            output = self.renderer.renderImage(camera)
            saveScreenshot(output, camera.properties.timestamp, color_mode, color_map)
            del output

        # Update the internal camera properties before each step
        new_camera = self._state.camera
        if new_camera is not None:
            self._gui_camera = new_camera
            self._gt_image_cache['idx'] = -1  # Invalidate cache

        gt_idx = self._state.gt_index
        gt_split = self._state.gt_split
        if gt_idx < 0:
            output = self.renderer.renderImage(self._gui_camera)
            output['type'] = 'render'
        else:
            if self._gt_image_cache['idx'] != gt_idx:
                self._gt_image_cache['result'] = transformGtImage(self.dataset.setMode(gt_split), gt_idx,
                                                                  self._gui_camera, self._state.window_size)
                self._gt_image_cache['idx'] = gt_idx
            output = {**self._gt_image_cache['result'], 'type': 'gt'}

        output['camera'] = self._gui_camera
        return output

    def runModel(self, gui_process: mp.Process) -> NoReturn:
        """Continually runs the model until the GUI process is closed, the exits the process."""
        self._state.running = True
        while gui_process.is_alive():
            self._state.frame = self.modelStepWithFramerate()

        sys.exit(0)
