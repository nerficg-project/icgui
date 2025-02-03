# -- coding: utf-8 --

"""ModelRunners/Base.py: Abstract Base Class for the model runner, interfacing the model with the GUI process."""

from abc import ABC, abstractmethod
from typing import Mapping, NoReturn

import torch
import torch.multiprocessing as mp

from ICGui.ModelRunners.FPSRollingAverage import FPSRollingAverage
from ICGui.ModelRunners.ModelState import ModelState
from Datasets.Base import BaseDataset


class BaseModelRunner(ABC):
    """Abstract Base Class for the model runner, interfacing the model with the GUI process using the synchronization
    objects created by the AsyncLauncher"""
    def __init__(self, dataset: BaseDataset, state: ModelState, *,
                 initial_resolution: list[int] = (1280, 720),
                 rolling_average_size: int = 10):
        super().__init__()
        self.dataset = dataset
        self._fps = FPSRollingAverage(window_size=rolling_average_size)
        self._state: ModelState = state

        # Default camera properties to first camera in dataset
        camera_properties = self.dataset[0]
        camera_properties.width = initial_resolution[0]
        camera_properties.height = initial_resolution[1]
        camera_properties.rgb = None
        camera_properties.alpha = None
        camera_properties.depth = None
        self.dataset.camera.setProperties(camera_properties)

    @abstractmethod
    def modelStep(self) -> Mapping[str, torch.Tensor | None]:
        """Performs a single model render step and return the results as a dictionary of tensors."""
        pass

    def modelStepWithFramerate(self) -> Mapping[str, torch.Tensor | None]:
        """Performs a single model render step and return the results as a dictionary of tensors, including the
        framerate."""
        self._fps.update()

        return {
            **self.modelStep(),
            **self._fps.stats,
        }

    @abstractmethod
    def runModel(self, gui_process: mp.Process) -> NoReturn:
        """Continually runs the model until the GUI process is closed, the exits the process."""
        pass
