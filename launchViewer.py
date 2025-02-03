#! /usr/bin/env python3
# -- coding: utf-8 --

"""launchViewer.py: Launches the standalone GUI to view a trained model."""

if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.multiprocessing as mp

from ICGui.Applications.AsyncLauncher import launchGuiProcess
from ICGui.ModelRunners.CheckpointRunner import CustomModelRunner as CheckpointModelRunner
from ICGui.GuiConfig import LaunchConfig

import Framework
from Datasets.Base import BaseDataset
from Implementations import Datasets as DI
from Logging import Logger


@torch.no_grad()
def main():
    """Main entrypoint for the GUI application."""
    Logger.setMode(Logger.MODE_DEBUG)
    config = LaunchConfig.fromCommandLine()

    Framework.setup(require_custom_config=True, config_path=config.training_config_path)
    dataset: BaseDataset = DI.getDataset(
        dataset_type=Framework.config.GLOBAL.DATASET_TYPE,
        path=Framework.config.DATASET.PATH
    )

    shared_state, gui_process = launchGuiProcess(config, dataset)
    model_runner = CheckpointModelRunner(dataset, shared_state,
                                         checkpoint_path=config.checkpoint_path,
                                         initial_resolution=config.initial_resolution,
                                         rolling_average_size=config.fps_rolling_average_size)
    model_runner.runModel(gui_process)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
