# -- coding: utf-8 --

"""Applications/AsyncLauncher.py: Provides functions for launching the GUI as a
   separate process for improved responsiveness."""

from typing import Unpack

import torch
import torch.multiprocessing as mp

from ICGui.GuiConfig import LaunchConfig
from ICGui.ModelRunners import ModelState
from ICGui.Applications.ViewerGUI import ViewerGUI, GuiKWArgs
from Cameras.Base import BaseCamera
from Cameras.utils import CameraProperties
from Datasets.Base import BaseDataset
from Logging import Logger

import Framework


def launchGuiProcess(gui_config: LaunchConfig, dataset: BaseDataset,
                     training: bool = False, **gui_kwargs: Unpack[GuiKWArgs]) -> tuple[ModelState, mp.Process]:
    """Launches the GUI as a separate process and returns synchronization objects."""
    shared_state = ModelState(start_as_training=training)

    previous_mode = dataset.mode
    dataset_poses = {}
    dataset_camera = None

    shared_state.splits = dataset.subsets.copy()
    # Reorder to ensure the default is val < train < test
    splits = shared_state.splits.copy()
    if 'val' in splits:
        splits.remove('val')
        splits.append('val')
    if 'train' in splits:
        splits.remove('train')
        splits.append('train')
    if 'test' in splits:
        splits.remove('test')
        splits.append('test')

    for mode in splits:
        try:
            dataset.setMode(mode)
            dataset_poses[mode] = [c.toSimple() for c in dataset]
            if dataset.camera is not None:
                dataset_camera = dataset.camera

            if len(dataset_poses[mode]) == 0 or dataset_camera is None:
                Logger.logInfo(f'No poses found for mode "{mode}".')
                shared_state.splits.remove(mode)
                if mode in dataset_poses:
                    del dataset_poses[mode]
            else:
                shared_state.renderGt(-1, mode)
        except Framework.DatasetError:
            Logger.logInfo(f'No poses found for mode "{mode}".')
            shared_state.splits.remove(mode)
            if mode in dataset_poses:
                del dataset_poses[mode]

    if len(dataset_poses) == 0 or dataset_camera is None:
        raise Framework.GUIError('No poses found for any mode. Aborting.')

    # Restore the dataset mode if in training mode, otherwise keep it as is
    dataset.setMode(previous_mode)

    dataset_bbox = dataset.getBoundingBox()
    bbox_size = torch.mean(dataset_bbox[1] - dataset_bbox[0]).item()

    proc = mp.Process(target=_launchGui, args=(gui_config, dataset_poses, dataset_camera, bbox_size, shared_state),
                      kwargs=gui_kwargs)
    proc.start()

    return shared_state, proc


def _launchGui(gui_config: LaunchConfig, dataset_poses: dict[str, list[CameraProperties]], dataset_camera: BaseCamera,
               bbox_size: float, shared_state: ModelState, **gui_kwargs: Unpack[GuiKWArgs]):
    """Target for launching the GUI in a separate process, initializes Framework and launches the GUI."""
    Framework.setup(require_custom_config=True, config_path=str(gui_config.training_config_path))
    gui = ViewerGUI(config=gui_config, dataset_poses=dataset_poses, dataset_camera=dataset_camera, bbox_size=bbox_size,
                    shared_state=shared_state, **gui_kwargs)
    gui.run()
