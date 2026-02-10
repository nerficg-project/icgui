"""util/Runner.py: Provides functions for launching the GUI as a separate process for improved responsiveness."""

import atexit
from collections import OrderedDict

import torch
import torch.multiprocessing as mp

from ICGui.Applications import Viewer
from ICGui.State import LaunchConfig, SharedState
from Cameras.Base import BaseCamera
from Datasets.Base import BaseDataset
from Datasets.utils import View
from Logging import Logger

import Framework


def launch_gui_process(gui_config: LaunchConfig, dataset: BaseDataset,
                       training: bool = False, **gui_kwargs) -> tuple[SharedState, mp.Process]:
    """Launches the GUI as a separate process and returns synchronization objects."""
    shared_state = SharedState(start_as_training=training)

    # Add train, val, and test splits in that order if they exist in the dataset, then add any other subsets
    shared_state.splits = [split for split in ['train', 'val', 'test'] if split in dataset.subsets]
    shared_state.splits += [split for split in dataset.subsets if split not in shared_state.splits]
    if len(shared_state.splits) == 0:
        # FIXME: Test datasets without splits
        raise Framework.GUIError('No splits found in the dataset. Aborting.')

    # Collect poses for each mode in the dataset
    dataset_poses = OrderedDict()
    for split in shared_state.splits.copy():
        if dataset.data[split] is None or len(dataset.data[split]) < 1:
            Logger.log_info(f'No poses found for split "{split}".')
            shared_state.splits.remove(split)
            continue
        dataset_poses[split] = [pose.to_simple() for pose in dataset.data[split]]

    # Check if the dataset has a default view
    default_view = dataset.default_view.to_simple()
    if default_view is None or not isinstance(default_view, View):
        raise Framework.GUIError('No default view found in the dataset. Aborting.')

    # Check if the default view contains a valid camera
    if default_view.camera is None or not isinstance(default_view.camera, BaseCamera):
        raise Framework.GUIError('No valid camera found in the dataset default view. Aborting.')

    # Get scene scale from the bounding box of the dataset
    bbox_size = torch.mean(dataset.bounding_box.size).item()

    # Launch the GUI process
    # TODO: Send over list of dataset cameras
    proc = mp.Process(target=_launch_gui, kwargs={
        'gui_config': gui_config,
        'dataset_poses': dataset_poses,
        'default_view': default_view,
        'bbox_size': bbox_size,
        'shared_state': shared_state,
        **gui_kwargs,
    })
    proc.start()
    atexit.register(lambda: shared_state.renderer_exited())

    # Forking the process resets the log-level in the parent process, so we need to set it again
    Logger.set_mode(Framework.config.GLOBAL.LOG_LEVEL)

    return shared_state, proc


def _launch_gui(*, gui_config: LaunchConfig, dataset_poses: dict[str, list[View]], default_view: View,
                bbox_size: float, shared_state: SharedState, **gui_kwargs):
    """Target for launching the GUI in a separate process, initializes Framework and launches the GUI."""
    default_view.camera.shared_settings.background_color = default_view.camera.shared_settings.background_color.cpu()
    Framework.setup(require_custom_config=True, config_path=str(gui_config.training_config_path))
    gui = Viewer(config=gui_config, dataset_poses=dataset_poses, default_view=default_view, bbox_size=bbox_size,
                 shared_state=shared_state, **gui_kwargs)
    gui.run()
