# -- coding: utf-8 --

"""ScreenshotUtils.py: Provides screenshot helper functions to save tensors."""

from datetime import datetime
from pathlib import Path
from typing import Mapping

import torch
from torchvision.utils import save_image

import Framework
from ICGui.Viewers.utils import toRgba
from Logging import Logger


def saveScreenshot(imgs: Mapping[str, torch.Tensor],
                   timestamp: float,
                   color_mode: str,
                   color_map: str = 'Grayscale',
                   **kwargs):
    """Saves the current frame in the user's screenshot directory."""
    rgba = toRgba(imgs, color_mode, color_map).permute((2, 0, 1))
    if rgba is None:
        return

    screenshot_name_extras = f'T{timestamp:.4f}_{color_mode}'
    if color_map != 'Grayscale':
        screenshot_name_extras += f'_{color_map}'
    for key, value in kwargs.items():
        screenshot_name_extras += f'_{key}={value}'
    screenshot_path = getScreenshotPath(screenshot_name_extras)
    save_image(rgba, screenshot_path)
    Logger.logInfo(f'SDL2-Viewer: Saved screenshot to {screenshot_path}')


def getScreenshotPath(screenshot_name_extras: str) -> Path:
    """Returns a path object to store the screenshot at."""
    # pylint: disable=invalid-name
    DIRECTORY_CHOICES = ['Screenshots', 'Pictures/Screenshots', 'Pictures']
    screenshot_name = f'{Framework.config.TRAINING.MODEL_NAME}_' \
                      f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")}_' \
                      f'{screenshot_name_extras}.png'

    try:
        home_path = Path.home()
        for path in DIRECTORY_CHOICES:
            if (home_path / path).is_dir():
                return home_path / path / screenshot_name
        (home_path / 'Screenshots').mkdir(exist_ok=True)
        return home_path / 'Screenshots' / screenshot_name
    except RuntimeError:
        (Path.cwd() / 'Screenshots').mkdir(exist_ok=True)
        return Path.cwd() / 'Screenshots' / screenshot_name
