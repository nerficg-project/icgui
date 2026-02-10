"""util/Screenshots.py: Provides screenshot helper functions to save tensors."""

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import torch
from torchvision.utils import save_image

import Framework
from ICGui.util.ColorChannels import to_rgba
from Logging import Logger


_DIRECTORY_CHOICES = ['Screenshots', 'Pictures/Screenshots', 'Pictures']


def save_screenshot(imgs: Mapping[str, torch.Tensor],
                    timestamp: float,
                    color_mode: str,
                    color_map: dict[str, Any] = None,
                    **kwargs):
    """Saves the current frame in the user's screenshot directory."""
    rgba = to_rgba(imgs, color_mode, color_map).permute(2, 0, 1)
    if rgba is None:
        return

    # TODO: See if we can replace image saving with something supporting metadata
    # That way we could not just clean up the current screenshot name,
    # but also save additional data such as camera pose etc.

    screenshot_name_extras = f'T{timestamp:.2f}_{color_mode}'
    if color_map['color_map'] != 'Grayscale':
        screenshot_name_extras += f'_{color_map["color_map"]}'
    for key, value in kwargs.items():
        screenshot_name_extras += f'_{key}={value}'
    screenshot_path = get_screenshot_path(screenshot_name_extras)
    save_image(rgba, screenshot_path)
    Logger.log_info(f'Saved screenshot to {screenshot_path}')


def get_screenshot_path(screenshot_name_extras: str) -> Path:
    """Returns a path object to store the screenshot at."""
    # pylint: disable=invalid-name
    screenshot_name = f'{Framework.config.TRAINING.MODEL_NAME}_' \
                      f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")}_' \
                      f'{screenshot_name_extras}.png'

    try:
        home_path = Path.home()
        for path in _DIRECTORY_CHOICES:
            if (home_path / path).is_dir():
                # Model name may contain slashes, so we ensure the directory exists
                (home_path / path / screenshot_name).parent.mkdir(parents=True, exist_ok=True)
                return home_path / path / screenshot_name
        (home_path / 'Screenshots').mkdir(exist_ok=True)
        return home_path / 'Screenshots' / screenshot_name
    except RuntimeError:
        (Path.cwd() / 'Screenshots' / screenshot_name).parent.mkdir(parents=True, exist_ok=True)
        return Path.cwd() / 'Screenshots' / screenshot_name
