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


def save_screenshot(
        imgs: Mapping[str, torch.Tensor],
        timestamp: float,
        color_mode: str,
        color_map: dict[str, Any] = None,
        name: str | None = None,
        directory: str | Path | None = None,
        verbose: bool = True,
        **kwargs
):
    """Saves the current frame in the user's screenshot directory."""
    rgba = to_rgba(imgs, color_mode, color_map)
    if rgba is None:
        Logger.log_warning('Cannot save screenshot as no valid output is available')
        return
    rgba = rgba.permute(2, 0, 1)

    screenshot_name_extras = f'T{timestamp:.2f}_{color_mode}'
    if color_map['color_map'] != 'Grayscale':
        screenshot_name_extras += f'_{color_map["color_map"]}'
    for key, value in kwargs.items():
        screenshot_name_extras += f'_{key}={value}'

    if directory is None:
        screenshot_path = get_screenshot_path(screenshot_name_extras)
    else:
        screenshot_path = Path(directory)
        screenshot_path.mkdir(parents=True, exist_ok=True)
        if name is None:
            screenshot_path /= get_screenshot_name(screenshot_name_extras)
        else:
            screenshot_path /= name
    save_image(rgba, screenshot_path)
    if verbose:
        Logger.log_info(f'Saved screenshot to {screenshot_path}')


def get_screenshot_name(screenshot_name_extras: str) -> str:
    return f'{Framework.config.TRAINING.MODEL_NAME}_' \
           f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")}_' \
           f'{screenshot_name_extras}.png'


def get_screenshot_path(screenshot_name_extras: str) -> Path:
    """Returns a path object to store the screenshot at."""
    # pylint: disable=invalid-name
    screenshot_name = get_screenshot_name(screenshot_name_extras)
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
