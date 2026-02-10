"""ModelViewer/utils.py: Contains color conversion utility functions for the model viewer."""

import math
from typing import Any, Mapping

import torch

from Logging import Logger
from Visual.utils import apply_color_map as _apply_color_map


_alpha = torch.ones(1, 1, 1, device='cuda')


def apply_color_map(color_map: str, image: torch.Tensor, logscale: bool = False,
                    min_max: tuple[float, float] | None = None,
                    interpolate: bool = False, invert: bool = False):
    if logscale:
        image = torch.log1p(image)
        if min_max is not None:
            min_max = (math.log1p(min_max[0]), math.log1p(min_max[1]))

    return _apply_color_map(
        color_map=color_map,
        image=image,
        min_max=min_max,
        interpolate=interpolate,
        invert=invert,
    )


def monochrome_to_rgba(img: torch.Tensor, color_map_settings: dict[str, Any] | None):
    """Converts the given monochrome tensor to RGBA format. The tensor must be on the GPU."""
    if len(img.shape) == 2:
        img = img.unsqueeze(-1)

    color_mapped = apply_color_map(image=img.permute(2, 0, 1), **color_map_settings).permute(1, 2, 0)

    return rgb_to_rgba(color_mapped)


def rgb_to_rgba(img: torch.Tensor):
    """Converts the given RGB tensor to RGBA format. The tensor must be on the GPU."""
    return torch.cat((img, _alpha.expand(img.shape[0], img.shape[1], 1)), dim=-1).contiguous()


# pylint: disable=too-many-return-statements
def to_rgba(imgs: Mapping[str, torch.Tensor],
            color_mode: str, color_map: dict[Any, str] | None = None) -> torch.Tensor | None:
    """Converts the given tensor to RGBA format. The tensor must be on the GPU."""
    if color_mode not in imgs.keys():
        Logger.log_warning(f'Received frame does not contain selected `{color_mode}` output, defaulting to `rgb`')
        color_mode = 'rgb'

    img = imgs[color_mode]
    if not isinstance(img, torch.Tensor):
        Logger.log_warning(f'ModelViewer: Received invalid tensor of type {type(img)}, skipping frame')
        return None

    if len(img.shape) == 3 and img.shape[2] == 4:
        return img
    if len(img.shape) == 3 and img.shape[2] == 3:
        return rgb_to_rgba(img)
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        return monochrome_to_rgba(img, color_map)

    Logger.log_warning(f'ModelViewer: Could not determine conversion for tensor of shape {img.shape}, '
                      f'skipping frame')
    return None
