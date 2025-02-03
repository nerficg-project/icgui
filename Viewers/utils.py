# -- coding: utf-8 --

"""Viewer/utils.py: Contains utility functions used for the implementation of the available viewer classes."""

from typing import Mapping

import torch
from torch.nn.functional import affine_grid, grid_sample

import Framework
from Datasets.Base import BaseDataset
from Cameras.Base import BaseCamera
from Logging import Logger
from Visual.ColorMap import ColorMap
from Visual.utils import pseudoColorDepth

_alpha = torch.ones(1, 1, 1, device='cuda')


# pylint: disable-next = too-few-public-methods
class ViewerError(Exception):
    """Raise in case of an exception regarding the viewer."""


def rgbToRgba(img: torch.Tensor):
    """Converts the given RGB tensor to RGBA format. The tensor must be on the GPU."""
    return torch.cat((img, _alpha.expand((img.shape[0], img.shape[1], 1))), dim=-1) \
        .contiguous(memory_format=torch.contiguous_format)


def monochromeToRgba(img: torch.Tensor, color_map: str = None):
    """Converts the given monochrome tensor to RGBA format. The tensor must be on the GPU."""
    if color_map is None or color_map == 'Grayscale':
        if len(img.shape) == 2:
            img = img.unsqueeze(-1)
        return rgbToRgba(img.expand((img.shape[0], img.shape[1], 3)))

    # Normalize to [0, 1]
    min_val = img.min().item()
    max_val = img.max().item()
    normalized = torch.clamp((img - min_val) / (max_val - min_val), min=0.0, max=1.0)

    # Get color map
    color_map = ColorMap.get(color_map)

    # Apply color map
    depth = (torch.index_select(color_map, dim=0, index=(normalized * 255).int().flatten())
             .reshape(*normalized.shape[:2], 3))

    return rgbToRgba(depth)


def depthToRgba(depth: torch.Tensor, alpha: torch.Tensor, color_map: str):
    """Converts the given depth tensor to RGBA format. The tensor must be on the GPU."""
    depth = depth.permute((2, 0, 1))
    alpha = alpha.permute((2, 0, 1))

    depth = pseudoColorDepth(
        color_map=color_map,
        depth=depth,
        near_far=None,
        # near_far=near_far,
        alpha=alpha,
        interpolate=False,
    )
    return rgbToRgba(depth.permute(1, 2, 0))


# pylint: disable=too-many-return-statements
def toRgba(imgs: Mapping[str, torch.Tensor],
           color_mode: str, color_map: str = 'Grayscale') -> torch.Tensor | None:
    """Converts the given tensor to RGBA format. The tensor must be on the GPU."""
    if color_mode not in imgs.keys():
        Logger.logWarning(f'Received frame does not contain selected `{color_mode}` output, defaulting to `rgb`')
        color_mode = 'rgb'

    # Special cases
    if color_mode == 'depth' \
            and 'depth' in imgs.keys() \
            and 'alpha' in imgs.keys() \
            and color_map != 'Grayscale':
        depth = imgs['depth']
        alpha = imgs['alpha']

        if not isinstance(depth, torch.Tensor):
            Logger.logWarning(f'SDL2-Viewer: Received invalid tensor of type {type(depth)}, skipping frame')
            return None
        if not isinstance(alpha, torch.Tensor):
            Logger.logWarning(f'SDL2-Viewer: Received invalid tensor of type {type(alpha)}, skipping frame')
            return None
        return depthToRgba(depth, alpha, color_map)

    # Generic conversion methods
    img = imgs[color_mode]
    if not isinstance(img, torch.Tensor):
        Logger.logWarning(f'SDL2-Viewer: Received invalid tensor of type {type(img)}, skipping frame')
        return None

    if len(img.shape) == 3 and img.shape[2] == 4:
        return img
    if len(img.shape) == 3 and img.shape[2] == 3:
        return rgbToRgba(img)
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        return monochromeToRgba(img, color_map)

    Logger.logWarning(f'SDL2-Viewer: Could not determine conversion for tensor of shape {img.shape}, '
                      f'skipping frame')
    return None


# pylint: disable=too-many-locals
def transformGtImage(dataset: BaseDataset, idx: int, camera: BaseCamera, window_size: tuple[int, int]):
    """Transforms the ground truth image to match the current camera properties."""
    sample = dataset[idx]
    gt_imgs = {
        channel: getattr(sample, channel).to(device='cuda')
        for channel in ['rgb', 'depth', 'alpha']
        if hasattr(sample, channel) and getattr(sample, channel) is not None
    }

    dataset_scale = Framework.config.DATASET.IMAGE_SCALE_FACTOR or 1.0

    s_x = camera.properties.width / sample.width
    s_y = camera.properties.height / sample.height
    f_x = (sample.focal_x * dataset_scale) / camera.properties.focal_x
    f_y = (sample.focal_y * dataset_scale) / camera.properties.focal_y
    p_x_old = (sample.principal_offset_x * dataset_scale) / sample.width
    p_y_old = (sample.principal_offset_y * dataset_scale) / sample.height
    p_x_new = camera.properties.principal_offset_x / camera.properties.width
    p_y_new = camera.properties.principal_offset_y / camera.properties.height

    affine_matrix = torch.tensor([
        [s_x * f_x, 0, -p_x_new * s_x + p_x_old * s_x * f_x],
        [0, s_y * f_y, -p_y_new * s_y + p_y_old * s_y * f_y],
    ], device='cuda', dtype=torch.float32)

    for k, gt_img in gt_imgs.items():
        grid = affine_grid(affine_matrix.unsqueeze(0), [1, gt_img.shape[0], window_size[1], window_size[0]],
                           align_corners=True)
        gt_imgs[k] = grid_sample(gt_img.unsqueeze(0), grid, mode='bicubic', padding_mode='zeros',
                                 align_corners=True).squeeze(0).permute((1, 2, 0))

    return gt_imgs
