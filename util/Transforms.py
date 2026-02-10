"""util/Transforms.py: Utilities for image transformations."""

import torch

from Cameras.Perspective import PerspectiveCamera
from Datasets.Base import BaseDataset
from Datasets.utils import apply_background_color


def transform_gt_image(dataset: BaseDataset, idx: int, camera: PerspectiveCamera):
    """Transforms the ground truth image to match the current camera properties."""
    sample = dataset[idx]
    if not isinstance(sample.camera, PerspectiveCamera):
        raise NotImplementedError('Only PerspectiveCamera samples are supported for ground truth transformation.')

    gt_imgs = {
        channel: getattr(sample, channel).to(device='cuda')
        for channel in ['rgb', 'depth', 'alpha']
        if hasattr(sample, channel) and getattr(sample, channel) is not None
    }
    if 'rgb' in gt_imgs and 'alpha' in gt_imgs:
        gt_imgs['rgb'] = apply_background_color(gt_imgs['rgb'], gt_imgs['alpha'], camera.background_color)

    ## Calculate homography for warping the image onto the camera
    ## (The commented-out code effectively explains the logic behind the transformation,
    ##  below it is the simplified version without inversion / full matmul)
    # K_render = torch.tensor([
    #     [camera.focal_x / camera.width, 0, (camera.center_x / camera.width * 2.0 - 1.0)],
    #     [0, camera.focal_y / camera.height, (camera.center_y / camera.height * 2.0 - 1.0)],
    #     [0, 0, 1],
    # ], dtype=torch.float32, device=device)
    # K_gt = torch.tensor([
    #     [sample.focal_x / sample.width, 0, (sample.center_x / sample.width * 2.0 - 1.0)],
    #     [0, sample.focal_y / sample.height, (sample.center_y / sample.height * 2.0 - 1.0)],
    #     [0, 0, 1],
    # ], dtype=torch.float32, device=device)
    # H_mat = (K_gt @ torch.linalg.inv(K_render))[:2, :].unsqueeze(0)
    ## The above simplifies to:
    px = camera.center_x - 0.5 * camera.width
    py = camera.center_y - 0.5 * camera.height
    px_gt = sample.camera.center_x - 0.5 * sample.camera.width
    py_gt = sample.camera.center_y - 0.5 * sample.camera.height
    H_mat = torch.tensor([[
        [sample.camera.focal_x * camera.width / (camera.focal_x * sample.camera.width), 0, 2 * (camera.focal_x * px_gt - sample.camera.focal_x * px) / (camera.focal_x * sample.camera.width)],
        [0, sample.camera.focal_y * camera.height / (camera.focal_y * sample.camera.height), 2 * (camera.focal_y * py_gt - sample.camera.focal_y * py) / (camera.focal_y * sample.camera.height)],
    ]], dtype=torch.float32, device='cuda')

    # Calculate affine grid and apply
    for k, gt_img in gt_imgs.items():
        grid = torch.nn.functional.affine_grid(
            H_mat, [1, gt_img.shape[0], camera.height, camera.width], align_corners=True
        )
        gt_imgs[k] = torch.nn.functional.grid_sample(
            gt_img[None], grid, mode='bilinear', padding_mode='zeros', align_corners=True
        )[0].permute(1, 2, 0)

    return gt_imgs


def transform_gt_changed(cam: PerspectiveCamera, new_cam: PerspectiveCamera) -> bool:
    """Checks if the camera has changed in a way that affects the ground truth transformation."""
    return (
        cam.width != new_cam.width or cam.height != new_cam.height or
        cam.focal_x != new_cam.focal_x or cam.focal_y != new_cam.focal_y or
        cam.center_x != new_cam.center_x or cam.center_y != new_cam.center_y
    )
