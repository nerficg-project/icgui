"""util/Cameras.py: Utilities for camera definitions and configurations."""

from copy import deepcopy
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, TypeVar, Generic

import numpy as np

from Cameras.Base import BaseCamera
from Cameras.Equirectangular import EquirectangularCamera
from Cameras.Perspective import PerspectiveCamera
from Cameras.utils import BaseDistortion, RadialTangentialDistortion, SharedCameraSettings

_T = TypeVar('_T')

@dataclass(frozen=True)
class ConfigurableProperty(Generic[_T]):
    key: str
    name: str
    property_type: type[_T]
    default: _T

    # Numerical properties
    min_value: _T | None = None
    max_value: _T | None = None
    change_speed: _T | None = None

    # Format specifier for number inputs
    format: _T | None = None


@dataclass(frozen=True)
class CameraDefinition:
    name: str
    camera_class: type[BaseCamera]
    configurable_properties: list[ConfigurableProperty]


@dataclass(frozen=True)
class DistortionDefinition:
    name: str
    distortion_class: type[BaseDistortion]
    configurable_properties: list[ConfigurableProperty]


_CUBE_SCALE = ConfigurableProperty(
    key='cube_scale',
    name='Cube Scale',
    property_type=float,
    default=1.0,
    min_value=0.0,
    max_value=10.0,
    change_speed=0.005,
)

CAMERA_TYPES: list[CameraDefinition] = [
    CameraDefinition('Perspective', PerspectiveCamera, []),
    CameraDefinition('Equirectangular', EquirectangularCamera, []),
]

DISTORTION_TYPES: list[DistortionDefinition] = [
    DistortionDefinition('Radial-Tangential Distortion', RadialTangentialDistortion, [
        ConfigurableProperty(key='k1', name='K1', property_type=float, default=0.0,
            min_value=-1.0, max_value=1.0, change_speed=0.005,
        ),
        ConfigurableProperty(key='k2', name='K2', property_type=float, default=0.0,
            min_value=-1.0, max_value=1.0, change_speed=0.005,
        ),
        ConfigurableProperty(key='k3', name='K3', property_type=float, default=0.0,
            min_value=-1.0, max_value=1.0, change_speed=0.005,
        ),
        ConfigurableProperty(key='p1', name='P1', property_type=float, default=0.0,
            min_value=-1.0, max_value=1.0, change_speed=0.005,
        ),
        ConfigurableProperty(key='p2', name='P2', property_type=float, default=0.0,
            min_value=-1.0, max_value=1.0, change_speed=0.005,
        ),
        ConfigurableProperty(key='undistortion_iterations', name='Iterations', property_type=int, default=10,
            min_value=1, max_value=100, change_speed=1,
        ),
        ConfigurableProperty(key='undistortion_eps', name='Undistortion ε', property_type=float, default=1e-9,
            min_value=1e-11, max_value=1.0, change_speed=1e-10, format='%.10f',
        ),
    ]),
]


def get_camera_idx(camera: BaseCamera) -> int:
    """Get the index of the camera type in the CAMERA_TYPES list."""
    for idx, cam_def in enumerate(CAMERA_TYPES):
        if isinstance(camera, cam_def.camera_class):
            return idx
    raise ValueError(f'Camera type {type(camera)} is not yet supported by the GUI. '
                     f'Please add a definition to the source file at: {Path(__file__).absolute()}')


def get_distortion_idx(distortion: BaseDistortion | None) -> int:
    """Get the index of the distortion type in the DISTORTION_TYPES list."""
    if distortion is None:
        return -1
    for idx, dist_def in enumerate(DISTORTION_TYPES):
        if isinstance(distortion, dist_def.distortion_class):
            return idx
    raise ValueError(f'Distortion type {type(distortion)} is not yet supported by the GUI. '
                     f'Please add a definition to the source file at: {Path(__file__).absolute()}')


def camera_to_dict(camera: BaseCamera) -> dict[str, Any]:
    """Serializes a camera into a human-readable, JSON-compatible dict."""
    def to_dict(obj, model: str, skip: tuple[str, ...] = ()) -> dict[str, Any]:
        values = {f.name: getattr(obj, f.name) for f in fields(obj) if f.init and f.name not in skip}
        return {'model': model} | {k: v.item() if isinstance(v, np.generic) else v for k, v in values.items()}

    data = to_dict(camera, CAMERA_TYPES[get_camera_idx(camera)].name, skip=('shared_settings', 'distortion'))
    if hasattr(camera, 'distortion'):
        distortion = camera.distortion
        data['distortion'] = None if distortion is None else to_dict(
            distortion, DISTORTION_TYPES[get_distortion_idx(distortion)].name)
    data['near_plane'] = float(camera.near_plane)
    data['far_plane'] = float(camera.far_plane)
    data['background_color'] = camera.shared_settings.background_color_numpy.tolist()
    return data


def camera_from_dict(data: dict[str, Any]) -> BaseCamera:
    """Deserializes a camera from a dict created by camera_to_dict."""
    data = dict(data)
    camera_class = {cam.name: cam.camera_class for cam in CAMERA_TYPES}[data.pop('model')]
    shared_settings = SharedCameraSettings(
        background_color=np.array(data.pop('background_color'), dtype=np.float32),
        near_plane=data.pop('near_plane'),
        far_plane=data.pop('far_plane'),
    )
    if (distortion := data.get('distortion')) is not None:
        distortion = dict(distortion)
        distortion_class = {dist.name: dist.distortion_class for dist in DISTORTION_TYPES}[distortion.pop('model')]
        data['distortion'] = distortion_class(**distortion)
    return camera_class(shared_settings=shared_settings, **data)


def rescale_camera(camera: BaseCamera, width: int, height: int,
                   crop: bool = False, maintain_pixel_aspect: bool = True) -> BaseCamera:
    """Returns a copy of the camera adapted to the given resolution. Cropping keeps the focal length, i.e. changes
    the field of view, stretching scales it either uniformly (maintaining the field of view in one axis and growing
    it in the other) or per axis."""
    camera = deepcopy(camera)
    if isinstance(camera, PerspectiveCamera):
        factor_x, factor_y = width / camera.width, height / camera.height
        if crop:
            factor_x = factor_y = 1.0
        elif maintain_pixel_aspect:
            factor_x = factor_y = min(factor_x, factor_y)
        camera.focal_x *= factor_x
        camera.focal_y *= factor_y
        camera.center_x = (camera.center_x - camera.width / 2) * factor_x + width / 2
        camera.center_y = (camera.center_y - camera.height / 2) * factor_y + height / 2
    camera.width, camera.height = width, height
    return camera


def interpolate_cameras(from_camera: BaseCamera, to_camera: BaseCamera, t: float) -> BaseCamera:
    """Interpolates the intrinsics and shared settings of two cameras of equal resolution. Settings that cannot be
    interpolated, i.e. a change of the camera model or distortion type, cut hard to to_camera at t = 1."""
    camera = deepcopy(from_camera)
    if type(from_camera) is not type(to_camera):
        return camera

    def lerp(from_value, to_value):
        return from_value + (to_value - from_value) * t

    # Set near/far on the shared settings directly, the camera setters validate against the outdated other plane
    shared_settings, to_shared_settings = camera.shared_settings, to_camera.shared_settings
    shared_settings.near_plane = lerp(shared_settings.near_plane, to_shared_settings.near_plane)
    shared_settings.far_plane = lerp(shared_settings.far_plane, to_shared_settings.far_plane)
    shared_settings.background_color = lerp(shared_settings.background_color_numpy, to_shared_settings.background_color_numpy)

    if isinstance(camera, PerspectiveCamera):
        for key in ('focal_x', 'focal_y', 'center_x', 'center_y'):
            setattr(camera, key, lerp(getattr(camera, key), getattr(to_camera, key)))
        if camera.distortion is not None and type(camera.distortion) is type(to_camera.distortion):
            camera.distortion = replace(camera.distortion, **{
                key: lerp(getattr(camera.distortion, key), getattr(to_camera.distortion, key))
                for key in ('k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'p1', 'p2')
            })
    return camera


def calculate_similarity(a: np.typing.NDArray[np.float32], b: np.typing.NDArray[np.float32], /):
    """Calculates the similarity between two view matrices using the Frobenius norm.

    Args:
      a: A numpy array representing the first view matrix.
      b: A numpy array representing the second view matrix.

    Returns:
      A float representing the similarity between the two view matrices, where higher values indicate greater similarity.
    """
    difference = a - b
    frobenius_norm = np.linalg.norm(difference, 'fro')
    similarity_score = 1.0 - frobenius_norm / np.linalg.norm(a, 'fro')
    return similarity_score


def argmax_similarity(target: np.typing.NDArray[np.float32], choices: list[np.typing.NDArray[np.float32]], /) -> int:
    """Finds the index of the element in choices that is most similar to the target.

    Args:
      target: A numpy array representing the target view matrix.
      choices: A list of numpy arrays representing the candidate view matrices.

    Returns:
      The index of the most similar view matrix in choices.
    """
    similarities = [calculate_similarity(target, choice) for choice in choices]
    return np.argmax(similarities).item()


def argmax_temporal_similarity(target_pose: np.typing.NDArray[np.float32], target_t: float,
                               choices: list[tuple[np.typing.NDArray[np.float32], float]],
                               prefer_time: bool = True) -> int:
    """Finds the index of the element in choices that is most similar to the target.

    Args:
      target_pose: A numpy array representing the target view matrix.
      target_t: A float representing the timestamp of the target view matrix.
      choices: A list of numpy arrays representing the candidate view matrices.
      prefer_time: If True, prioritize temporal proximity over pose similarity. If False, prioritize pose.

    Returns:
      The index of the most similar view matrix / timestamp in choices, prioritized according to prefer_time.
    """
    if prefer_time:
        time_differences = np.array([abs(target_t - choice_t) for _, choice_t in choices])
        time_argmin = np.argwhere(time_differences == np.min(time_differences)).flatten()
        choice_poses = np.array([choice_pose for choice_pose, _ in choices])[time_argmin]
        pose_similarities = [calculate_similarity(target_pose, choice) for choice in choice_poses]
        pose_index = np.argmax(pose_similarities).item()
        return time_argmin[pose_index]
    else:
        pose_similarities = np.array([calculate_similarity(target_pose, choice_pose) for choice_pose, _ in choices])
        pose_argmax = np.argwhere(pose_similarities == np.max(pose_similarities)).flatten()
        time_differences = np.array([abs(target_t - choice_t) for _, choice_t in choices])[pose_argmax]
        time_index = np.argmin(time_differences).item()
        return pose_argmax[time_index]


def parse_mat4(tensor_str: str) -> np.ndarray:
    """Parses a string representation of a 4x4 array into a numpy array. Entries are
    expected to be floats separated by whitespace and/or commas, with optional square brackets.
    """
    tensor_str = tensor_str.replace('[', ' ').replace(']', ' ').replace(',', ' ').strip()
    values = [float(x) for x in tensor_str.split() if x]
    if len(values) != 16:
        raise ValueError('Expected 16 values for a 4x4 matrix.')
    return np.array(values, dtype=np.float32).reshape((4, 4))
