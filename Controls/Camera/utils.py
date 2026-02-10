"""Controls/Camera/utils.py: Animation and quaternion utilities for the camera classes."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

import numpy as np
import quaternion as quat


_Interpolatable = TypeVar("_Interpolatable")
_InterpolationFunction = Callable[[_Interpolatable, _Interpolatable, float], _Interpolatable]


def apply_quaternion(q: quat.quaternion, v: np.ndarray) -> np.ndarray:
    """
    Applies a quaternion rotation to a vector. See https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

    Args:
        q (quat.quaternion): The quaternion representing the rotation.
        v (np.ndarray): The vector to be rotated, must be a 3-element numpy array.

    Returns:
        np.ndarray: The rotated vector.
    """
    rotated = v + 2 * np.cross(q.vec, np.cross(q.vec, v) + q.w * v)
    return rotated


def swing_twist_decomposition(q: quat.quaternion, d: np.ndarray) -> tuple[quat.quaternion, quat.quaternion]:
    """Decomposes a quaternion into swing and twist components.
    Params:
        q: The quaternion to decompose.
        d: The direction vector along which to decompose the quaternion.
    Returns:
        A tuple containing the swing and twist components as quaternions.
        Twist: The rotation around the direction vector `d`.
        Swing: The rotation perpendicular to the direction vector `d`.
        Rotation can be composed back as q = swing * twist.
    """
    projection = np.dot(q.vec, d) * d
    twist = quat.quaternion(q.w, *projection).normalized()
    swing = q * twist.conjugate()
    return swing, twist


@dataclass
class Animation(ABC):
    """Abstract base class for all animations."""
    t: float = field(default=0.0, init=False)
    finished: bool = field(default=False, init=False)

    @abstractmethod
    def run(self, obj: object, delta_time: float):
        """
        Runs the animation, updating the object's attributes.

        Args:
            obj (object): The object whose attributes are being animated.
            delta_time (float): The time elapsed since the last update.
        """
        pass

    @abstractmethod
    def finish(self, obj: object):
        """Immediately finishes the animation, setting the object's attributes to their final values."""
        pass

    @staticmethod
    def lerp(from_value: _Interpolatable, to_value: _Interpolatable, t) -> _Interpolatable:
        """Linear interpolation between two values (that support addition and multiplication)."""
        return from_value * (1 - t) + to_value * t

    @staticmethod
    def slerp(from_value: quat.quaternion, to_value: quat.quaternion, t) -> quat.quaternion:
        """Spherical linear interpolation between two quaternions or vectors."""
        # noinspection PyUnresolvedReferences
        return np.slerp_vectorized(from_value, to_value, t)  # (provided by numpy-quaternion)


@dataclass
class EasingAnimation(Animation):
    """Class to handle easing animations for camera transformations."""
    from_attributes: dict[str, Any]
    to_attributes: dict[str, Any]
    interpolation: dict[str, _InterpolationFunction]
    duration: float = 1.0
    easing_function: Callable[[float], float] = \
        lambda x: (4 * x * x * x) if x < 0.5 else (1 - math.pow(-2 * x + 2, 3) / 2)

    def run(self, obj: object, delta_time: float):
        """
        Runs the easing animation, updating the object's attributes.

        Args:
            obj (object): The object whose attributes are being animated.
            delta_time (float): The time elapsed since the last update.
        """
        if self.finished:
            return

        self.t += delta_time / self.duration
        if self.t >= 1.0:
            self.t = 1.0
            self.finished = True

        for attr, from_value in self.from_attributes.items():
            to_value = self.to_attributes[attr]
            interpolation_function = self.interpolation.get(attr, self.lerp)
            # noinspection PyArgumentList
            t = self.easing_function(self.t)
            value = interpolation_function(from_value, to_value, t)
            setattr(obj, attr, value)

    def finish(self, obj: object):
        """Immediately finishes the animation, setting the object's attributes to their final values."""
        self.t = 1.0
        self.finished = True
        for attr, to_value in self.to_attributes.items():
            setattr(obj, attr, to_value)


@dataclass
class KeyframeAnimation(Animation):
    """Class to handle keyframe animations for camera transformations."""
    keyframes: list[dict[str, Any]]
    interpolation: dict[str, _InterpolationFunction]
    duration: float = 1.0
    loop: bool = False

    def run(self, obj: object, delta_time: float):
        """
        Runs the keyframe animation, updating the object's attributes.

        Args:
            obj (object): The object whose attributes are being animated.
            delta_time (float): The time elapsed since the last update.
        """
        if self.finished:
            return

        self.t += delta_time / self.duration
        if self.t >= 1.0:
            if self.loop:
                self.t = self.t % 1.0
            else:
                self.t = 1.0
                self.finished = True

        num_keyframes = len(self.keyframes)
        if num_keyframes < 1:
            return
        if num_keyframes == 1:
            for attr, value in self.keyframes[0].items():
                setattr(obj, attr, value)
            return

        total_segments = num_keyframes - 1
        segment = min(int(self.t * total_segments), total_segments - 1)
        segment_t = (self.t - (segment / total_segments)) * total_segments
        from_kf = self.keyframes[segment]
        to_kf = self.keyframes[segment + 1]
        for attr, from_value in from_kf.items():
            to_value = to_kf[attr]
            interpolation_function = self.interpolation.get(attr, self.lerp)
            # noinspection PyArgumentList
            value = interpolation_function(from_value, to_value, segment_t)
            setattr(obj, attr, value)

    def finish(self, obj: object):
        """Immediately finishes the animation, setting the object's attributes to their final values."""
        self.t = 1.0
        self.finished = True
        if len(self.keyframes) > 0:
            for attr, to_value in self.keyframes[-1].items():
                setattr(obj, attr, to_value)
