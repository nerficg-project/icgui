"""Controls/Camera/BaseControls.py: Base class for different camera navigation implementations"""

from typing import Callable, ClassVar, Self
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, InitVar

import numpy as np
import quaternion as quat

from Cameras.utils import invert_3d_affine
from ICGui.Controls.utils import InputCallback
from ICGui.State.Volatile import GlobalState
from ICGui.util.Enums import Action
from .utils import Animation, EasingAnimation


@dataclass
class BaseControls(ABC):
    """Abstract base class for all different camera navigation implementations."""
    initial_c2w: InitVar[np.ndarray] = None
    initial_w2c: InitVar[np.ndarray] = None
    register_callback: InitVar[Callable[[Action | str, InputCallback, str], None]] = field(kw_only=True, default=None)
    unregister_callback: InitVar[Callable[[Action | str], None]] = field(kw_only=True, default=None)
    position: np.ndarray = field(init=False)
    rotation: quat.quaternion = field(init=False, default=None)
    has_moved: bool = field(init=False, default=True)
    _animation: Animation | None = field(init=False, default=None, repr=False)

    # Direction vectors in world space
    _RIGHT: np.ndarray = field(init=False, repr=False, default_factory=lambda: np.array([1.0, 0.0, 0.0]))  # pylint: disable=invalid-name
    _UP: np.ndarray = field(init=False, repr=False, default_factory=lambda: np.array([0.0, 1.0, 0.0]))  # pylint: disable=invalid-name
    _BACKWARD: np.ndarray = field(init=False, repr=False, default_factory=lambda: np.array([0.0, 0.0, 1.0]))  # pylint: disable=invalid-name
    _model_transform: ClassVar[np.ndarray] = np.diag([1.0, -1.0, -1.0, 1.0])  # By default Colmap -> OpenGL transformation

    # Camera configuration parameters
    travel_speed: float = field(kw_only=True, default=1.0)
    travel_speed_scale: float = field(kw_only=True, default=1.0)
    rotation_speed: float = field(kw_only=True, default=0.1)
    zoom_speed: float = field(kw_only=True, default=0.25)

    invert_rotation_x: bool = field(kw_only=True, default=False)
    invert_rotation_y: bool = field(kw_only=True, default=False)
    invert_panning_x: bool = field(kw_only=True, default=False)
    invert_panning_y: bool = field(kw_only=True, default=False)
    supports_capture: ClassVar[bool] = False
    supports_panning: ClassVar[bool] = False
    _is_mouse_rotating: bool = field(init=False, default=False, repr=False)
    _mouse_captured: bool = field(init=False, default=False, repr=False)

    def __post_init__(self, initial_c2w, initial_w2c, register_callback, unregister_callback) -> None:
        if (initial_c2w is None) == (initial_w2c is None):
            raise ValueError('Either initial_c2w or initial_w2c must be provided, but not both.')
        if initial_c2w is not None:
            self.c2w = initial_c2w
        else:
            self.w2c = initial_w2c

        # Register controls
        if register_callback is not None:
            register_callback(Action.FORWARD, InputCallback(self.forward, interrupt_animation=True), 'Move Forward')
            register_callback(Action.BACKWARD, InputCallback(self.backward, interrupt_animation=True), 'Move Backward')
            register_callback(Action.LEFT, InputCallback(self.left, interrupt_animation=True), 'Move Left')
            register_callback(Action.RIGHT, InputCallback(self.right, interrupt_animation=True), 'Move Right')
            register_callback(Action.UP, InputCallback(self.up, interrupt_animation=True), 'Move Up')
            register_callback(Action.DOWN, InputCallback(self.down, interrupt_animation=True), 'Move Down')
            register_callback(Action.SPEED_UP, InputCallback(self.increase_travel_speed, continuous=False, interrupt_animation=True), 'Increase Speed')
            register_callback(Action.SPEED_DOWN, InputCallback(self.decrease_travel_speed, continuous=False, interrupt_animation=True), 'Decrease Speed')
            unregister_callback(Action.ANIMATE)

    def __init_subclass__(cls):
        """Enforce that either w2c or c2w is overridden by subclasses."""
        super().__init_subclass__()
        has_implementation = False
        for parent_class in cls.__mro__:
            if parent_class is BaseControls:
                break
            has_c2w_getter = isinstance(parent_class.__dict__.get('c2w', None), property)
            has_c2w_setter = has_c2w_getter and parent_class.__dict__['c2w'].fset is not None
            has_w2c_getter = isinstance(parent_class.__dict__.get('w2c', None), property)
            has_w2c_setter = has_w2c_getter and parent_class.__dict__['w2c'].fset is not None
            has_implementation |= ((has_c2w_getter and has_c2w_setter) or (has_w2c_getter and has_w2c_setter))

        if not has_implementation:
            raise TypeError(
                f"Class '{cls.__name__}' must implement at least one of "
                "'c2w' or 'w2c' with both getter and setter."
            )

    @property
    def w2c(self) -> np.ndarray:
        """Returns the world-to-camera matrix."""
        return invert_3d_affine(self.c2w)

    @w2c.setter
    def w2c(self, w2c: np.ndarray) -> None:
        """Sets the position and rotation from the given world-to-camera matrix."""
        self.c2w = invert_3d_affine(w2c)

    @property
    def c2w(self) -> np.ndarray:
        """Returns the camera-to-world matrix."""
        return invert_3d_affine(self.w2c)

    @c2w.setter
    def c2w(self, c2w: np.ndarray) -> None:
        """Sets the position and rotation from the given camera-to-world matrix."""
        self.w2c = invert_3d_affine(c2w)

    @property
    def model_transform(self) -> np.ndarray:
        """Returns the model transformation matrix to convert from the camera coordinate system to OpenGL."""
        return self._model_transform

    @abstractmethod
    def forward(self, dt: float):
        """Moves the camera forward."""
        raise NotImplementedError

    @abstractmethod
    def backward(self, dt: float):
        """Moves the camera backward."""
        raise NotImplementedError

    @abstractmethod
    def left(self, dt: float):
        """Moves the camera left."""
        raise NotImplementedError

    @abstractmethod
    def right(self, dt: float):
        """Moves the camera right."""
        raise NotImplementedError

    @abstractmethod
    def up(self, dt: float):
        """Moves the camera up."""
        raise NotImplementedError

    @abstractmethod
    def down(self, dt: float):
        """Moves the camera down."""
        raise NotImplementedError

    @abstractmethod
    def start_rotation(self):
        """Starts the rotation of the camera."""
        raise NotImplementedError

    @abstractmethod
    def rotate(self, mouse_delta: tuple[float, float]):
        """Rotates the camera."""
        raise NotImplementedError

    @abstractmethod
    def stop_rotation(self):
        """Stops the rotation of the camera."""
        raise NotImplementedError

    def start_panning(self):
        """Starts panning of the camera (if panning is supported)."""
        pass

    def pan(self, mouse_delta: tuple[float, float]):
        """Pans the camera (if panning is supported)."""
        pass

    def stop_panning(self):
        """Stops panning the camera (if panning is supported)."""
        pass

    @abstractmethod
    def zoom(self, amount: float):
        """Zooms the camera."""
        raise NotImplementedError

    def increase_travel_speed(self):
        """Increases the travel speed of the camera."""
        self.travel_speed_scale *= 1.1

    def decrease_travel_speed(self):
        """Decreases the travel speed of the camera."""
        self.travel_speed_scale *= 0.9

    def ease_to(self, *, to_w2c: np.ndarray = None, to_c2w: np.ndarray = None, duration: float = 1.0) -> Self:
        """Eases the camera to the given target camera."""
        # TODO: do this in the time dimension as well
        self.has_moved = False

        if GlobalState().skip_animations:
            if to_c2w is not None:
                self.c2w = to_c2w
            elif to_w2c is not None:
                self.w2c = to_w2c
            self._recalculate_directions()
            return self

        target_cam = type(self)(initial_c2w=to_c2w, initial_w2c=to_w2c)
        self._animation = EasingAnimation(
            duration=duration,
            from_attributes={
                'position': self.position,
                'rotation': self.rotation,
            },
            to_attributes={
                'position': target_cam.position,
                'rotation': target_cam.rotation,
            },
            interpolation={
                'rotation': EasingAnimation.slerp,
            }
        )
        return target_cam

    def stop_animation(self):
        self._animation = None

    def run_animation(self, delta_time: float):
        """Runs the current animation if it exists."""
        if self._animation is None:
            return

        if GlobalState().skip_animations:
            self._animation.finish(self)
            self._animation = None
            self._recalculate_directions()
            return

        # Update animation time
        self._animation.run(self, delta_time)
        if self._animation.finished:
            self._animation = None  # Reset animation
        self._recalculate_directions()

    def _recalculate_directions(self):
        pass