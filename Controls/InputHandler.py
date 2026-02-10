"""Controls/InputHandler.py: Handles mouse and keyboard inputs for GUI controls and contains a Camera object
defining the control scheme."""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field, InitVar
from typing import Callable, ClassVar, TypeAlias

import numpy as np
from imgui_bundle import imgui

from ICGui.Backend import SDL3Window
from ICGui.util.Enums import Action, Modifier

from .Camera import BaseControls, OrbitalControls, WalkingControls, FlyingControls
from .utils import InputCallback, DEFAULT_KEYMAP, MODIFIER_CODES


_Predicate: TypeAlias = Callable[[], bool] | None


@dataclass(slots=True)
class InputHandler:
    """Handles mouse and keyboard inputs for GUI controls, containing a Camera object defining the control scheme."""
    backend_window: SDL3Window = field(repr=False)

    # Camera Parameters, passed to camera constructor
    initial_c2w: InitVar[np.ndarray]
    camera_invert_x: InitVar[bool] = field(kw_only=True, default=False)
    camera_invert_y: InitVar[bool] = field(kw_only=True, default=False)
    camera_travel_speed: InitVar[float] = field(kw_only=True, default=1.0)
    camera_rotation_speed: InitVar[float] = field(kw_only=True, default=0.05)
    camera_zoom_speed: InitVar[float] = field(kw_only=True, default=0.25)

    control_scheme: BaseControls = field(init=False, repr=False)
    _controls_idx: int = field(init=False, repr=False, default=0)
    _CONTROL_TYPES: ClassVar[tuple[str, type[BaseControls]]] = [
        ('Orbital', OrbitalControls),
        ('Walking', WalkingControls),
        ('Flying', FlyingControls),
    ]

    # Input handling
    _action_descriptions: dict[Action | str, str] = field(init=False, default_factory=dict, repr=False)
    _input_callbacks: dict[Action | str, InputCallback] = field(init=False, default_factory=dict, repr=False)
    _keymap: OrderedDict[Modifier, OrderedDict[imgui.Key, set[Action | str]]] = field(init=False, default_factory=lambda: deepcopy(DEFAULT_KEYMAP), repr=False)
    _key_state: dict[Action | str, bool] = field(init=False, default_factory=dict, repr=False)
    _keys_additional: OrderedDict[Modifier, OrderedDict[str, Action]] = field(init=False, default_factory=OrderedDict, repr=False)
    _registered_actions: list[tuple[Action | str, _Predicate]] = field(init=False, default_factory=list, repr=False)

    _mouse_captured: bool = field(init=False, default=False, repr=False)
    _is_mouse_rotating: bool = field(init=False, default=False, repr=False)
    _is_mouse_panning: bool = field(init=False, default=False, repr=False)

    # Cache
    _hotkeys_cache: OrderedDict[Action, tuple[str, list[tuple[Modifier, imgui.Key | str, bool]]]] | None = field(
        init=False, default=None, repr=False)

    def __post_init__(self, initial_c2w, camera_invert_x, camera_invert_y, camera_travel_speed,
                      camera_rotation_speed, camera_zoom_speed) -> None:
        # Order of keybind registration determine the order in the GUI
        self.register_fake_keybind(Action.ROTATE_VIEW, 'Left Mouse Button')
        self.register_fake_keybind(Action.PAN_VIEW, 'Right Mouse Button', predicate=lambda: self.control_scheme.supports_panning)
        self.register_fake_keybind(Action.ZOOM, 'Mouse Wheel')
        self.register_fake_keybind(Action.CAPTURE_MOUSE, 'Double Click Left Mouse Button', predicate=lambda: self.control_scheme.supports_capture)
        self.control_scheme = self._CONTROL_TYPES[self._controls_idx][1](
            initial_c2w,
            travel_speed=camera_travel_speed,
            rotation_speed=camera_rotation_speed,
            zoom_speed=camera_zoom_speed,
            invert_rotation_x=camera_invert_x,
            invert_rotation_y=camera_invert_y,
            register_callback=self.register_callback,
            unregister_callback=self.unregister_callback,
        )

        self.register_callback(Action.CYCLE_CAMERA_CONTROLS, InputCallback(self.cycle_controls, continuous=False, interrupt_animation=True), 'Cycle Camera Controls')
        leave_fullscreen = lambda: self.backend_window.is_fullscreen and self.backend_window.toggle_fullscreen()
        self.register_callback(Action.EXIT_FULLSCREEN, InputCallback(leave_fullscreen, continuous=False), 'Exit Full Screen')
        self.register_callback(Action.CANCEL_MOUSE_CAPTURE, None, 'Cancel Mouse Capture')  # Handled by _handle_mouse_inputs
        self.register_callback(Action.SCREENSHOT, None, 'Screenshot')  # Handled by main application window
        self.register_callback(Action.FULLSCREEN, InputCallback(self.backend_window.toggle_fullscreen, continuous=False), 'Toggle Fullscreen')

    def register_callback(self, action: Action | str, callback: InputCallback | None, description: str):
        if not any(action == reg_action for reg_action, _ in self._registered_actions):
            self._registered_actions.append((action, None))
        self._action_descriptions[action] = description
        if callback is not None:
            self._input_callbacks[action] = callback

    def unregister_callback(self, action: Action | str):
        for i, (reg_action, _) in enumerate(self._registered_actions):
            if reg_action == action:
                del self._registered_actions[i]
                break
        else:
            return

        if action in self._action_descriptions:
            del self._action_descriptions[action]
        if action in self._input_callbacks:
            del self._input_callbacks[action]

    def register_fake_keybind(self, action: Action | str, key: str, modifier: Modifier = Modifier.NONE, *,
                              description: str | None = None, predicate: _Predicate = None):
        """Registers a fake keybind for an action, which is not actually bound to a key. Useful when controls are handled elsewhere."""
        if modifier not in self._keys_additional:
            self._keys_additional[modifier] = OrderedDict()
        self._keys_additional[modifier][key] = action
        self._registered_actions.append((action, predicate))

        # Register Description
        if description is None:
            if isinstance(action, Action):
                description = action.name.replace('_', ' ').title()
            else:
                description = action.title()
        self._action_descriptions[action] = description

    def cycle_controls(self):
        self.choose_camera(self._controls_idx + 1)

    def choose_camera(self, idx: int):
        """Chooses the controls to use."""
        self._controls_idx = idx % len(self._CONTROL_TYPES)
        self.control_scheme = self._CONTROL_TYPES[self._controls_idx][1](
            self.control_scheme.c2w,
            travel_speed=self.control_scheme.travel_speed,
            travel_speed_scale=self.control_scheme.travel_speed_scale,
            rotation_speed=self.control_scheme.rotation_speed,
            zoom_speed=self.control_scheme.zoom_speed,
            invert_rotation_x=self.control_scheme.invert_rotation_x,
            invert_rotation_y=self.control_scheme.invert_rotation_y,
            invert_panning_x=self.control_scheme.invert_panning_x,
            invert_panning_y=self.control_scheme.invert_panning_y,
            register_callback=self.register_callback,
            unregister_callback=self.unregister_callback,
        )

        # Reset mouse capture if necessary
        if not self.control_scheme.supports_capture and self._mouse_captured:
            self.backend_window.set_mouse_capture(False)
            self._mouse_captured = False
            self._is_mouse_rotating = False
            self._is_mouse_panning = False

        # Reset hotkeys cache
        self._hotkeys_cache = None

    def is_pressed(self, action: Action | str) -> bool:
        """Checks if the given action is currently pressed."""
        return self._key_state.get(action, False)

    def handle_inputs(self, delta_time: float):
        """Detects mouse and keyboard inputs and updates the camera accordingly. Also handles easing."""
        self._handle_keyboard_inputs(delta_time)
        self._handle_mouse_inputs()
        self.control_scheme.run_animation(delta_time)

    def _handle_keyboard_inputs(self, delta_time: float):
        # Discard keyboard inputs if imgui is already capturing them
        if imgui.get_io().want_capture_keyboard:
            return
        if imgui.is_any_item_focused():
            return

        # Reset all key states
        for action, _ in self._registered_actions:
            self._key_state[action] = False

        keys_captured = set()
        actions_performed = set()
        for modifier in self._keymap:
            for key, actions in self._keymap[modifier].items():
                # Check if the key has been pressed with a higher-precedence modifier
                if key in keys_captured:
                    continue

                for action in actions:
                    # Filter out actions that are not currently registered
                    if action not in self._key_state:
                        continue

                    # If the action has already been performed this frame, skip it
                    if action in actions_performed:
                        continue

                    # Check if there is an associated callback for this action
                    if action not in self._input_callbacks:
                        # If not, just save the key state
                        self._key_state[action] |= imgui.is_key_pressed(key)
                        continue

                    # Check if the key is pressed or down
                    check_key = imgui.is_key_down if self._input_callbacks[action].continuous else imgui.is_key_chord_pressed
                    mod_key = key | MODIFIER_CODES[modifier]
                    self._key_state[action] |= check_key(mod_key)
                    if self._key_state[action]:
                        keys_captured.add(key)
                        actions_performed.add(action)
                        callback = self._input_callbacks[action]
                        if callback.continuous:
                            callback.callback(delta_time)
                        else:
                            callback.callback()

                        if callback.interrupt_animation:
                            self.control_scheme.stop_animation()

    def _handle_mouse_inputs(self):
        # Double click for mouse capture (as rotation)
        if (self.control_scheme.supports_capture and imgui.is_mouse_double_clicked(0) and
                (not imgui.get_io().want_capture_mouse or self.backend_window.mouse_captured)):
            self._mouse_captured = not self._mouse_captured
            self._is_mouse_rotating = self._mouse_captured
            self._is_mouse_panning = False
            self.backend_window.set_mouse_capture(self._mouse_captured)
            self.control_scheme.stop_animation()
            if self._mouse_captured:
                self.control_scheme.start_rotation()

        # Cancel mouse capture with ESC
        if self.is_pressed(Action.CANCEL_MOUSE_CAPTURE):
            self._mouse_captured = False
            self._is_mouse_rotating = False
            self._is_mouse_panning = False
            self.backend_window.set_mouse_capture(False)
            self.control_scheme.stop_animation()

        # Handle mouse rotation
        if imgui.is_mouse_released(imgui.MouseButton_.left):
            if not self._mouse_captured:
                self.backend_window.set_mouse_capture(False)
                self._is_mouse_rotating = False
                self.control_scheme.stop_rotation()
        if imgui.is_mouse_clicked(imgui.MouseButton_.left) and not imgui.get_io().want_capture_mouse:
            if not self._mouse_captured and not self._is_mouse_panning:
                self.backend_window.set_mouse_capture(True)
                self._is_mouse_rotating = True
                self.control_scheme.start_rotation()
            self.control_scheme.stop_animation()
        if self._is_mouse_rotating:
            mouse_delta = self.backend_window.mouse_delta
            self.control_scheme.rotate(mouse_delta)
            self.control_scheme.stop_animation()

        # Handle mouse panning
        if imgui.is_mouse_released(imgui.MouseButton_.right):
            if not self._mouse_captured:
                self.backend_window.set_mouse_capture(False)
                self._is_mouse_panning = False
                self.control_scheme.stop_panning()
            self.control_scheme.stop_animation()
        if imgui.is_mouse_clicked(imgui.MouseButton_.right) and not imgui.get_io().want_capture_mouse:
            if not self._mouse_captured and not self._is_mouse_rotating and self.control_scheme.supports_panning:
                self.backend_window.set_mouse_capture(True)
                self._is_mouse_panning = True
                self.control_scheme.start_panning()
            self.control_scheme.stop_animation()
        if self._is_mouse_panning:
            mouse_delta = self.backend_window.mouse_delta
            self.control_scheme.pan(mouse_delta)
            self.control_scheme.stop_animation()

        # Handle mouse wheel zoom
        if imgui.get_io().mouse_wheel != 0 and not imgui.get_io().want_capture_mouse:
            self.control_scheme.zoom(imgui.get_io().mouse_wheel)
            self.control_scheme.stop_animation()

    @property
    def camera_controls(self) -> list[str]:
        """Returns an iterator over the available camera controls."""
        return [self._CONTROL_TYPES[i][0] for i in range(len(self._CONTROL_TYPES))]

    @property
    def current_camera_controls_idx(self) -> int:
        """Returns the index of the current camera controls."""
        return self._controls_idx

    @property
    def hotkeys(self) -> OrderedDict[Action, tuple[str, list[tuple[Modifier, imgui.Key | str, bool]]]]:
        """Returns a list of all registered hotkeys for the input handler.
        Returns:
            OrderedDict: Action -> (Description, List of (Modifier, Key, IsEditable))
        """
        if self._hotkeys_cache is None:
            hotkeys = OrderedDict()

            for action, pred in self._registered_actions:
                if pred is not None and not pred():
                    continue
                action_name = action.name if isinstance(action, Action) else action
                description = self._action_descriptions.get(action, action_name.replace('_', ' ').title())
                hotkeys[action] = (description, [])

            for modifier, keys in self._keymap.items():
                for key, actions in keys.items():
                    for action in actions:
                        if action not in hotkeys:
                            continue
                        hotkeys[action][1].append((modifier, key, True))

            for modifier, keys in self._keys_additional.items():
                for key, action in keys.items():
                    if action not in hotkeys:
                        continue
                    hotkeys[action][1].append((modifier, key, False))

            self._hotkeys_cache = hotkeys
        return self._hotkeys_cache
