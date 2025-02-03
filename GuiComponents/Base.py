# -- coding: utf-8 --

"""GuiComponents/Base.py: Abstract Base Class for GUI components that can be hidden and shown."""

import itertools
import math
from abc import ABC, abstractmethod
from typing import Collection, Callable

import imgui
from sdl2 import SDL_GetKeyName, SDL_Scancode, SDL_SCANCODE_TO_KEYCODE

from ICGui.GuiConfig import LaunchConfig


class GuiComponent(ABC):
    """Abstract Base Class for any GUI components, that can be hidden or shown using keyboard shortcuts."""
    windows = []
    registered_hotkeys: dict[str, Collection[str]] = {}
    registered_window_hotkeys: dict[str, Collection[str]] = {}
    registered_external_hotkeys: dict[str, Collection[str]] = {}
    save_window_positions: bool = True
    gui_config: LaunchConfig | None = None

    def __init__(self, name: str, hotkeys: Collection[SDL_Scancode] = None, *,
                 config: 'NeRFConfig | None' = None,
                 closable: bool = True, default_open: bool = True,
                 force_size: tuple[int, int] | None = None, force_center: bool = False, force_focus: bool = False,
                 window_flags: int | None = None,
    ):
        super().__init__()

        GuiComponent.windows.append(self)

        self._name = name
        self._closable: bool = closable
        self._open: bool = (GuiComponent.gui_config.gui_window_states.get(name, default_open)
                            if GuiComponent.gui_config else default_open)
        self._hotkeys: Collection[SDL_Scancode] | None = hotkeys
        self._hotkey_actions: dict[SDL_Scancode, tuple[Callable[[], None], bool]] = {}
        self._config = config

        self._force_size: tuple[int, int] | None = force_size
        self._force_center: bool = force_center
        self._force_focus: bool = force_focus

        self._set_window_position: tuple[int, int] | None = None
        self._set_window_size: tuple[int, int] | None = None

        self._window_flags: int = window_flags or 0

        if hotkeys is not None and len(hotkeys) > 0:
            GuiComponent.registered_window_hotkeys[f'Show {name} Window'] = \
                [SDL_GetKeyName(SDL_SCANCODE_TO_KEYCODE(hotkey)).decode('ascii')
                 for hotkey in hotkeys]

    @staticmethod
    def registerExternalHotkey(name: str, hotkeys: Collection[SDL_Scancode | str]):
        """Registers an external hotkey under the given name."""
        GuiComponent.registered_external_hotkeys[name] = [
            hotkey if isinstance(hotkey, str)
            else SDL_GetKeyName(SDL_SCANCODE_TO_KEYCODE(hotkey)).decode('utf-8')
            for hotkey in hotkeys
        ]

    def _registerHotkey(self, name: str, action: Callable[[], None], hotkeys: Collection[SDL_Scancode],
                        repeat: bool = False):
        """Registers a hotkey for the component."""
        GuiComponent.registered_hotkeys[name] = \
            [SDL_GetKeyName(SDL_SCANCODE_TO_KEYCODE(hotkey)).decode('ascii')
             for hotkey in hotkeys]
        self._hotkey_actions.update({hotkey: (action, repeat) for hotkey in hotkeys})

    @staticmethod
    def tileWindows():
        """Tiles all open windows."""
        # pylint: disable=invalid-name
        PADDING = 8  # Padding between windows in pixels

        # pylint: disable=protected-access
        open_windows = [window for window in GuiComponent.windows if window._open and not window._force_center]
        window_count = len(open_windows)

        if window_count == 0:
            return

        grid_width = math.ceil(window_count ** 0.5)
        grid_height = math.ceil(window_count / grid_width)

        total_width, total_height = imgui.get_main_viewport().work_size
        total_width -= PADDING  # One outer padding (the other is accounted for by the windows themselves)
        total_height -= PADDING  # One outer padding (the other is accounted for by the windows themselves)

        window_width = math.floor(total_width / grid_width) - PADDING
        window_height = math.floor(total_height / grid_height) - PADDING

        for (x, y), window in zip(itertools.product(range(grid_width), range(grid_height)), open_windows):
            window._set_window_position = (x * (window_width + PADDING) + PADDING,
                                           y * (window_height + PADDING) + PADDING)
            window._set_window_size = (window_width, window_height)

    def _handleOpenHotkeys(self):
        if self._hotkeys is not None and not imgui.get_io().want_capture_keyboard and not imgui.is_any_item_focused():
            for hotkey in self._hotkeys:
                # noinspection PyArgumentList
                # (Incorrect annotation in pyimgui bindings)
                if imgui.is_key_pressed(hotkey, repeat=False):
                    self._setOpenState(not self._open)

    def _handleCustomHotkeys(self):
        # Ignore hotkeys if imgui is capturing keyboard input or an item is focused
        if imgui.get_io().want_capture_keyboard or imgui.is_any_item_focused():
            return

        for hotkey, (action, repeat) in self._hotkey_actions.items():
            if imgui.is_key_pressed(hotkey, repeat=repeat):
                action()

    def render(self, **kwargs):
        """Calls the _render function if open and handles hotkey interactions."""
        self._handleOpenHotkeys()

        # If not open, still handle custom hotkeys
        if not self._open:
            self._handleCustomHotkeys()
            return

        # Handle special window attributes
        if self._force_focus:
            imgui.set_next_window_focus()

        if self._force_size is not None:
            imgui.set_next_window_size(*self._force_size)
        elif self._set_window_size is not None:
            imgui.set_next_window_size(*self._set_window_size)
            self._set_window_size = None

        if self._force_center:
            imgui.set_next_window_position(*imgui.get_main_viewport().get_center(), pivot_x=0.5, pivot_y=0.5)
        elif self._set_window_position is not None:
            imgui.set_next_window_position(*self._set_window_position)
            self._set_window_position = None

        # Render the window
        window_flags = self._window_flags
        if not GuiComponent.save_window_positions:
            window_flags |= imgui.WINDOW_NO_SAVED_SETTINGS
        with imgui.begin(self._name, closable=self._closable, flags=window_flags) as window_state:
            # Need to check for is_window_collapsed since window_state[0] is wrong upon program reload
            if window_state[0] and not imgui.is_window_collapsed():
                self._render(**kwargs)
        if not window_state[1]:
            self._setOpenState(False)

        self._handleCustomHotkeys()

    def _setOpenState(self, open_state: bool):
        """Sets the open state of the window."""
        self._open = open_state

        if GuiComponent.gui_config:
            GuiComponent.gui_config.gui_window_states[self._name] = open_state
            GuiComponent.gui_config.save()

    def close(self):
        """Closes the GUI component."""
        self._setOpenState(False)

    def open(self):
        """Opens the GUI component."""
        self._setOpenState(True)

    @abstractmethod
    def _render(self, **kwargs):
        """Renders the GUI component."""
        pass
