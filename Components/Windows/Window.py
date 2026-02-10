"""Components/Windows/Window.py: Base Class for ImGui Window components + Window management
(hiding, showing, tiling, etc.) handled by the WindowManager class."""

from dataclasses import dataclass, field
from typing import Any, ClassVar, TypeVar

from imgui_bundle import imgui, imgui_ctx

from ICGui.Components.Windows.ModalWindow import ModalWindow
from ICGui.Controls import InputCallback
from ICGui.State import PersistentState
from ICGui.State.Volatile import GlobalState
from ICGui.util.Enums import Action


@dataclass
class Window:
    """Empty window base class, that can be hidden or shown using keyboard shortcuts."""
    name: ClassVar[str] = 'Window'
    default_open: ClassVar[bool] = True

    closable: bool = True
    force_center: bool = False
    force_focus: bool = False
    force_size: tuple[int, int] | None = None
    persist_open: bool = True
    persist_position: bool = True
    flags: int = 0

    is_open: bool = field(init=False)

    def __post_init__(self):
        GlobalState().input_manager.register_callback(
            f'SHOW_{self.name.upper()}',
            InputCallback(self.toggle_open, continuous=False),
            f'Show {self.name} Window'
        )

        # Get initial open state (either from state manager or default)
        if self.persist_open:
            self.is_open = PersistentState().window_open.get(self.name, self.default_open)
        else:
            self.is_open = self.default_open

    def toggle_open(self):
        self.set_open(not self.is_open)

    def set_open(self, open_state: bool):
        self.is_open = open_state
        if self.persist_open:
            PersistentState().window_open[self.name] = open_state

    def render(self, **kwargs):
        """Renders a window with the specified window options and calls the _render function if open"""
        # Handle special window attributes
        if self.force_focus:
            imgui.set_next_window_focus()
        if self.force_size is not None:
            imgui.set_next_window_size(self.force_size)
        if self.force_center:
            imgui.set_next_window_pos(imgui.get_main_viewport().get_center(), pivot=(0.5, 0.5))

        # Set flags
        window_flags = self.flags
        if not self.persist_position:
            window_flags |= imgui.WindowFlags_.no_saved_settings

        # Render the window
        with imgui_ctx.begin(self.name, p_open=self.is_open if self.closable else None, flags=window_flags) as (expanded, is_open):
            if expanded:
                self._render(**kwargs)
        if not is_open and self.closable:
            self.set_open(False)

    def _render(self, **kwargs):
        """Renders the GUI component."""
        pass


T = TypeVar('T', bound=Window)


@dataclass(slots=True)
class WindowManager:
    save_window_positions: bool = True
    modals: dict[str, ModalWindow] = field(default_factory=dict)
    windows: list[Window] = field(default_factory=list)

    def __post_init__(self):
        pass  # FIXME: Re-enable once we have multiple windows again
        # GlobalState().input_manager.register_callback(
        #     Action.TILE_WINDOWS,
        #     InputCallback(self.tile_windows, continuous=False, interrupt_animation=False),
        #     'Tile Sub-Windows'
        # )

    def add_window(self, window_cls: type[T], *args: Any, **kwargs: Any) -> T:
        """Adds a new window to the manager."""
        window = window_cls(*args, **kwargs)
        self.windows.append(window)
        return window

    def add_modal_window(self, window: ModalWindow):
        self.modals[window.title] = window

    def close_modal_window(self, title: str):
        """Closes a modal window with the given title."""
        if title in self.modals:
            self.modals.pop(title)

    def get_modal_response(self, title: str) -> int | None:
        modal = self.modals.get(title)
        if modal is not None and not modal.is_open:
            self.modals.pop(title)
            return modal.result

        return None

    def is_modal_open(self, title: str) -> bool:
        """Checks if a modal window with the given title is currently open."""
        return title in self.modals and self.modals[title].is_open

    def tile_windows(self):
        """Tiles all open windows."""
        # Note: Not currently very useful, as only all settings have been combined into one window.
        #   We can re-add this, once window-splitting functionality is added. See the git history
        #   for the most-recent implementation.
        pass

    def render(self):
        for window in self.windows:
            if window.is_open:
                window.render()

        for modal in self.modals.values():
            modal.render()
