"""Components/Windows/ModalWindow.py: Popup modals to show a message + user choices which can be queried once closed."""

from dataclasses import dataclass, field, InitVar

from imgui_bundle import imgui, hello_imgui

from ICGui.Components.Colors import apply_ui_color


@dataclass
class ModalWindow:
    """Modal window with configurable title, message and choice buttons for the user."""
    title: str
    message: str
    options: list[str] = field(default_factory=lambda: ['OK', 'Cancel'])
    option_colors: list[str] = field(default_factory=lambda: ['WARNING', 'DEFAULT'])
    vertical_options: bool = False
    close_action: int | None = None

    start_open: InitVar[bool] = True
    result: int = field(init=False, default=None)

    _absolute_paths: bool = False

    def __post_init__(self, start_open: bool):
        if start_open:
            self.open()

    def render(self):
        """Renders the modal window component."""
        if not self.is_open:
            return

        # Render the popup, show close button if close_action is set
        imgui.set_next_window_pos(imgui.get_main_viewport().get_center(), imgui.Cond_.none, (0.5, 0.5))
        shown, stay_open = imgui.begin_popup_modal(
            self.title,
            p_open=True if self.close_action is not None else None,
            flags=imgui.WindowFlags_.always_auto_resize
        )
        if not shown:
            return

        # Render the popup content
        imgui.push_text_wrap_pos(hello_imgui.em_to_vec2(24, 0).x)
        imgui.text_wrapped(self.message)
        imgui.pop_text_wrap_pos()

        imgui.spacing()

        # Show user options as buttons
        for (idx, option), theme in zip(enumerate(self.options), self.option_colors):
            with apply_ui_color(theme):
                if self.vertical_options:
                    if imgui.button(option, (-1, 0)):
                        self.result = idx
                        imgui.close_current_popup()
                else:
                    if imgui.button(option):
                        self.result = idx
                        imgui.close_current_popup()
                    imgui.same_line()

        # stay_open is None when no close_action is set, otherwise True/False to indicate close button press
        if stay_open is False:
            self.result = self.close_action
            imgui.close_current_popup()

        imgui.end_popup()

    def open(self):
        """Opens the modal window."""
        imgui.open_popup(self.title)

    @property
    def is_open(self) -> bool:
        return imgui.is_popup_open(self.title)
