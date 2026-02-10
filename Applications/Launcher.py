"""Applications/Launcher.py: Implements the GUI launcher, providing configuration
   options to the user before launching the main GUI."""

import sys
from dataclasses import fields, Field
from pathlib import Path

from imgui_bundle import imgui, imgui_ctx, portable_file_dialogs

from ICGui.Backend import SDL3Window, FontManager, FontVariants
from ICGui.Components.Colors import apply_ui_color
from ICGui.Components.HelpIndicator import help_indicator
from ICGui.Components.LaunchSettingsInput import resettable_input
from ICGui.State import LaunchConfig
from ICGui.util.Enums import CallbackType


# pylint: disable = too-few-public-methods
class Launcher:
    """Launcher GUI class for configuring the main GUI before launching it"""
    DIRECTORY_SELECTOR_TIMEOUT = 1000 // 30  # ms, high values slow down rendering in the background while the dialog is open

    def __init__(self):
        self.config: 'LaunchConfig | None' = None

        # Initialize Window
        self.window_size: tuple[int, int] = (1280, 720)
        self.backend: SDL3Window = SDL3Window('NerfICG Launcher', self.window_size, clear=True, vsync=True)
        self.backend.register_callback(CallbackType.RESIZE, self._update_size)

        # Set up ImGUI
        self._ctx: imgui.internal.Context | None = None
        self.fonts: FontVariants | None = None
        self.setup_imgui()

        # Application state
        self._directory_selector: portable_file_dialogs.select_folder | None = None

    def setup_imgui(self):
        """Sets up the ImGui context"""
        self._ctx = imgui.create_context()
        imgui.get_io().display_size = self.window_size
        imgui.get_io().set_ini_filename(None)
        # TODO: Set font size based on display DPI
        self.fonts = FontManager.set_font('sans-serif', 18)

    def configure(self, config: 'LaunchConfig') -> 'LaunchConfig':
        """Runs the GUI loop as long as the backend reports that the window is open,
         allowing the user to interactively modify the LaunchConfig."""
        self.config = config

        with self.backend as backend:
            keep_open = True
            while keep_open:
                # Close the GUI if the window is closed
                if not backend.prepare_frame():
                    sys.exit(0)

                # Update in-memory references to the current GUI state
                # If the accept button is clicked, keep_open is set to False
                keep_open = self._render()
                if not keep_open:
                    keep_open = not config.valid  # If the config is not valid, reset keep_open to True

                # Render the frame
                backend.finalize_frame()

        if self._ctx is not None:
            imgui.destroy_context(self._ctx)

        return config

    def _render(self) -> bool:
        """Renders the ImGui components"""
        row_height = imgui.get_frame_height_with_spacing()

        # Set up screen filling window
        flags = (imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_move
                 | imgui.WindowFlags_.no_collapse | imgui.WindowFlags_.no_saved_settings)
        imgui.set_next_window_size((self.window_size[0] - 8, self.window_size[1] - 8))
        imgui.set_next_window_pos((4, 4))

        with imgui_ctx.begin('Config', flags=flags):
            # Render main configuration section with optional scroll bars, leaving space for the footer
            with imgui_ctx.begin_child('ScrollingRegion', imgui.ImVec2(0.0, -row_height),
                                       child_flags=imgui.ChildFlags_.borders):
                self._render_directory_selector()
                self._render_gui_settings()

            # Render footer with accept button
            keep_open = self._render_footer()

        imgui.render()
        imgui.end_frame()

        return keep_open

    def _render_footer(self):
        """Renders the footer with the accept button and guess missing path button"""
        # "Guess missing path" button is only available in non-training mode
        num_buttons = 1 if self.config.is_training else 2

        # Calculate spacing to center the button(s)
        avail = imgui.get_content_region_avail().x
        width = imgui.get_style().item_spacing.x * (num_buttons - 1) \
                + imgui.get_style().frame_padding.x * 2 * num_buttons \
                + (imgui.calc_text_size('Guess Missing Path').x if not self.config.is_training else 0) \
                + imgui.calc_text_size('Launch').x
        imgui.set_cursor_pos_x((avail - width) / 2)

        with apply_ui_color('ACCEPT'):
            # Render the Guess Missing Path button if not in training mode
            if not self.config.is_training:
                if imgui.button('Guess Missing Path'):
                    self.config.guess_missing_paths()
                if imgui.is_item_hovered():
                    imgui.begin_tooltip()
                    imgui.text('Tries to automatically fill in checkpoint / config '
                               'path if the other is given.')
                    imgui.end_tooltip()

                imgui.same_line()

            if imgui.button('Launch'):
                return False
            if imgui.is_item_hovered():
                imgui.begin_tooltip()
                imgui.text('Save settings and launch the viewer...')
                imgui.end_tooltip()

        return True

    def _render_directory_selector(self):
        """Renders the model directory selector (button + dialog)"""
        if self.config.is_training:
            return

        if imgui.button('Select trained model directory...'):
            # Open directory selector
            default_path = Path('./output')
            if not default_path.exists():
                default_path = Path.cwd()

            self._directory_selector = portable_file_dialogs.select_folder(
                title='Select model directory...',
                default_path=str(default_path.absolute()),
            )
        help_indicator('Automatically fills in config and checkpoint path inputs '
                       'if a valid output directory is selected.')

        # Check directory selector status
        if self._directory_selector is None:
            return
        if not self._directory_selector.ready(self.DIRECTORY_SELECTOR_TIMEOUT):
            return

        result = Path(self._directory_selector.result()).resolve()
        if result.is_dir():
            self.config.infer_from_model_dir(result)

    def _render_gui_settings(self):
        """Renders the GUI settings section (resolution, font, etc.)"""
        cfg_fields: dict[str, Field] = {f.name: f for f in fields(self.config)}
        for field_name, field_info in cfg_fields.items():
            if field_name.startswith('_'):
                continue

            metadata = field_info.metadata
            if metadata is None or metadata == {}:
                continue
            if metadata.get('ui_disabled', False):
                continue
            if metadata.get('training_disabled', False) and self.config.is_training:
                continue

            try:
                new_val = resettable_input(getattr(self.config, field_name), field_info,
                                           locked=metadata.get('training_locked', False) and self.config.is_training)
                if new_val is not None:
                    setattr(self.config, field_name, new_val)
            except NotImplementedError:
                # If the field type is not implemented, skip rendering it
                continue

            # Render a help tooltip if available
            if (help_tooltip := metadata.get('help_tooltip', None)) is not None:
                help_indicator(help_tooltip)

    def _update_size(self, width: int, height: int):
        """Resize callback used to update the width and height in the backend"""
        self.window_size = (width, height)
