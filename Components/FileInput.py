"""Components/FileInput.py: File path input component with text input for path and browse input to select
a path with the native file selector."""

from functools import partial
from pathlib import Path
from typing import Collection

from imgui_bundle import imgui, portable_file_dialogs, icons_fontawesome_6

from ICGui.util import AutoComplete


_FILE_SELECTOR_TIMEOUT = 1000 // 60  # ms, high values slow down rendering in the background while the dialog is open
_file_dialogs: dict[str, portable_file_dialogs.open_file] = {}


def browse_files_button(label: str, value: str, extensions: Collection[str] | None) -> tuple[bool, str]:
    """Renders a button to open a file selector dialog and returns on each render,
    if the user confirmed their choice, along with the chosen path."""
    global _file_dialogs

    # Render button
    open_dialog = imgui.button(f'{icons_fontawesome_6.ICON_FA_FOLDER_OPEN}##{label}')
    if open_dialog:
        # Open file selector
        filters = [] if extensions is None else \
            [f'{label} Files'] + [f'*{ext}' for ext in extensions if ext.startswith('.')]
        filters.append('All Files')
        filters.append('*')
        _file_dialogs[label] = portable_file_dialogs.open_file(
            title=f'Select {label}...',
            default_path=str(Path(value).absolute()),
            filters=filters
        )
    if imgui.is_item_hovered():
        imgui.begin_tooltip()
        imgui.text('Browse files...')
        imgui.end_tooltip()

    # Check file selector status
    file_dialog = _file_dialogs.get(label, None)
    if file_dialog is None:
        return False, value
    if not file_dialog.ready(_FILE_SELECTOR_TIMEOUT):
        return False, value

    # Get selected file
    results = file_dialog.result()
    del _file_dialogs[label]  # Clear the dialog to prevent reusing it
    if len(results) != 1:
        return False, value

    return True, results[0]


def file_input(label: str, value: Path | None, extensions: Collection[str] = None):
    """Renders a file input field with a text input for the path and a button to open a file selector dialog."""
    in_value = str(value) if value is not None else ''

    # Note: Because the callback_completion handler does not respond to Shift+Tab, we use the callback_always flag.
    #       This allows us to check for the correct key / key combination ourselves. To prevent Shift+Tab from
    #       triggering tab navigation, we also need to set the input element as the owner of the tab key.
    flags = (imgui.InputTextFlags_.callback_always  # imgui.InputTextFlags_.callback_completion
             | imgui.InputTextFlags_.callback_edit
             | imgui.InputTextFlags_.callback_char_filter)
    callback = partial(_file_input_callback, label=label, extensions=extensions, value=value)
    modified, new_value = imgui.input_text(
        f'##inp_text_{label}', in_value,
        flags=flags, callback=callback,
    )
    imgui.set_item_key_owner(imgui.Key.tab)

    # Show file browser button
    imgui.same_line(spacing=imgui.get_style().item_inner_spacing.x)
    file_selected, new_value = browse_files_button(label, in_value, extensions)
    if file_selected:
        modified = True

        # Convert to relative path if inside the current working directory
        new_value_path = Path(new_value).absolute()
        if new_value_path.is_relative_to(Path.cwd()):
            new_value = str(new_value_path.relative_to(Path.cwd()))

    # Show label
    imgui.same_line()
    imgui.text(label)

    return modified, Path(new_value) if modified else value


def _file_input_callback(data: imgui.InputTextCallbackData, label: str, value: str, extensions: tuple[str]) -> int:
    """Auto-complete callback function for the file input field."""
    # Invalidate the current auto-complete options on any manual edit
    if data.event_flag & imgui.InputTextFlags_.callback_edit:
        AutoComplete.invalidate(label)
        return 0

    # The following handles situations, where multiple auto-complete options are available. Since tab just selects
    #  the next option, the user has to type another symbol to start autocompleting the next path component. Since
    #  a trailing slash is already present on any auto-completed directory, we ignore the next slash typed by the
    #  user. This way the user can "request" the next auto-complete option by typing a slash. This tries to emulate
    #  shell-like auto-completion behavior.
    if data.event_flag & imgui.InputTextFlags_.callback_char_filter:
        if data.event_char == ord('/') and AutoComplete.has(label):
            AutoComplete.invalidate(label)
            return 1
        return 0

    # Handle auto-completion, see note in file_input function
    if imgui.is_key_pressed(imgui.Key.tab):
        AutoComplete.path(label, value, extensions=extensions, event_data=data)

    return 0
