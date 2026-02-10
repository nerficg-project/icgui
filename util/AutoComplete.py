"""util/AutoComplete.py: Utilities for implementing auto complete for input fields."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from imgui_bundle import imgui

from ICGui.Backend import FontManager


@dataclass
class AutoCompleteOption:
    """Auto-complete option for an input field"""
    options: list[Any]
    target_idx: int
    _current_index: int = -1

    @property
    def next_choice(self):
        """Returns the next auto-complete choice"""
        if imgui.is_key_down(imgui.Key.mod_shift):
            if self._current_index == -1:
                self._current_index = 0
            self._current_index = (self._current_index - 1) % len(self.options)
        else:
            self._current_index = (self._current_index + 1) % len(self.options)
        choice = self.options[self._current_index]
        return choice


class AutoComplete:
    _completions: dict[str, AutoCompleteOption] = {}

    @classmethod
    def invalidate(cls, label: str):
        """Invalidates the auto-complete options for the given label"""
        cls._completions.pop(label, None)

    @classmethod
    def has(cls, label: str) -> bool:
        """Checks if there are any auto-complete options for the given label"""
        return label in cls._completions

    @classmethod
    def path(cls, label: str, value: str, extensions: tuple[str], event_data: imgui.InputTextCallbackData):
        """Callback used to auto-complete file paths"""
        if not cls.has(label):
            # Split path into directory and glob pattern
            last_slash = value.rfind('/') + 1
            path = Path(value[:last_slash])
            glob = value[last_slash:] + '*'

            # Collect valid options (directories, or files matching the extensions)
            options = [p for p in Path(path).glob(glob)
                       if p.is_dir()
                       or len(extensions) == 0
                       or p.suffix in extensions]
            if len(options) == 0:
                return  # No options found, abort auto-completion

            # Accept immediately if there is only one option
            if len(options) == 1:
                target_path = options[0]
                path_name = target_path.name
                if target_path.is_dir() and not path_name.endswith('/'):
                    path_name += '/'
                event_data.delete_chars(last_slash, event_data.buf_text_len - last_slash)
                event_data.insert_chars(last_slash, path_name)
                return

            # There are multiple options, store them for auto-completion
            cls._completions[label] = AutoCompleteOption(options, target_idx=last_slash)

        # Write next choice to input field buffer
        target_path = cls._completions[label].next_choice
        last_slash = cls._completions[label].target_idx
        path_name = target_path.name
        if target_path.is_dir() and not path_name.endswith('/'):
            path_name += '/'
        event_data.delete_chars(last_slash, event_data.buf_text_len - last_slash)
        event_data.insert_chars(last_slash, path_name)

    @classmethod
    def font(cls, label: str, value: str, event_data: imgui.InputTextCallbackData):
        """Callback used to auto-complete font family names"""
        if not cls.has(label):
            # Collect valid options (font families that start with the input value)
            options = [font for font in FontManager.AVAILABLE_FONTS
                       if font.startswith(value) and font != value]
            if len(options) == 0:
                return

            # Accept immediately if there is only one option
            if len(options) == 1:
                font = options[0]
                event_data.delete_chars(0, event_data.buf_text_len)
                event_data.insert_chars(0, font)
                return

            # There are multiple options, store them for auto-completion
            cls._completions[label] = AutoCompleteOption(options, target_idx=0)

        # Write next choice to input field buffer
        font = cls._completions[label].next_choice
        event_data.delete_chars(0, event_data.buf_text_len)
        event_data.insert_chars(0, font)
