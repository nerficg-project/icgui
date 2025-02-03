# -- coding: utf-8 --

"""Applications/SetupGUI.py: Implements the setup GUI, providing configuration
   options to the user before launching the main GUI."""

import sys
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, fields, Field, MISSING
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin

import imgui
import yaml
from imgui.core import _ImGuiInputTextCallbackData
from platformdirs import user_config_dir
from tkinter.filedialog import askopenfilename, askdirectory

from Logging import Logger
from ICGui.Backends import BaseBackend, SDL2Backend

try:
    from matplotlib import font_manager
except ImportError:
    font_manager = None


# pylint: disable = too-few-public-methods
class SetupGUI:
    """Setup GUI class for the configuring the main GUI before launching it"""
    SHIFT_KEY = imgui.KEY_MOD_SHIFT

    @dataclass
    class _AutoCompleteOption:
        """Auto-complete option for an input field"""
        options: list[Any]
        _current_index: int = -1

        @property
        def next_choice(self):
            """Returns the next auto-complete choice"""
            if imgui.is_key_down(SetupGUI.SHIFT_KEY):
                if self._current_index == -1:
                    self._current_index = 0
                self._current_index = (self._current_index - 1) % len(self.options)
            else:
                self._current_index = (self._current_index + 1) % len(self.options)
            choice = self.options[self._current_index]
            return choice

    ERROR_COLORS = {
        imgui.COLOR_FRAME_BACKGROUND: (0.31, 0.137, 0.177)
    }
    ACCEPT_COLOR = (0.133, 0.284, 0.113)
    ACCEPT_COLOR_ACTIVE = (0.157, 0.334, 0.133)
    ACCEPT_COLOR_HOVERED = (0.18, 0.384, 0.153)

    @staticmethod
    @contextmanager
    def validateField(fld: Field, value, error_styles: tuple[int], validator: callable = None) -> bool:
        """Validates a single field"""
        validator_kwargs = {
            'ext': fld.metadata.get('ext', None),
            'min_val': fld.metadata.get('min', None),
            'max_val': fld.metadata.get('max', None),
        }
        if validator is not None:
            valid = validator(value, **validator_kwargs)
        else:
            valid = True

        if not valid:
            for style in error_styles:
                imgui.push_style_color(style, *SetupGUI.ERROR_COLORS[style])

        try:
            yield valid
        finally:
            if not valid:
                imgui.pop_style_color(len(error_styles))

    def __init__(self):
        self._window_size: tuple[int, int] = (1280, 720)
        self._backend: BaseBackend = (
            SDL2Backend('NerfICG Launcher', self._window_size,
                        vsync=True, resize_callback=self._updateSize))
        SetupGUI.SHIFT_KEY = self._backend.getKeyCode('Shift')
        self.config: 'LaunchConfig | None' = None
        self._input_cache: dict[str, str] = {}
        self._autocomplete_option: dict[str, SetupGUI._AutoCompleteOption] = {}
        self._dropdown_state: dict[str, bool] = {}
        self._invalid_inputs = set()
        self._setupImGui()

        self._setup_config = {
            'last_dialog_dirs': {},
        }
        self._readConfig()

        if font_manager is not None:
            self._available_fonts = ['sans-serif'] + sorted(font_manager.get_font_names())
        else:
            self._available_fonts = ['Fonts not available.']

    def _readConfig(self):
        try:
            with open(Path(user_config_dir('NerfICG', 'TUBS-ICG',
                                           ensure_exists=True)) / 'setup_gui.yaml', 'r') as f:
                self._setup_config = yaml.safe_load(f)
        except FileNotFoundError:
            self._setup_config = {
                'last_dialog_dirs': {},
            }
            self._writeConfig()
        except yaml.YAMLError as e:
            Logger.logDebug(f'Error reading setup_gui.yaml: {e}')
            self._setup_config = {
                'last_dialog_dirs': {},
            }

        # Convert types
        for directory in self._setup_config['last_dialog_dirs']:
            self._setup_config['last_dialog_dirs'][directory] = Path(self._setup_config['last_dialog_dirs'][directory])

    def _writeConfig(self):
        config = deepcopy(self._setup_config)
        for directory in config['last_dialog_dirs']:
            config['last_dialog_dirs'][directory] = str(config['last_dialog_dirs'][directory].absolute())

        with open(Path(user_config_dir('NerfICG', 'TUBS-ICG',
                                       ensure_exists=True)) / 'setup_gui.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def _setupImGui(self):
        """Sets up the ImGui context"""
        self._ctx = imgui.create_context()
        imgui.get_io().display_size = self._window_size
        imgui.get_io().ini_file_name = None

        if font_manager is not None:
            font_file = font_manager.findfont(font_manager.FontProperties(family='sans-serif', style='normal'),
                                              fontext='ttf', fallback_to_default=True)
            self._default_font = imgui.get_io().fonts.add_font_from_file_ttf(font_file, 22)

            # font_file = font_manager.findfont(font_manager.FontProperties(family='sans-serif', style='italic'),
            #                                   fontext='ttf', fallback_to_default=True)
            # self._italics_font = imgui.get_io().fonts.add_font_from_file_ttf(font_file, 22)

            imgui.get_io().font_global_scale = 1
        else:
            imgui.get_io().fonts.add_font_default()
            imgui.get_io().font_global_scale = 1.5

    def _updateSize(self, width: int, height: int):
        """Resize callback used to update the width and height in the backend"""
        self._window_size = (width, height)

    # pylint: disable=too-many-return-statements
    def _autocompletePath(self, fieldname: str, extensions: tuple[str], data: _ImGuiInputTextCallbackData):
        """Callback used to auto-complete paths"""
        if data.event_flag & imgui.INPUT_TEXT_CALLBACK_EDIT:
            if fieldname in self._autocomplete_option:
                del self._autocomplete_option[fieldname]
            return None

        if data.event_flag & imgui.INPUT_TEXT_CALLBACK_CHAR_FILTER:
            if data.event_char == '/' and fieldname in self._autocomplete_option:
                del self._autocomplete_option[fieldname]
                if data.has_selection():
                    return None
                return 1

        if not data.event_flag & imgui.INPUT_TEXT_CALLBACK_COMPLETION:
            return None

        if fieldname not in self._autocomplete_option:
            last_slash = data.buffer.rfind('/') + 1
            path = Path(data.buffer[:last_slash])
            glob = data.buffer[last_slash:] + '*'

            options = [p for p in Path(path).glob(glob)
                       if p.is_dir()
                       or len(extensions) == 0
                       or p.suffix in extensions]
            if len(options) == 0:
                return None
            self._autocomplete_option[fieldname] = SetupGUI._AutoCompleteOption(options)

        if len(self._autocomplete_option[fieldname].options) == 1:
            path_str = str(self._autocomplete_option[fieldname].options[0])
            if self._autocomplete_option[fieldname].options[0].is_dir():
                path_str += '/'
            data.delete_chars(0, len(data.buffer))
            data.insert_chars(0, path_str)
            del self._autocomplete_option[fieldname]
            return None

        choice = self._autocomplete_option[fieldname].next_choice
        path_str = str(choice)
        if choice.is_dir():
            path_str += '/'
        data.delete_chars(0, len(data.buffer))
        data.insert_chars(0, path_str)
        return None

    def _autocompleteFont(self, fieldname: str, data: _ImGuiInputTextCallbackData):
        """Callback used to auto-complete font family names"""
        if data.event_flag & imgui.INPUT_TEXT_CALLBACK_EDIT:
            if fieldname in self._autocomplete_option:
                del self._autocomplete_option[fieldname]
            return None

        if not data.event_flag & imgui.INPUT_TEXT_CALLBACK_COMPLETION:
            return None

        if fieldname not in self._autocomplete_option:
            options = [font for font in self._available_fonts if font.startswith(data.buffer) and font != data.buffer]
            self._autocomplete_option[fieldname] = SetupGUI._AutoCompleteOption(options)

        if len(self._autocomplete_option[fieldname].options) == 1:
            font = self._autocomplete_option[fieldname].options[0]
            data.delete_chars(0, len(data.buffer))
            data.insert_chars(0, font)
            del self._autocomplete_option[fieldname]
            return None

        font = self._autocomplete_option[fieldname].next_choice
        data.delete_chars(0, len(data.buffer))
        data.insert_chars(0, font)
        return None

    def _renderFileSelector(self, field: Field, locked: bool = False):
        """Renders a single file selector field"""
        value: Path | None = getattr(self.config, field.name)
        valid_extensions = field.metadata.get('ext', ())
        if field.name not in self._input_cache:
            if value is None:
                self._input_cache[field.name] = ''
            else:
                self._input_cache[field.name] = str(value)

        with self.validateField(field, value,
                                (imgui.COLOR_FRAME_BACKGROUND,),
                                field.metadata.get('validator', None)):
            flags = (imgui.INPUT_TEXT_CALLBACK_COMPLETION
                     | imgui.INPUT_TEXT_CALLBACK_EDIT
                     | imgui.INPUT_TEXT_CALLBACK_CHAR_FILTER)
            if locked:
                flags |= imgui.INPUT_TEXT_READ_ONLY

            modified, self._input_cache[field.name] = \
                imgui.input_text(f'##{field.name}', self._input_cache[field.name],
                                 flags=flags,
                                 callback=lambda d: self._autocompletePath(field.name,
                                                                           valid_extensions,
                                                                           d))

        if not locked:
            imgui.same_line()
            if imgui.button(f'...##{field.name}'):
                # Open file selector
                filters = None if len(valid_extensions) == 0 else \
                    [(field.metadata.get('filechooser_label', f'*{ext}'), f'*{ext}') for ext in valid_extensions]
                path = askopenfilename(title=f'Select {field.metadata.get("name", field.name)}...',
                                       initialdir=self._setup_config['last_dialog_dirs']
                                           .get(field.name, str(Path(__file__).resolve().parents[3]
                                                                / field.metadata.get('filechooser_dir', ''))),
                                       filetypes=filters)
                if path is not None and path != tuple():
                    self._input_cache[field.name] = path
                    self._setup_config['last_dialog_dirs'][field.name] = Path(path).parent
                    self._writeConfig()
                    modified = True

        imgui.same_line()
        imgui.text(field.metadata.get('name', field.name))

        if modified:
            setattr(self.config, field.name, Path(self._input_cache[field.name]))

    def _renderFontSelector(self, field: Field):
        """Renders a single font selector field"""
        label = field.metadata.get('name', field.name)
        value = getattr(self.config, field.name)
        force_focus = False

        with self.validateField(field, value, (imgui.COLOR_FRAME_BACKGROUND,),
                                field.metadata.get('validator', None)):
            changed, value = imgui.input_text(f'##{label}_text', value,
                                              flags=imgui.INPUT_TEXT_CALLBACK_COMPLETION |
                                                    imgui.INPUT_TEXT_CALLBACK_EDIT,
                                              callback=lambda d: self._autocompleteFont(field.name, d))
        if changed:
            force_focus = True
            setattr(self.config, field.name, value)

        try:
            item_idx = self._available_fonts.index(value)
        except ValueError:
            item_idx = -1

        imgui.same_line()
        clicked = imgui.arrow_button(f'##{label}_arrow', imgui.DIRECTION_DOWN if self._dropdown_state.get(label, False)
                                     else imgui.DIRECTION_RIGHT)
        if clicked:
            self._dropdown_state[label] = not self._dropdown_state.get(label, False)

        imgui.same_line()
        imgui.text(label)

        if self._dropdown_state.get(label, False):
            with imgui.begin_list_box(f'##{label}_listbox'):
                for idx, font in enumerate(self._available_fonts):
                    selected = idx == item_idx
                    clicked, _ = imgui.selectable(font, selected)
                    if clicked:
                        item_idx = idx
                        setattr(self.config, field.name, font)

                    if selected:
                        imgui.set_item_default_focus()
                    if selected and force_focus:
                        imgui.set_scroll_here_y()

        imgui.spacing()

    # pylint: disable=too-many-branches,too-many-locals
    def _renderInputField(self, field: Field, field_type: type, container: type | None = None):
        """Renders a single list field"""
        label = field.metadata.get('name', field.name)
        value = getattr(self.config, field.name)
        input_style: str = field.metadata.get('input_style', 'input')
        min_val = field.metadata.get('min', 0)
        max_val = field.metadata.get('max', 2**32)
        step = field.metadata.get('step', 1)
        additional_args = ()
        if input_style == 'slider':
            additional_args = (min_val, max_val)
        elif input_style == 'drag':
            additional_args = (step, min_val, max_val)

        input_type: str = field_type.__name__.lower()
        if input_type == 'str':
            input_type = 'text'

        length = ''
        if container is not None:
            if len(value) > 4:
                return
            length = str(len(value)) if len(value) > 1 else ''
        if container is None:
            value = [value]

        with self.validateField(field, value,
                                (imgui.COLOR_FRAME_BACKGROUND,),
                                field.metadata.get('validator', None)):
            changed, values = imgui.__dict__[f'{input_style}_{input_type}{length}'](label, *value, *additional_args)

        if changed:
            if container is not None:
                if len(values) == 1:
                    values = [values]
                if input_type != 'text':
                    for i, value in enumerate(values):
                        if value < min_val:
                            values[i] = min_val
                        if value > max_val:
                            values[i] = max_val
                setattr(self.config, field.name, list(values))
            else:
                if input_type != 'text':
                    setattr(self.config, field.name, min(max(values, min_val), max_val))
                else:
                    setattr(self.config, field.name, values)

    def _renderCheckbox(self, field: Field):
        """Renders a single checkbox field"""
        state = getattr(self.config, field.name)
        with self.validateField(field, state,
                                (imgui.COLOR_FRAME_BACKGROUND,),
                                field.metadata.get('validator', None)):
            clicked, state = imgui.checkbox(field.metadata.get('name', field.name), state)
        if clicked:
            setattr(self.config, field.name, state)

    def _renderInput(self, field: Field):
        """Renders a single input field"""
        if field.name.startswith('_'):
            return None
        if field.metadata.get('config_disabled', False):
            return None
        if field.metadata.get('training_disabled', False) and self.config.is_training:
            return None

        if field.default is not MISSING or field.default_factory is not MISSING:
            if imgui.button(f'«##{field.name}'):
                if field.default is not MISSING:
                    setattr(self.config, field.name, field.default)
                else:
                    setattr(self.config, field.name, field.default_factory())
                # Invalidate cache
                self._input_cache = {}
            imgui.same_line()

        field_type = (field.metadata.get('override_type', field.type),)
        generic = None
        if get_origin(field.type) in (Union, UnionType):
            field_type = get_args(field.type)
        if get_origin(field.type) is list:
            field_type = (get_args(field.type)[0],)
            generic = get_origin(field.type)
        if get_origin(field.type) is dict:
            field_type = (get_args(field.type)[1],)
            generic = get_origin(field.type)

        if generic is not None:
            return self._renderInputField(field, field_type[0], generic)
        if Path in field_type:
            return self._renderFileSelector(field, locked=field.metadata.get('training_locked', False)
                                                          and self.config.is_training)
        if bool in field_type:
            return self._renderCheckbox(field)
        if 'font-family' in field_type:
            return self._renderFontSelector(field)
        if field_type[0] in (int, float, str):
            return self._renderInputField(field, field_type[0])

        return None

    def _renderDirectorySelector(self):
        """Renders the directory selector"""
        if self.config.is_training:
            return

        if imgui.button('Select trained model directory...'):
            # Open directory selector
            path = askdirectory(title='Select model directory...',
                                mustexist=True,
                                initialdir=self._setup_config['last_dialog_dirs']
                                    .get('model_dir', str(Path(__file__).resolve().parents[3] / 'output')))

            if path is not None and path != tuple():
                self.config.fromDirectory(Path(path))
                self._setup_config['last_dialog_dirs']['model_dir'] = Path(path)
                self._writeConfig()

                # Invalidate cache
                self._input_cache = {}

    def _render(self) -> bool:
        """Renders the ImGui components"""
        keep_open = True
        flags = imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE \
                | imgui.WINDOW_NO_SAVED_SETTINGS

        imgui.set_next_window_size(self._window_size[0] - 8, self._window_size[1] - 8)
        imgui.set_next_window_position(4, 4)
        with imgui.begin('Config', flags=flags):
            with imgui.begin_child('ScrollingRegion', 0, -imgui.get_frame_height_with_spacing(), border=True):
                self._renderDirectorySelector()
                for field in fields(self.config):
                    self._renderInput(field)

            num_buttons = 2
            if not self.config.is_training:
                num_buttons = 1
            avail = imgui.get_content_region_available_width()
            width = imgui.get_style().item_spacing.x * (num_buttons - 1) \
                    + imgui.get_style().frame_padding.x * 2 * num_buttons \
                    + imgui.calc_text_size('Guess Missing Path').x if not self.config.is_training else 0 \
                    + imgui.calc_text_size('Launch').x
            imgui.set_cursor_pos_x((avail - width) / 2)
            imgui.push_style_color(imgui.COLOR_BUTTON, *self.ACCEPT_COLOR)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *self.ACCEPT_COLOR_ACTIVE)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *self.ACCEPT_COLOR_HOVERED)

            if not self.config.is_training:
                if imgui.button('Guess Missing Path'):
                    if self.config.guessMissingPath():
                        pass
                    # Invalidate cache
                    self._input_cache = {}
                imgui.same_line()

            if imgui.button('Launch'):
                if self.config.validate():
                    keep_open = False

            imgui.pop_style_color(3)

        imgui.render()
        imgui.end_frame()

        return keep_open

    def configure(self, config: 'LaunchConfig') -> 'LaunchConfig':
        """Runs the GUI loop as long as the backend reports that the window is open
           and fills in the passed LaunchConfig."""
        self.config = config

        with self._backend as backend:
            keep_open = True
            while keep_open:
                # Close the GUI if the window is closed
                if not backend.beforeFrame():
                    sys.exit(0)

                # Update in-memory references to the current GUI state
                keep_open = self._render()

                # Render the frame
                backend.clearFrame()
                backend.afterFrame()

        imgui.destroy_context(self._ctx)
        return config
