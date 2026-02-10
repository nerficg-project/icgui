"""Backend/FontManager.py: ImGui Font Manager for the ICGui framework."""

from dataclasses import dataclass
from functools import lru_cache

from imgui_bundle import imgui#, hello_imgui
from typing_extensions import Callable, ParamSpec, TypeVar

import Framework
from Logging import Logger


try:
    from matplotlib import font_manager
except ImportError:
    font_manager = None


@dataclass(slots=True)
class FontSpec:
    """Font specification for loading fonts."""
    name: str = 'sans-serif'
    size: int = 18

@dataclass
class FontVariants:
    """Container for different font styles."""
    normal: imgui.ImFont
    italics: imgui.ImFont | None = None
    bold: imgui.ImFont | None = None

    def __post_init__(self):
        if self.italics is None:
            self.italics = self.normal
        if self.bold is None:
            self.bold = self.normal


_P = ParamSpec('_P')
_T = TypeVar('_T')


class FontManager:
    """Manages fonts for the ImGui interface, allowing for custom fonts and font styles."""
    current: FontVariants | None = None

    @classmethod
    def set_font(cls, font_name: str, size: int = 18) -> FontVariants:
        """Loads and sets the current default font."""
        cls.current = cls.load_font(font_name, size)
        return cls.current

    @classmethod
    def bold(cls, func: Callable[_P, _T]) -> Callable[_P, _T]:
        """Decorator to apply bold font style to a function."""
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            if cls.current is not None and cls.current.bold is not None:
                imgui.push_font(cls.current.bold, 0.0)
            result = func(*args, **kwargs)
            if cls.current is not None and cls.current.bold is not None:
                imgui.pop_font()
            return result
        return wrapper

    @classmethod
    def italics(cls, func: Callable[_P, _T]) -> Callable[_P, _T]:
        """Decorator to apply italics font style to a function."""
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            if cls.current is not None and cls.current.italics is not None:
                imgui.push_font(cls.current.italics, 0.0)
            result = func(*args, **kwargs)
            if cls.current is not None and cls.current.italics is not None:
                imgui.pop_font()
            return result
        return wrapper

    @classmethod
    def font_size(cls, func: Callable[_P, _T], font_size: float) -> Callable[_P, _T]:
        """Decorator to apply the given font size to a function."""
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            if cls.current is not None:
                imgui.push_font(None, font_size)
            result = func(*args, **kwargs)
            if cls.current is not None:
                imgui.pop_font()
            return result
        return wrapper

    @staticmethod
    def muted_color(func: Callable[_P, _T]) -> Callable[_P, _T]:
        """Decorator to apply a muted color to a function."""
        text_color = imgui.get_style_color_vec4(imgui.Col_.text)
        muted_text = imgui.ImVec4(text_color.x, text_color.y, text_color.z, text_color.w)
        muted_text.x *= 0.8
        muted_text.y *= 0.8
        muted_text.z *= 0.8

        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            imgui.push_style_color(imgui.Col_.text, muted_text)
            result = func(*args, **kwargs)
            imgui.pop_style_color()
            return result
        return wrapper

    @staticmethod
    def _load_font_with_icons(font_file: str, size: int):
        """Loads a font with FontAwesome icons. We load the icon font ourselves
           instead of hello_imgui.load_font_ttf_with_font_awesome_icons, since the
           latter does not load the icons as a monospaced font.
        """
        font = imgui.get_io().fonts.add_font_from_file_ttf(font_file, size)
        font_config = imgui.ImFontConfig()
        font_config.merge_mode = True
        font_config.glyph_min_advance_x = size  # Make sure the icon is monospaced
        icons_font_path = Framework.Directories.ICGUI_ROOT / 'resources' / 'FontAwesome6.ttf'
        if icons_font_path.is_file():
            try:
                imgui.get_io().fonts.add_font_from_file_ttf(str(icons_font_path), size, font_config)
            except Exception as e:
                Logger.log_warning(f'Failed to load font {e}')
        else:
            Logger.log_warning(f'Icons font not found at {icons_font_path}')
        return font

    # Use matplotlib's font manager if available, otherwise fall back to default fonts
    if font_manager is not None:
        AVAILABLE_FONTS = ['Default', 'sans-serif'] + sorted(font_manager.get_font_names())
        _CUSTOM_FONTS_AVAILABLE = True

        @staticmethod
        @lru_cache
        def load_font(font_name: str, size: int = 18) -> FontVariants:
            """Loads a font by name and size."""
            try:
                if font_name == 'Default':
                    # Use default font files
                    font_file = str(Framework.Directories.ICGUI_ROOT / 'resources' / 'OpenSans-Regular.ttf')
                    font_file_italics = str(Framework.Directories.ICGUI_ROOT / 'resources' / 'OpenSans-Italic.ttf')
                    font_file_bold = str(Framework.Directories.ICGUI_ROOT / 'resources' / 'OpenSans-Bold.ttf')
                else:
                    # Find font files using matplotlib's font manager
                    font_file = font_manager.findfont(font_manager.FontProperties(family=font_name, style='normal'),
                                                      fontext='ttf', fallback_to_default=True)
                    try:
                        font_file_italics = font_manager.findfont(
                            font_manager.FontProperties(family=font_name, style='italic'),
                            fontext='ttf', fallback_to_default=False
                        )
                    except ValueError:
                        font_file_italics = None
                    try:
                        font_file_bold = font_manager.findfont(
                            font_manager.FontProperties(family=font_name, weight='bold'),
                            fontext='ttf', fallback_to_default=False
                        )
                    except ValueError:
                        font_file_bold = None

                normal_font = FontManager._load_font_with_icons(font_file, size)

                # Load italics and bold variants if available
                if font_file_italics is not None:
                    italics_font = FontManager._load_font_with_icons(font_file, size)
                else:
                    italics_font = normal_font

                if font_file_bold is not None:
                    bold_font = FontManager._load_font_with_icons(font_file_bold, size)
                else:
                    bold_font = normal_font

                return FontVariants(normal=normal_font, italics=italics_font, bold=bold_font)
            except RuntimeError as e:
                Logger.log_error(f'Failed to load font: {e}')
                return FontVariants(normal=None)

    else:
        AVAILABLE_FONTS = ['Default']
        _CUSTOM_FONTS_AVAILABLE = False

        @staticmethod
        @lru_cache
        def load_font(_: str, size: int = 18):
            """Loads the default font with icons."""
            try:
                normal_font = FontManager._load_font_with_icons(
                    str(Framework.Directories.ICGUI_ROOT / 'resources' / 'OpenSans-Regular.ttf'),
                    size,
                )
                italics_font = FontManager._load_font_with_icons(
                    str(Framework.Directories.ICGUI_ROOT / 'resources' / 'OpenSans-Italic.ttf'),
                    size,
                )
                bold_font = FontManager._load_font_with_icons(
                    str(Framework.Directories.ICGUI_ROOT / 'resources' / 'OpenSans-Bold.ttf'),
                    size,
                )

                return FontVariants(normal=normal_font, italics=italics_font, bold=bold_font)
            except RuntimeError as e:
                Logger.log_error(f'Failed to load font: {e}')
                return FontVariants(normal=None)
