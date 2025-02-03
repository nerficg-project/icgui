# -- coding: utf-8 --

"""GuiConfig/ConfigValidator.py: Validation methods for the launch configuration."""

from pathlib import Path

try:
    from matplotlib import font_manager
except ImportError:
    font_manager = None

ListElement = int | float | str


class ConfigValidator:
    """Class for validating the GUI configuration."""
    @staticmethod
    def validatePath(path: Path | None, ext: tuple[str] | None, logger_func: callable = lambda _: None, **_) -> bool:
        """Validates a path."""
        if path is None:
            logger_func('Missing Path.')
            return False
        if not path.is_file():
            logger_func(f'Not a valid file: {path}')
            return False
        if ext is not None and len(ext) > 0 and path.suffix not in ext:
            logger_func(f'Not a valid file extension: {path} (expected: {", ".join(ext)})')
            return False
        logger_func('Valid.')
        return True

    @staticmethod
    def validateFontFamily(font_family: str, logger_func: callable = lambda _: _, **_) -> bool:
        """Validates a font family input."""
        # Is matplotlib installed?
        if font_manager is None:
            # Ignore font if matplotlib is not installed
            logger_func('Matplotlib not installed, skipping font validation.')
            return True

        # Can a font with the given family be found?
        try:
            font = font_manager.findfont(font_manager.FontProperties(family=font_family, style='normal'), fontext='ttf',
                                         fallback_to_default=False)
        except ValueError:
            logger_func(f'Invalid font family: {font_family}')
            return False

        # Can the font file be opened?
        try:
            with open(font, 'rb') as _:
                pass
        except OSError:
            logger_func(f'Unable to open font file: {font}')
            return False

        logger_func('Valid.')
        return True

    @staticmethod
    def validateInput(values, min_val=None, max_val=None, logger_func: callable = lambda _: _, **_) -> bool:
        """Validates a single input value or list of input values."""
        if isinstance(values, list):
            for i, val in enumerate(values):
                if val < min_val:
                    logger_func(f'Invalid value at index {i}: {val} < min ({min_val})')
                    return False
                if val > max_val:
                    logger_func(f'Invalid value at index {i}: {val} > max ({max_val})')
                    return False
        else:
            if values < min_val:
                logger_func(f'Invalid value: {values} < min ({min_val})')
                return False
            if values > max_val:
                logger_func(f'Invalid value: {values} > max ({max_val})')
                return False

        logger_func('Valid.')
        return True
