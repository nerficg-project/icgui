"""Required to make multiprocessing find the source directory by setting
   sys.path if necessary."""

import os
import sys
from pathlib import Path

_SRC_PATH = str(Path(__file__).resolve().parent)
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

# Disable PySDL3 version check
os.environ["SDL_CHECK_VERSION"] = '0'

# Don't expose temporary module imports
__all__ = []
