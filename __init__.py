"""Required to make multiprocessing find the source directory by setting
   sys.path if necessary."""

import sys
from pathlib import Path

_SRC_PATH = str(Path(__file__).resolve().parents[0])
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
