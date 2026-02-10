"""State/util.py: Utilities related to state management"""

from pathlib import Path

from platformdirs import user_config_dir


class Directories:
    USER_CONFIG_DIR: Path = Path(
        user_config_dir('NerfICG', 'TUBS-ICG', ensure_exists=True)
    )
