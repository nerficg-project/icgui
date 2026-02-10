"""State/Persistent.py: Globally persistent application state, which is stored to a file."""

import atexit
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, ClassVar

import yaml

import Framework
from Logging import Logger
from ICGui.util.Singleton import Singleton
from ICGui.util.Dataclasses import apply_overrides
from .utils import Directories


@dataclass
class PersistentState(metaclass=Singleton):
    """Persistent application state"""
    # Config path
    _path: ClassVar[Path] = Directories.USER_CONFIG_DIR / 'gui.yaml'

    # Window states
    window_open: dict[str, bool] = field(default_factory=lambda: {})
    section_expanded: dict[str, bool] = field(default_factory=lambda: {})
    section_hidden: dict[str, bool] = field(default_factory=lambda: {})

    def __post_init__(self):
        """Initializes the persistent state."""
        # Load existing state from file if it exists
        self.load()
        atexit.register(self.save)  # Save state on application exit

    def load(self) -> None:
        """Loads the persistent state from a file."""
        if not self._path.is_file():
            return

        try:
            with open(self._path, 'r') as config_file:
                yaml_dict: dict[str, Any] = yaml.safe_load(config_file)
            apply_overrides(self, yaml_dict)
        except (OSError, IOError, yaml.YAMLError) as e:
            Logger.log_warning(f'Invalid persistent GUI state config: "{self._path}", {e}')

    def save(self) -> None:
        """Saves the persistent state to a file."""
        # Ensure the directory exists
        config_dir = self._path.parent
        config_dir.mkdir(exist_ok=True)

        # Get all fields of the config
        config = asdict(self)

        # Save the configuration to a YAML file
        try:
            with open(self._path, 'w') as config_file:
                yaml.dump(config, config_file, sort_keys=False)
            Logger.log_debug(f'Saved persistent GUI state to "{self._path}"')
        except (OSError, IOError) as _:
            Logger.log_warning('Failed to save persistent GUI state.')
