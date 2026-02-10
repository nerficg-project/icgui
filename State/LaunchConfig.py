"""State/LaunchConfig.py: Argument definition and parsing for the GUI application."""

from dataclasses import dataclass, asdict, field, fields, Field
from functools import lru_cache
from pathlib import Path

import yaml

import Framework
from Logging import Logger
from ICGui.Backend.FontManager import FontSpec
from ICGui.util.Validation import validate_field
from .utils import Directories


def default_config_path():
    """Returns the default config path and ensures its parent directory exists."""
    config_dir = Path(Directories.USER_CONFIG_DIR)
    return config_dir / f'last_launch.yaml'


_NO_CONFIGURE = {
    'ui_disabled': True,
    'argparse_disabled': True,
    'validation_disabled': True,
}


@dataclass
class LaunchConfig:
    """Configuration for the GUI application."""
    launch_config_path: Path = field(init=False, default_factory=default_config_path, repr=False, metadata=_NO_CONFIGURE)
    is_training: bool = field(default=False, repr=False, metadata=_NO_CONFIGURE)

    training_config_path: Path | None = field(default=None, metadata={
        'name': 'Training Config', 'ext': ('.yaml',), 'training_locked': True,
        'filechooser_dir': 'output', 'filechooser_label': 'YAML Configuration Files',
        'flags': ('-c',), 'override_default': True,
        'argparse_kwargs': {'metavar': 'path/to/training_config.yaml', 'type': Path,
                            'help': 'Path to the training configuration file used to train the model.'},
        'help_tooltip': 'Path to the config file used to train the model. '
                        'Typically this should be the config file copied to the output '
                        'directory at the start of training.',
    })
    checkpoint_path: Path | None = field(default=None, metadata={
        'name': 'Checkpoint', 'ext': ('.pt',), 'training_disabled': True,
        'filechooser_dir': 'output', 'filechooser_label': 'Checkpoint Files',
        'flags': ('-p',), 'override_default': True,
        'argparse_kwargs': {'metavar': 'path/to/checkpoint.pt', 'type': Path,
                            'help': 'Path to the checkpoint file of the trained model.'},
        'help_tooltip': 'Path to the checkpoint file of the trained model.',
    })

    initial_resolution: list[int] = field(default_factory=lambda: [1280, 720], metadata={
        'name': 'Initial Resolution',
        'input_style': 'input', 'min': 1, 'max': 4096,
        'flags': ('-r',),
        'argparse_kwargs': {'metavar': ('width', 'height'), 'nargs': 2, 'type': int,
                            'help': 'Initial resolution of the GUI window.'},
        'help_tooltip': 'The initial resolution of the GUI window. May be overridden by your OS.',
    })
    resolution_factor: float = field(default=1.0, metadata={
        'name': 'Resolution Factor',
        'input_style': 'slider', 'min': 0.01, 'max': 4.0,
        'flags': ('-f',),
        'argparse_kwargs': {'metavar': 'factor', 'type': float,
                            'help': 'Initial resolution factor of the rendered model in the GUI window.'},
        'help_tooltip': 'The initial resolution factor of the GUI, e.g. a factor of 0.5 '
                        'means the model is rendered at half the window resolution.'
    })
    fps_rolling_average_size: int = field(default=10, metadata={
        'name': 'FPS Rolling Average Size',
        'input_style': 'input', 'min': 1, 'max': 1024,
        'argparse_kwargs': {'metavar': 'size', 'type': int,
                            'help': 'Size of the rolling average window for the FPS counter in frames.'},
        'help_tooltip': 'Number of frames over which to average the displayed FPS. '
                        'Large values can take a while to adapt to new viewpoints '
                        'or resolutions for slower models.',
    })
    dataset_near_far: bool = field(default=True, metadata={
        'name': 'Use Dataset Near/Far',
        'argparse_kwargs': {'help': 'Start with near and far planes set to the dataset values.'},
        'help_tooltip': 'Use near/far planes specified in the dataset. Otherwise, '
                        'uses arbitrary "large" values (0.01 and 1024.0).',
    })
    vsync: bool = field(default=True, metadata={
        'name': 'VSync',
        'argparse_kwargs': {'help': 'vertical frame synchronization.'},
        'help_tooltip': 'Lock the GUI framerate to your monitor\'s refresh rate. '
                        'Useful to turn this off to test model performance.',
    })
    synchronize_extras: bool = field(default=True, metadata={
        'name': 'Synchronize Extras with Model',
        'argparse_kwargs': {'help': 'synchronization of extra renderings with the model.'},
        'help_tooltip': 'Synchronize the rendering of extras (such as camera frustums) '
                        'with the currently displayed frame. Otherwise extras are updated '
                        'immediately according to user inputs.',
    })
    # TODO: Set default size based on display DPI
    font: FontSpec = field(default_factory=lambda: FontSpec('sans-serif', 18), metadata={
        'name': 'Font',
        'argparse_kwargs': {
            'font-family': {
                'help': 'Name of the font family to use for the GUI.'
            },
            'font-size': {
                'help': 'Font size to use for the GUI.',
            }
        },
    })

    @property
    def valid(self) -> bool:
        """Validates all provided fields."""
        valid = True
        cfg_fields: dict[str, Field] = {f.name: f for f in fields(self)}

        for field_name, field_info in cfg_fields.items():
            metadata = field_info.metadata
            if metadata is None:
                continue
            if metadata.get('validation_disabled', False):
                continue
            if metadata.get('training_disabled', False) and self.is_training:
                continue

            field_value = getattr(self, field_name)
            valid &= validate_field(field_value, metadata).valid

        return valid

    @Framework.catch()
    def save_to_disk(self):
        """Saves the configuration to a config file."""
        if self.launch_config_path is None:
            return

        config_dir = self.launch_config_path.parent
        config_dir.mkdir(exist_ok=True)

        # Get all fields of the config, excluding those with repr=False
        config = asdict(self)
        for config_field in fields(self):
            if config_field.repr is False:
                config.pop(config_field.name)

        # Convert Path objects to strings for YAML serialization
        for key, value in config.items():
            if isinstance(value, Path):
                config[key] = str(value)

        # Save the configuration to a YAML file
        try:
            with open(self.launch_config_path, 'w') as config_file:
                yaml.dump(config, config_file, sort_keys=False)
            Logger.log_debug(f'Saved GUI config to "{self.launch_config_path}"')
        except (OSError, IOError) as _:
            raise Framework.GUIError(f'Failed to save GUI config to "{self.launch_config_path}"')

    def infer_from_model_dir(self, path: Path):
        """Attempts to find the training config and checkpoint path given a directory."""
        # Infer config file path
        for config in path.glob('*.yaml'):
            self.training_config_path = config
            break

        # Infer checkpoint
        suggestions = self.suggest_checkpoint_paths(path)
        if len(suggestions) > 0:
            self.checkpoint_path = suggestions[0]
        else:
            self.checkpoint_path = None

    def infer_ckpt_from_cfg(self):
        """Infers the checkpoint path from the training config file."""
        if self.checkpoint_path is not None and self.checkpoint_path.is_file():
            return
        if self.training_config_path is None or not self.training_config_path.is_file():
            return

        # Infer checkpoint
        suggestions = self.suggest_checkpoint_paths(self.training_config_path.parent)
        if len(suggestions) > 0:
            self.checkpoint_path = suggestions[0]
        else:
            self.checkpoint_path = None

    def infer_cfg_from_ckpt(self):
        """Infers the training config path from the checkpoint file."""
        if self.training_config_path is not None and self.training_config_path.is_file():
            return
        if self.checkpoint_path is None or not self.checkpoint_path.is_file():
            return

        # Infer config
        config_guess = self.checkpoint_path.parent.parent / 'training_config.yaml'
        if config_guess.is_file():
            self.training_config_path = config_guess

    def guess_missing_paths(self):
        """Tries to guess the missing path from the other one."""
        cfg_fields: dict[str, Field] = {f.name: f for f in fields(self)}
        training_cfg_valid = validate_field(self.training_config_path, cfg_fields['training_config_path'].metadata)
        ckpt_valid = validate_field(self.checkpoint_path, cfg_fields['checkpoint_path'].metadata)

        # We can / need to do work only if exactly one of the paths is valid
        if not (training_cfg_valid.valid ^ ckpt_valid.valid):
            return

        if training_cfg_valid.valid:
            self.infer_ckpt_from_cfg()
        else:
            self.infer_cfg_from_ckpt()

    @staticmethod
    @lru_cache(maxsize=16)
    def suggest_checkpoint_paths(path: Path) -> list[Path]:
        """Suggests checkpoint paths based on the given path (config file or model dir)."""
        checkpoints = []
        if not path.is_dir():
            path = path.parent
        if path.name != 'checkpoints':
            path = path / 'checkpoints'
        if not path.is_dir():
            return checkpoints

        # Is there a 'latest.pt' checkpoint?
        latest = path / 'latest.pt'
        if latest.is_file():
            checkpoints.append(latest)

        # Add additional checkpoints in reverse order of modification time
        checkpoints += sorted(filter(lambda p: p.name not in checkpoints, path.glob('**/*.pt')),
                              key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoints
