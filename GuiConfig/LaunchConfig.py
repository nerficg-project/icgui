# -- coding: utf-8 --

"""LaunchConfig.py: Argument definition and parsing for the GUI application."""

from argparse import ArgumentParser, Namespace
from dataclasses import Field, dataclass, field, fields, asdict
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin

import yaml
from platformdirs import user_config_dir

from ICGui.GuiConfig.utils import addArgumentsFromFields, md5
from ICGui.GuiConfig.ConfigValidator import ConfigValidator
from ICGui.Applications.SetupGUI import SetupGUI
import Framework
from Logging import Logger


@dataclass
class LaunchConfig:
    """Configuration for the GUI application."""
    _config_path: Path = field(init=False, default=None, repr=False)
    _training: bool = field(default=False, repr=False)
    gui_window_states: dict[str, bool] = field(init=False, default_factory=dict, metadata={
        'config_disabled': True,
        'argparse_disabled': True,
    })

    training_config_path: Path | None = field(default=None, metadata={
        'name': 'Training Config Path', 'ext': ('.yaml',), 'training_locked': True,
        'filechooser_dir': 'output', 'filechooser_label': 'YAML Configuration Files',
        'validator': ConfigValidator.validatePath,
        'flags': ('-c',),
        'argparse_kwargs': {'metavar': 'path/to/training_config.yaml', 'type': Path,
                            'help': 'Path to the training configuration file used to train the model.'}
    })
    checkpoint_path: Path | None = field(default=None, metadata={
        'name': 'Checkpoint Path', 'ext': ('.pt',), 'training_disabled': True,
        'filechooser_dir': 'output', 'filechooser_label': 'Checkpoint Files',
        'validator': ConfigValidator.validatePath,
        'flags': ('-p',),
        'argparse_kwargs': {'metavar': 'path/to/checkpoint.pt', 'type': Path,
                            'help': 'Path to the checkpoint file of the trained model.'},
    })

    initial_resolution: list[int] = field(default_factory=lambda: [1280, 720], metadata={
        'name': 'Initial Resolution',
        'input_style': 'input', 'min': 1, 'max': 4096,
        'validator': ConfigValidator.validateInput,
        'flags': ('-r',),
        'argparse_kwargs': {'metavar': ('width', 'height'), 'nargs': 2, 'type': int,
                            'help': 'Initial resolution of the GUI window.'},
    })
    resolution_factor: float = field(default=1.0, metadata={
        'name': 'Resolution Factor',
        'input_style': 'slider', 'min': 0.01, 'max': 4.0,
        'validator': ConfigValidator.validateInput,
        'flags': ('-f',),
        'argparse_kwargs': {'metavar': 'factor', 'type': float,
                            'help': 'Initial resolution factor of the rendered model in the GUI window.'},
    })
    dataset_near_far: bool = field(default=True, metadata={
        'name': 'Use Dataset Near/Far',
        'argparse_kwargs': {'help': 'Start with near and far planes set to the dataset values.'},
    })
    save_window_positions: bool = field(default=True, metadata={
        'name': 'Save Window Positions',
        'argparse_kwargs': {'help': 'saving of GUI window positions between sessions.'},
    })
    font_family: str = field(default='sans-serif', metadata={
        'name': 'Font Family', 'override_type': 'font-family',
        'validator': ConfigValidator.validateFontFamily,
        'argparse_kwargs': {'metavar': 'font-family', 'type': str,
                            'help': 'Name of the font family to use for the GUI.'},
    })
    font_size: int = field(default=12, metadata={
        'name': 'Font Size',
        'input_style': 'input', 'min': 1, 'max': 144,
        'validator': ConfigValidator.validateInput,
        'argparse_kwargs': {'metavar': 'font-size', 'type': int,
                            'help': 'Font size used for the GUI in pt.'},
    })
    fps_rolling_average_size: int = field(default=10, metadata={
        'name': 'FPS Rolling Average Size',
        'input_style': 'input', 'min': 1, 'max': 1024,
        'validator': ConfigValidator.validateInput,
        'argparse_kwargs': {'metavar': 'size', 'type': int,
                            'help': 'Size of the rolling average window for the FPS counter in frames.'},
    })
    vsync: bool = field(default=True, metadata={
        'name': 'VSync',
        'argparse_kwargs': {'help': 'vertical frame synchronization.'},
    })
    synchronize_extras: bool = field(default=True, metadata={
        'name': 'Synchronize Extras with Model',
        'argparse_kwargs': {'help': 'synchronization of extra renderings with the model.'},
    })

    def validate(self) -> bool:
        """Validates all provided fields."""
        def loggerFunction(msg: str):
            Logger.logDebug('[Validator]   => ' + msg)

        own_fields: dict[str, Field] = {f.name: f for f in fields(self)}
        valid = True

        Logger.logDebug('[Validator] Validating GUI config...')
        Logger.logDebug('[Validator] - Training Config Path...')
        valid &= ConfigValidator.validatePath(self.training_config_path,
                                              own_fields['training_config_path'].metadata.get('ext', None),
                                              logger_func=loggerFunction)
        if not self._training:
            Logger.logDebug('[Validator] - Checkpoint Path...')
            valid &= ConfigValidator.validatePath(self.checkpoint_path,
                                                  own_fields['checkpoint_path'].metadata.get('ext', None),
                                                  logger_func=loggerFunction)
        Logger.logDebug('[Validator] - Initial Resolution...')
        valid &= ConfigValidator.validateInput(self.initial_resolution,
                                               own_fields['checkpoint_path'].metadata.get('min', 1),
                                               own_fields['checkpoint_path'].metadata.get('max', 2 ** 16),
                                               logger_func=loggerFunction)
        Logger.logDebug('[Validator] - Resolution Factor...')
        valid &= ConfigValidator.validateInput(self.resolution_factor,
                                               own_fields['checkpoint_path'].metadata.get('min', 0.01),
                                               own_fields['checkpoint_path'].metadata.get('max', 4.0),
                                               logger_func=loggerFunction)
        Logger.logDebug('[Validator] - Font Family...')
        valid &= ConfigValidator.validateFontFamily(self.font_family, logger_func=loggerFunction)
        Logger.logDebug('[Validator] - Font Size...')
        valid &= ConfigValidator.validateInput(self.font_size,
                                               own_fields['font_size'].metadata.get('min', 0.01),
                                               own_fields['font_size'].metadata.get('max', 4.0),
                                               logger_func=loggerFunction)
        Logger.logDebug('[Validator] - FPS Rolling Average Size...')
        valid &= ConfigValidator.validateInput(self.fps_rolling_average_size,
                                               own_fields['checkpoint_path'].metadata.get('min', 1),
                                               own_fields['checkpoint_path'].metadata.get('max', 1024),
                                               logger_func=loggerFunction)

        Logger.logDebug('[Validator] => GUI config is valid.' if valid
                        else '[Validator] => GUI config is invalid!')

        return valid

    def fillConfigPath(self):
        """Fills the config path if it is not set."""
        if self._config_path is not None:
            return

        config_dir = Path(user_config_dir('NerfICG', 'TUBS-ICG', ensure_exists=True)) / 'gui_config'
        config_dir.mkdir(exist_ok=True)
        config_name = md5(str(Path(self.checkpoint_path).absolute()))
        self._config_path = config_dir / f'{config_name}.yaml'

    def fromDirectory(self, path: Path) -> bool:
        """Tries to find the training config and checkpoint path given a directory."""
        for config in path.glob('*.yaml'):
            self.training_config_path = config
            break
        else:
            return False

        if (checkpoint := path / 'checkpoints' / 'final.pt').exists() and checkpoint.is_file():
            self.checkpoint_path = checkpoint
            return True
        for checkpoint in path.glob('**/*.pt'):
            if checkpoint.is_file():
                self.checkpoint_path = checkpoint
                return True

        return False

    def findCheckpointPath(self) -> bool:
        """Tries to find the checkpoint path if given a directory."""
        if self.checkpoint_path is None or not self.checkpoint_path.is_dir():
            return False

        checkpoints = sorted(self.checkpoint_path.glob('**/*.pt'), key=lambda p: p.stat().st_mtime, reverse=True)
        if len(checkpoints) < 1:
            return False

        self.checkpoint_path = checkpoints[0]
        return True

    def guessCheckpointPath(self) -> bool:
        """Tries to guess the checkpoint path from the training config path."""
        # Is the training config a valid file?
        if not self.training_config_path.is_file() or self.training_config_path.suffix != '.yaml':
            Logger.logDebug(f'[PathGuesser] Invalid training config path: "{self.training_config_path}"')
            return False
        Logger.logDebug('[PathGuesser] Searching for checkpoint file...')

        # Check if the specified config is in the output directory
        # (i.e. it is the backup config saved during training)
        potential_output_directory = self.training_config_path.parent
        if not (potential_output_directory / 'checkpoints').is_dir():
            Logger.logDebug(f'[PathGuesser]  => Not a valid directory: {potential_output_directory / "checkpoints"}')
            # Instead it might be in the configs directory:
            # In this case, we need to parse it to find the correct output directory
            try:
                with open(self.training_config_path, 'r') as config_file:
                    yaml_dict: dict[str, Any] = yaml.safe_load(config_file)
                model_name = yaml_dict['TRAINING']['MODEL_NAME']
                Logger.logDebug(f'[PathGuesser]  => Determined model name: {model_name}')
                candidates = sorted((potential_output_directory.parent / 'output').glob(f'{model_name}_*'),
                                    reverse=True)
                if len(candidates) < 1:
                    Logger.logDebug('[PathGuesser]  => Found no potential output directories')
                    return False
                potential_output_directory = candidates[0]
                if not (potential_output_directory / 'checkpoints').is_dir():
                    Logger.logDebug(f'[PathGuesser]  => Not a valid directory: '
                                    f'{potential_output_directory / "checkpoints"}')
                    return False
            except (OSError, IOError, yaml.YAMLError, KeyError):
                return False
        Logger.logDebug(f'[PathGuesser]  => Determined potential output directory: {potential_output_directory}')

        # Is there a 'latest.pt' checkpoint?
        checkpoints = [potential_output_directory / 'checkpoints' / 'latest.pt']
        if not checkpoints[0].is_file():
            Logger.logDebug(f'[PathGuesser]  => Did not find latest.pt checkpoint under: {checkpoints[0]}')
            # If not, is there any other checkpoint?
            checkpoints = sorted((potential_output_directory / 'checkpoints').glob('*.pt'),
                                 key=lambda p: p.stat().st_mtime, reverse=True)
            if len(checkpoints) < 1:
                Logger.logDebug('[PathGuesser]  => Found no checkpoints in directory')
                return False

        self.checkpoint_path = checkpoints[0]
        Logger.logDebug(f'[PathGuesser]  => Successfully determined checkpoint path: {self.checkpoint_path}')
        return True

    def guessTrainingConfigPath(self) -> bool:
        """Tries to guess the training config path from the checkpoint path."""
        if not self.checkpoint_path.is_file():
            return False
        if self.checkpoint_path.suffix != '.pt':
            return False

        # Checkpoint file is assumed to be in outputs/*/checkpoints, with the training config under outputs/*/
        potential_output_directory = self.checkpoint_path.parent.parent
        if (potential_output_directory / 'training_config.yaml').is_file():
            self.training_config_path = potential_output_directory / 'training_config.yaml'
            return True

        return False

    def guessMissingPath(self) -> bool:
        """Tries to guess the missing path from the other one."""
        training_config_path_valid = self.training_config_path is not None \
                                     and self.training_config_path.is_file() \
                                     and self.training_config_path.suffix == '.yaml'
        checkpoint_path_valid = self.checkpoint_path is not None \
                                and self.checkpoint_path.is_file() \
                                and self.checkpoint_path.suffix == '.pt'
        checkpoint_path_valid |= self.checkpoint_path is not None \
                                    and self.checkpoint_path.is_dir() \
                                    and self.findCheckpointPath()

        if training_config_path_valid and checkpoint_path_valid:
            Logger.logDebug('[PathGuesser] Both paths are already valid.')
            return False

        if not (training_config_path_valid or checkpoint_path_valid):
            Logger.logDebug('[PathGuesser] Both paths are missing or invalid.')
            return False

        if training_config_path_valid:
            return self.guessCheckpointPath()

        return self.guessTrainingConfigPath()

    def overrideFromConfig(self, config_name: str | None) -> 'LaunchConfig':
        """Overrides the configuration from a config file."""
        config_dir = Path(user_config_dir('NerfICG', 'TUBS-ICG', ensure_exists=True)) / 'gui_config'
        config_dir.mkdir(exist_ok=True)

        if config_name is None:
            try:
                config_path = sorted(config_dir.glob('*.yaml'), key=lambda p: p.stat().st_mtime, reverse=True)[0]
            except IndexError:
                return self
        else:
            config_path = config_dir / f'{config_name}.yaml'

        self._config_path = config_path
        if not config_path.exists():
            return self

        try:
            with open(config_path, 'r') as config_file:
                yaml_dict: dict[str, Any] = yaml.safe_load(config_file)
        except (OSError, IOError, yaml.YAMLError) as exc:
            raise Framework.GUIError(f'invalid config file path: "{config_path}"') from exc

        for config_field in fields(LaunchConfig):
            if config_field.name in yaml_dict and yaml_dict[config_field.name] is not None:
                # If the type is a union type, use the first type within it
                field_type = config_field.type
                if get_origin(config_field.type) in (Union, UnionType):
                    field_type = get_args(config_field.type)[0]

                # Convert type if necessary
                setattr(self, config_field.name,
                        field_type(yaml_dict[config_field.name]))

        return self

    def overrideFromCmdArgs(self, args: Namespace):
        """Overrides the configuration from command line arguments."""
        if getattr(args, 'model_dir', None) is not None:
            self.fromDirectory(args.model_dir)

        for config_field in fields(LaunchConfig):
            if hasattr(args, config_field.name) and getattr(args, config_field.name) is not None:
                # If the type is a union type, use the first type within it
                field_type = config_field.type
                if get_origin(config_field.type) in (Union, UnionType):
                    field_type = get_args(config_field.type)[0]

                # Convert type if necessary
                setattr(self, config_field.name,
                        field_type(getattr(args, config_field.name)))
                config_field.default = getattr(self, config_field.name)

    def applyOverrides(self, overrides: dict[str, Any]):
        """Applies overrides dict to the configuration."""
        for config_field in fields(LaunchConfig):
            if config_field.name in overrides:
                # If the type is a union type, use the first type within it
                field_type = config_field.type
                if get_origin(config_field.type) in (Union, UnionType):
                    field_type = get_args(config_field.type)[0]

                # Convert type if necessary
                setattr(self, config_field.name,
                        field_type(overrides[config_field.name]))
                config_field.default = getattr(self, config_field.name)

    def save(self):
        """Saves the configuration to a config file."""
        if self._config_path is None:
            return

        config_dir = self._config_path.parent
        config_dir.mkdir(exist_ok=True)

        config = asdict(self)
        for fld in fields(self):
            if fld.repr is False:
                config.pop(fld.name)

        for key, value in config.items():
            if isinstance(value, Path):
                config[key] = str(value)

        try:
            with open(self._config_path, 'w') as config_file:
                yaml.dump(config, config_file, sort_keys=False)
            Logger.logDebug(f'Saved GUI config to "{self._config_path}"')
        except (OSError, IOError) as _:
            Logger.logDebug(f'Failed to save GUI config to "{self._config_path}"')

    @staticmethod
    def fromCommandLine(config_name: str = None,
                        disable_cmd_args: list[str] = None,
                        overrides: dict[str, Any] = None,
                        training: bool = False,
                        skip_gui_setup: bool = False) -> 'LaunchConfig':
        """Parses the command line arguments for the GUI application."""
        parser: ArgumentParser = ArgumentParser(prog='NeRFICG GUI',
                                                description='Graphical User Interface for the NeRFICG framework.')
        config = LaunchConfig(_training=training)
        disable_cmd_args = disable_cmd_args or []
        overrides = overrides or {}

        if not training:
            parser.add_argument('--ignore-cache-config', '-i', action='store_true', required=False,
                                help='Ignore cached configuration and use default values instead.')
            parser.add_argument('--skip-gui-setup', '-s', action='store_true', required=False,
                                help='Launch viewer without GUI setup. If no training config or checkpoint are '
                                     'specified, the most recent run is loaded.')
            parser.add_argument('--model-dir', '-m', type=Path, required=False,
                                help='Load the configuration and latest checkpoint from a trained model directory, '
                                     'i.e. a directory containing the .yaml config file and a checkpoints directory.',
                                metavar='path/to/model_dir')
        parser.add_argument('--verbose', '-v', action='store_true', required=False,
                            help='Log additional information during startup.')

        parser = addArgumentsFromFields(parser, LaunchConfig, disable_cmd_args)
        args = parser.parse_known_args()[0]

        if args.verbose:
            Logger.setMode(Logger.MODE_DEBUG)
        else:
            Logger.setMode(Logger.MODE_NORMAL)

        if config_name is None:
            try:
                cfg_name_part = overrides.get('checkpoint_path', None)
                cfg_name_part = cfg_name_part or getattr(args, 'model_dir', None)
                cfg_name_part = cfg_name_part or getattr(args, 'checkpoint_path', None)
                cfg_name_part = cfg_name_part or overrides.get('training_config_path', None)
                cfg_name_part = cfg_name_part or getattr(args, 'training_config_path', None)

                config_name = md5(str(Path(cfg_name_part).absolute()))
            except TypeError:
                config_name = None

        if config_name is not None and not getattr(args, 'ignore_cache_config', False):
            config.overrideFromConfig(config_name)

        config.overrideFromCmdArgs(args)
        config.applyOverrides(overrides)
        if ((config.training_config_path is None) ^ (config.checkpoint_path is None)) and not training:
            config.guessMissingPath()

        setup = True
        if skip_gui_setup or getattr(args, 'skip_gui_setup', False):
            setup = False
            if not config.validate():
                setup = True

        if setup:
            setup_gui = SetupGUI()
            config = setup_gui.configure(config)

        config.fillConfigPath()
        config.save()

        return config

    @property
    def is_training(self):
        """Returns whether the GUI is running in training mode."""
        return self._training
