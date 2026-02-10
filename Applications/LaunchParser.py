"""Applications/LaunchParser.py: Argument / Override definitions and parsing for the GUI application."""

from argparse import ArgumentParser, Namespace
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml
try:
    from matplotlib import font_manager
except ImportError:
    font_manager = None

from ICGui.Applications import Launcher
from ICGui.Backend import FontSpec
from ICGui.State import LaunchConfig
from ICGui.util.Dataclasses import add_arguments_from_fields, apply_overrides
import Framework
from Logging import Logger


def apply_launch_config(cfg: LaunchConfig):
    """Overrides the configuration from the launch config path.
    cfg.launch_config_path must be set before."""
    if not cfg.launch_config_path.is_file():
        return

    try:
        with open(cfg.launch_config_path, 'r') as config_file:
            yaml_dict: dict[str, Any] = yaml.safe_load(config_file)
    except (OSError, IOError, yaml.YAMLError) as exc:
        raise Framework.GUIError(f'Invalid config file path: "{cfg.launch_config_path}"') from exc

    # Filter out invalid keys
    for config_field in fields(cfg):
        if (cfg.is_training and config_field.metadata is not None
                and config_field.metadata.get('training_disabled', False)):
            yaml_dict.pop(config_field.name)
            continue

    apply_overrides(cfg, yaml_dict)


def apply_command_line_args(cfg: LaunchConfig, args: Namespace):
    """Overrides the configuration from command line arguments."""
    if getattr(args, 'model_dir', None) is not None:
        cfg.infer_from_model_dir(args.model_dir)
    apply_overrides(cfg, vars(args))


def apply_resets(cfg: LaunchConfig):
    """Resets fields to default values if invalid configuration is detected."""
    if font_manager is None:
        Logger.log_warning('matplotlib is not installed, resetting font to default.')
        cfg.font = FontSpec(name='Default')  # Reset to default font


def from_command_line(overrides: dict[str, Any] = None, argparse_ignore: list[str] = None,
                      training: bool = False, skip_gui_setup: bool = False) -> 'LaunchConfig':
    """Parses the command line arguments for the GUI application.
    Args:
        overrides: A dictionary of overrides to apply to the configuration.
        argparse_ignore: A list of field names to not expose as command line arguments.
        training: Whether the GUI is running in training mode.
        skip_gui_setup: Whether to attempt skipping the GUI setup dialog
            (requires passing at least the model directory or training config + checkpoint path)
    """
    parser: ArgumentParser = ArgumentParser(prog='NeRFICG GUI',
                                            description='Graphical User Interface for the NeRFICG framework.')
    cfg = LaunchConfig(is_training=training)

    disable_cmd_args = argparse_ignore or []
    overrides = overrides or {}

    # Inference only arguments
    if not training:
        parser.add_argument('--ignore-cache-config', '-i', action='store_true', required=False,
                            help='Ignore cached configuration and use default values instead.')
        parser.add_argument('--model-dir', '-m', type=Path, required=False,
                            help='Load the configuration and latest checkpoint from a trained model directory, '
                                 'i.e. a directory containing the .yaml config file and a checkpoints directory.',
                            metavar='path/to/model_dir')

    # General arguments
    parser.add_argument('--skip-gui-setup', '-s', action='store_true', required=False,
                        help='Launch viewer without GUI setup. If no training config or checkpoint are '
                             'specified, the most recent run is loaded.')
    parser.add_argument('--verbose', '-v', action='store_true', required=False,
                        help='Log additional information during startup.')

    # Add arguments from LaunchConfig dataclass
    parser = add_arguments_from_fields(parser, LaunchConfig, disable_cmd_args)

    args = parser.parse_known_args()[0]
    if args.verbose:
        Logger.set_mode(Logger.MODE_DEBUG)
    else:
        Logger.set_mode(Logger.MODE_NORMAL)

    # Override from config file, command line, and overrides in that order
    apply_launch_config(cfg)
    apply_command_line_args(cfg, args)
    apply_overrides(cfg, overrides)
    apply_resets(cfg)
    if not training:
        cfg.guess_missing_paths()  # Guess missing paths if exactly one is provided

    if (skip_gui_setup or getattr(args, 'skip_gui_setup', False)) and cfg.valid:
        # All fields valid, no manual intervention required
        return cfg

    # Manual GUI setup requested or required
    launcher = Launcher()
    launcher.configure(cfg)
    cfg.save_to_disk()

    return cfg
