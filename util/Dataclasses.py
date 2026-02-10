"""util/Dataclasses.py: Utilities for interacting with dataclasses through argparse."""

from argparse import ArgumentParser
from dataclasses import dataclass, fields, is_dataclass, Field
from typing import Any, Mapping, Union, get_args, get_origin
from types import UnionType

from ICGui.Backend.FontManager import FontSpec


def _add_argument(parser: ArgumentParser, config_field: Field, disable_cmd_args: list[str] = None) -> None:
    # Replace underscores with dashes for command line arguments
    arg_name = config_field.name.replace('_', '-')

    # If the argument is in the list of disabled command line arguments, skip it
    if arg_name in disable_cmd_args:
        return

    # Add the argument to the parser
    flags = config_field.metadata.get('flags', ())
    parser.add_argument(f'--{arg_name}', *flags, required=False, dest=config_field.name,
                        **config_field.metadata.get('argparse_kwargs', {}))


def _add_bool_argument(parser: ArgumentParser, config_field: Field, disable_cmd_args: list[str] = None):
    # Replace underscores with dashes for command line arguments
    arg_name = config_field.name.replace('_', '-')

    # If the argument is in the list of disabled command line arguments, skip it
    if arg_name in disable_cmd_args:
        return

    # Add the argument to the parser (one for enabling, one for disabling)
    flags = config_field.metadata.get('flags', ())
    help_raw, help_text = None, None
    if 'argparse_kwargs' in config_field.metadata \
            and 'help' in config_field.metadata['argparse_kwargs']:
        help_raw = config_field.metadata['argparse_kwargs']['help']
        del config_field.metadata['argparse_kwargs']['help']
        help_text = 'Force enable ' + help_raw
    parser.add_argument(f'--{arg_name}', *(flags if not config_field.default else ()), required=False,
                        dest=config_field.name, action='store_true', default=None, help=help_text,
                        **config_field.metadata.get('argparse_kwargs', {}))

    if help_raw is not None:
        help_text = 'Force disable ' + help_raw
    parser.add_argument(f'--no-{arg_name}', *(flags if config_field.default else ()), required=False,
                        dest=config_field.name, action='store_false', default=None, help=help_text,
                        **config_field.metadata.get('argparse_kwargs', {}))


def _add_font_argument(parser: ArgumentParser, config_field: Field, disable_cmd_args: list = None):
    # Replace underscores with dashes for command line arguments
    arg_name = config_field.name.replace('_', '-')

    # If the argument is in the list of disabled command line arguments, skip it
    if arg_name in disable_cmd_args:
        return

    # Add the argument to the parser
    flags = config_field.metadata.get('flags', ())
    parser.add_argument(f'--{arg_name}-family', *flags, required=False, dest=config_field.name,
                        type=str, metavar=f'{arg_name}-family',
                        **config_field.metadata.get('argparse_kwargs', {}).get('font-family', {}))
    parser.add_argument(f'--{arg_name}-size', *flags, required=False, dest=config_field.name,
                        type=int, metavar=f'{arg_name}-size',
                        **config_field.metadata.get('argparse_kwargs', {}).get('font-size', {}))


def add_arguments_from_fields(parser, dclass: type[dataclass], disable_cmd_args: list[str] = None):
    """Adds arguments to the parser from the fields of a dataclass."""
    for config_field in fields(dclass):
        # Ignore private fields, or those marked as disabled for the command line
        if config_field.name.startswith('_'):
            continue
        if config_field.metadata.get('argparse_disabled', False):
            continue

        if config_field.type is bool:
            _add_bool_argument(parser, config_field, disable_cmd_args)
        elif config_field.type is FontSpec:
            _add_font_argument(parser, config_field, disable_cmd_args)
        else:
            _add_argument(parser, config_field, disable_cmd_args)


    return parser


def apply_overrides(dclass: dataclass, overrides: Mapping[str, Any]):
    """Applies overrides to the configuration from a dictionary or other mapping type in-place."""
    own_fields = {f.name: f for f in fields(dclass)}
    for key in set(overrides.keys()).intersection(own_fields.keys()):
        override_val = overrides.get(key, None)
        if override_val is None:
            continue

        if is_dataclass(own_fields[key].type):
            # If the field is a dataclass, recursively apply overrides
            assert isinstance(override_val, Mapping)
            override_val = apply_overrides(own_fields[key].type(), override_val)
        else:
            # Convert type if necessary
            field_type = own_fields[key].type
            if get_origin(own_fields[key].type) in (Union, UnionType):
                # For Union types, assume the first type is the canonical type
                field_type = get_args(own_fields[key].type)[0]
            override_val = field_type(override_val)

        setattr(dclass, key, override_val)
        if own_fields[key].metadata.get('override_default', False):
            own_fields[key].default = getattr(dclass, key)  # Set default for reset button
    return dclass
