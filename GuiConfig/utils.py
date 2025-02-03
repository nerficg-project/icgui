# -- coding: utf-8 --

"""GuiConfig/utils.py: Utility functions for the launch configuration."""

import hashlib
from dataclasses import dataclass, fields


def md5(content: str):
    """Returns the MD5 hash of a string."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()


def addArgumentsFromFields(parser, dclass: type[dataclass], disable_cmd_args: list[str] = None):
    """Adds arguments to the parser from the fields of a dataclass."""
    for config_field in fields(dclass):
        if config_field.name.startswith('_'):
            continue

        arg_name = config_field.name.replace('_', '-')
        if arg_name in disable_cmd_args:
            continue

        disabled = config_field.metadata.get('argparse_disabled', False)
        if disabled:
            continue

        flags = config_field.metadata.get('flags', ())

        if config_field.type is bool:
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
        else:
            parser.add_argument(f'--{arg_name}', *flags, required=False, dest=config_field.name,
                                **config_field.metadata.get('argparse_kwargs', {}))

    return parser
