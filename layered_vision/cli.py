import argparse
import logging
import re
import os
from abc import ABC, abstractmethod
from typing import Dict, List

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

# pylint: disable=wrong-import-position
# flake8: noqa: E402

from layered_vision.config import T_Value
from layered_vision.app import LayeredVisionApp


LOGGER = logging.getLogger(__name__)


class SubCommand(ABC):
    def __init__(self, name, description):
        self.name = name
        self.description = description

    @abstractmethod
    def add_arguments(self, parser: argparse.ArgumentParser):
        pass

    @abstractmethod
    def run(self, args: argparse.Namespace):
        pass


def add_common_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )


def parse_value_expression(value_str: str) -> T_Value:
    if value_str.lower() == 'false':
        return False
    if value_str.lower() == 'true':
        return True
    if re.match(r'^\d+$', value_str):
        return int(value_str)
    if re.match(r'^\d+\.\d+$', value_str):
        return float(value_str)
    return value_str


def parse_set_value(text: str) -> Dict[str, Dict[str, T_Value]]:
    try:
        key, value = text.split('=', maxsplit=1)
    except ValueError as exc:
        raise ValueError('value expected, format: <layer id>.<prop name>=<value>') from exc
    parsed_value = parse_value_expression(value)
    try:
        layer_id, prop_name = key.split('.', maxsplit=1)
    except ValueError as exc:
        raise ValueError('layer id expected, format: <layer id>.<prop name>=<value>') from exc
    return {layer_id: {prop_name: parsed_value}}


def get_merged_set_values(
    set_values: List[Dict[str, Dict[str, T_Value]]]
) -> Dict[str, Dict[str, T_Value]]:
    if not set_values:
        return {}
    result: Dict[str, Dict[str, T_Value]] = {}
    for set_value in set_values:
        for layer_id, props in set_value.items():
            for prop_name, value in props.items():
                result.setdefault(layer_id, {})[prop_name] = value
    return result


class StartSubCommand(SubCommand):
    def __init__(self):
        super().__init__("start", "Start Layered Vision")

    def add_arguments(self, parser: argparse.ArgumentParser):
        add_common_arguments(parser)
        parser.add_argument(
            "--config-file",
            default="config.yml",
            help="The path to the config file"
        )
        parser.add_argument(
            "--set",
            action='append',
            type=parse_set_value,
            help="Allows to override config options in the format: <layer id>.<prop name>=value"
        )

    def run(self, args: argparse.Namespace):  # pylint: disable=unused-argument
        override_map = get_merged_set_values(args.set)
        LOGGER.debug('override_map: %s', override_map)
        with LayeredVisionApp(config_path=args.config_file, override_map=override_map) as app:
            app.run()


SUB_COMMANDS: List[SubCommand] = [
    StartSubCommand()
]

SUB_COMMAND_BY_NAME: Dict[str, SubCommand] = {
    sub_command.name: sub_command for sub_command in SUB_COMMANDS
}


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True
    for sub_command in SUB_COMMANDS:
        sub_parser = subparsers.add_parser(
            sub_command.name, help=sub_command.description
        )
        sub_command.add_arguments(sub_parser)

    args = parser.parse_args(argv)
    return args


def run(args: argparse.Namespace):
    sub_command = SUB_COMMAND_BY_NAME[args.command]
    sub_command.run(args)


def main(argv: List[str] = None):
    args = parse_args(argv)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    LOGGER.debug("args: %s", args)
    run(args)


if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    main()
