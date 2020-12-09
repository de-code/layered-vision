import logging
from typing import Dict, Iterable, List

import yaml

from .utils.io import read_text


LOGGER = logging.getLogger(__name__)


class LayerConfig:
    def __init__(self, props: dict):
        self.props = props

    @staticmethod
    def from_json(data: dict) -> 'LayerConfig':
        return LayerConfig(props=data)

    def get(self, key: str):
        return self.props.get(key)

    def __repr__(self):
        return '%s(props=%r)' % (
            type(self).__name__,
            self.props
        )


class AppConfig:
    def __init__(self, layers: List[LayerConfig]):
        self.layers = layers

    @staticmethod
    def from_json(data: dict) -> 'AppConfig':
        LOGGER.debug('app config data: %r', data)
        return AppConfig(layers=[
            LayerConfig.from_json(layer_data)
            for layer_data in data.get('layers', [])
        ])

    def iter_layers(self) -> Iterable[LayerConfig]:
        return self.layers

    def __repr__(self):
        return '%s(layer=%r)' % (
            type(self).__name__,
            self.layers
        )


def load_raw_config(config_path: str) -> dict:
    return yaml.safe_load(read_text(config_path))


def load_config(config_path: str) -> AppConfig:
    return AppConfig.from_json(load_raw_config(config_path))


def apply_config_override_map(app_config: AppConfig, override_map: Dict[str, Dict[str, str]]):
    if not override_map:
        return
    for layer_config in app_config.iter_layers():
        layer_id = layer_config.get('id')
        if not layer_id:
            continue
        layer_override_map = override_map.get(layer_id)
        if not layer_override_map:
            continue
        for prop_name, value in layer_override_map.items():
            layer_config.props[prop_name] = value
