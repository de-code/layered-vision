import logging
from typing import (
    Callable, Dict, Iterable, List, Optional, Union, Any, TypeVar, Type
)

import yaml

from .utils.io import read_text


LOGGER = logging.getLogger(__name__)

T = TypeVar('T')

T_Value = Union[str, int, float, bool]


def parse_bool(value: str) -> bool:
    value_lower = value.lower()
    if value_lower == 'false':
        return False
    if value_lower == 'true':
        return True
    raise ValueError('invalid boolean value: %r' % value)


def parse_str_list(value: str) -> List[str]:
    value = value.strip()
    if not value:
        return []
    return [item.strip() for item in value.split(',')]


def get(
    props: dict,
    key: str,
    default_value: Any = None
) -> Optional[Any]:
    value = props.get(key)
    if value is None and default_value is not None:
        return default_value
    return value


def get_typed(
    props: dict,
    key: str,
    value_type: Type[T],
    default_value: T = None,
    parse_fn: Callable[[str], T] = None
) -> Optional[T]:
    value = props.get(key)
    if value is None and default_value is not None:
        return default_value
    if parse_fn is not None and isinstance(value, str):
        return parse_fn(value)
    if value is not None and not isinstance(value, value_type):
        return value_type(value)  # type: ignore[call-arg]
    return value


def get_bool(props, key: str, default_value: bool = None):
    return get_typed(props, key, bool, default_value, parse_bool)


class PropsConfig:
    def __init__(self, props: dict):
        self.props = props

    def get(self, key: str, default_value: T = None) -> Optional[T]:
        return get(self.props, key, default_value)

    def get_typed(
        self,
        key: str,
        value_type: Type[T],
        default_value: T = None,
        parse_fn: Callable[[str], T] = None
    ) -> Optional[T]:
        return get_typed(self.props, key, value_type, default_value, parse_fn)

    def get_str(self, key: str, default_value: str = None):
        return self.get_typed(key, str, default_value)

    def get_bool(self, key: str, default_value: bool = None):
        return self.get_typed(key, bool, default_value, parse_bool)

    def get_int(self, key: str, default_value: int = None):
        return self.get_typed(key, int, default_value)

    def get_float(self, key: str, default_value: float = None):
        return self.get_typed(key, float, default_value)

    def get_str_list(self, key: str, default_value: str = None):
        return self.get_typed(key, list, default_value, parse_str_list)

    def get_dict(self, key: str, default_value: dict = None) -> Optional[dict]:
        result = self.get(key, default_value=default_value)
        if result is None:
            return None
        if not isinstance(result, dict):
            raise AssertionError(
                f'dict value required for key={repr(key)}, but was: {repr(result)}'
            )
        return result

    def get_list(self, key: str, default_value: list = None) -> Optional[list]:
        result = self.get(key, default_value=default_value)
        if result is None:
            return None
        if not isinstance(result, list):
            raise AssertionError(
                f'list value required for key={repr(key)}, but was: {repr(result)}'
            )
        return result

    def __repr__(self):
        return '%s(props=%r)' % (
            type(self).__name__,
            self.props
        )

    def __eq__(self, other):
        if isinstance(other, PropsConfig):
            return self.props == other.props
        return super().__eq__(other)

    def __hash__(self):
        return hash(self.props)


class LayerConfig(PropsConfig):
    @staticmethod
    def from_json(data: dict) -> 'LayerConfig':
        return LayerConfig(props=data)


def _iter_find_nested_layer_props(parent: Union[dict, list, Any]) -> Iterable[dict]:
    if isinstance(parent, dict):
        for key, value in parent.items():
            if key == 'layers':
                yield from value
            yield from _iter_find_nested_layer_props(value)
    if isinstance(parent, list):
        for item in parent:
            yield from _iter_find_nested_layer_props(item)


class AppConfig:
    def __init__(self, layers: List[LayerConfig]):
        self.layers = layers

    @staticmethod
    def from_json(data: dict) -> 'AppConfig':
        LOGGER.debug('app config data: %r', data)
        return AppConfig(
            layers=[
                LayerConfig.from_json(layer_data)
                for layer_data in data.get('layers', [])
            ]
        )

    def iter_layers(self) -> Iterable[LayerConfig]:
        return self.layers

    def iter_flatten_layer_props(self) -> Iterable[dict]:
        for layer in self.layers:
            yield layer.props
            yield from _iter_find_nested_layer_props(layer.props)

    def __repr__(self):
        return '%s(layers=%r)' % (
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
    consumed_override_layer_ids = set()
    valid_layer_ids = set()
    for layer_config_props in app_config.iter_flatten_layer_props():
        layer_id = layer_config_props.get('id')
        if not layer_id:
            continue
        valid_layer_ids.add(layer_id)
        layer_override_map = override_map.get(layer_id)
        if not layer_override_map:
            continue
        for prop_name, value in layer_override_map.items():
            layer_config_props[prop_name] = value
        consumed_override_layer_ids.add(layer_id)
    unknown_override_layer_ids = set(override_map.keys()) - consumed_override_layer_ids
    if unknown_override_layer_ids:
        raise ValueError('invalid override layer ids: %s (valid ids are: %s)' % (
            unknown_override_layer_ids,
            valid_layer_ids
        ))
