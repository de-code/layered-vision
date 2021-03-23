from enum import Enum
from typing import Dict, List, Optional

from layered_vision.config import AppConfig, LayerConfig


class ResolvedLayerType(Enum):
    INPUT_SOURCE = 1
    OUTPUT_SINK = 2
    FILTER = 3


def get_resolved_layer_type(layer_config: LayerConfig) -> ResolvedLayerType:
    if layer_config.get('filter'):
        return ResolvedLayerType.FILTER
    if layer_config.get('input_path'):
        return ResolvedLayerType.INPUT_SOURCE
    if layer_config.get('output_path'):
        return ResolvedLayerType.OUTPUT_SINK
    raise ValueError('unable to determine resolved layer type for: %s' % layer_config)


class ResolvedLayerConfig(LayerConfig):
    def __init__(self, props: dict, default_input_layer: 'ResolvedLayerConfig' = None):
        super().__init__(props)
        self.layer_id = props['id']
        self.input_layers: List[ResolvedLayerConfig] = []
        if default_input_layer and not self.is_no_source:
            self.input_layers.append(default_input_layer)
        self.resolved_layer_type = get_resolved_layer_type(self)

    @property
    def input_layer_ids(self):
        return [layer_config.layer_id for layer_config in self.input_layers]

    @property
    def is_enabled(self) -> bool:
        return self.get_bool('enabled', True)

    @property
    def is_no_source(self) -> bool:
        return self.get_bool('no_source', False)

    @property
    def is_output_layer(self) -> bool:
        return bool(self.get('output_path'))

    @property
    def is_input_layer(self) -> bool:
        return bool(self.get('input_path'))

    @property
    def is_filter_layer(self) -> bool:
        return bool(self.get('filter'))

    def __repr__(self):
        return '%s(props=%r, input_layer_ids=%s)' % (
            type(self).__name__,
            self.props,
            self.input_layer_ids
        )


class ResolvedAppConfig:
    def __init__(self, app_config: AppConfig):
        self.layer_by_id: Dict[str, ResolvedLayerConfig] = {}
        self.layers: List[ResolvedLayerConfig] = []
        self._add_layers_recursively(app_config.layers)

    def _add_resolved_layer(
        self,
        resolved_layer_config: ResolvedLayerConfig,
    ) -> ResolvedLayerConfig:
        self.layers.append(resolved_layer_config)
        self.layer_by_id[resolved_layer_config.layer_id] = resolved_layer_config
        return resolved_layer_config

    def _add_layers_recursively(
        self,
        layers: List[LayerConfig],
        default_id_prefix: str = '',
        default_input_layer: Optional[ResolvedLayerConfig] = None
    ) -> List[ResolvedLayerConfig]:
        resolved_layers: List[ResolvedLayerConfig] = []
        for layer_index, layer_config in enumerate(layers):
            if not layer_config.get_bool('enabled', True):
                continue
            resolved_layer_config = self._add_layer_recursively(
                layer_config,
                default_id=f'{default_id_prefix}l{layer_index}',
                default_input_layer=default_input_layer
            )
            if not resolved_layer_config.is_output_layer:
                default_input_layer = resolved_layer_config
                resolved_layers.append(resolved_layer_config)
        return resolved_layers

    def _add_layer_recursively(
        self,
        layer_config: LayerConfig,
        default_id: str,
        default_input_layer: Optional[ResolvedLayerConfig]
    ) -> ResolvedLayerConfig:
        layer_id = layer_config.get('id') or default_id
        branches = layer_config.get('branches')
        if branches:
            return self._add_branches_recursively(
                layer_config,
                layer_id=layer_id,
                default_input_layer=default_input_layer
            )
        resolved_layer_config = ResolvedLayerConfig({
            **layer_config.props,
            'id': layer_id
        }, default_input_layer=default_input_layer)
        return self._add_resolved_layer(resolved_layer_config)

    def _add_branch_recursively(
        self,
        layer_config: LayerConfig,
        default_id: str,
        default_input_layer: Optional[ResolvedLayerConfig]
    ):
        layers: Optional[List[dict]] = layer_config.get_list('layers')
        assert layers
        resolved_layers = self._add_layers_recursively(
            [LayerConfig(props) for props in layers],
            default_id_prefix=default_id,
            default_input_layer=default_input_layer
        )
        if not resolved_layers:
            return None
        return resolved_layers[-1]

    def _add_branches_recursively(
        self,
        layer_config: LayerConfig,
        layer_id: str,
        default_input_layer: ResolvedLayerConfig
    ):
        branches: Optional[List[dict]] = layer_config.get_list('branches')
        assert branches
        branch_layers = []
        for branch_index, branch_config in enumerate(branches):
            branch_layer = self._add_branch_recursively(
                LayerConfig(branch_config),
                default_id=f'{layer_id}b{branch_index}',
                default_input_layer=default_input_layer
            )
            if branch_layer:
                branch_layers.append(branch_layer)
        assert branch_layers
        if len(branch_layers) == 1:
            return branch_layers[0]
        resolved_layer_config = ResolvedLayerConfig({
            **{key: value for key, value in layer_config.props.items() if key != 'branches'},
            'id': layer_id,
            'filter': 'composite'
        })
        resolved_layer_config.input_layers = branch_layers
        return self._add_resolved_layer(resolved_layer_config)

    def __repr__(self):
        return '%s(layers=%r)' % (
            type(self).__name__,
            self.layers
        )
