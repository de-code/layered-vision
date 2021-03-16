from layered_vision.config import AppConfig, LayerConfig
from layered_vision.resolved_config import ResolvedAppConfig


class TestResolvedAppConfig:
    def test_should_connect_input_to_output_layer(self):
        resolved_app_config = ResolvedAppConfig(AppConfig([
            LayerConfig({
                'id': 'in',
                'input_path': 'input_path_1'
            }),
            LayerConfig({
                'id': 'out',
                'output_path': 'output_path_1'
            })
        ]))
        assert resolved_app_config.layer_by_id['in'].get('input_path') == (
            'input_path_1'
        )
        assert resolved_app_config.layer_by_id['out'].get('output_path') == (
            'output_path_1'
        )
        assert resolved_app_config.layer_by_id['out'].input_layer_ids == ['in']

    def test_should_connect_input_to_branch_layers(self):
        resolved_app_config = ResolvedAppConfig(AppConfig([
            LayerConfig({
                'id': 'in',
                'input_path': 'input_path_1'
            }),
            LayerConfig({
                'id': 'branches',
                'branches': [{
                    'layers': [{
                        'id': 'branch_1_layer_1',
                        'filter': 'dummy'
                    }]
                }]
            }),
            LayerConfig({
                'id': 'out',
                'output_path': 'output_path_1'
            })
        ]))
        assert resolved_app_config.layer_by_id['in'].get('input_path') == (
            'input_path_1'
        )
        assert resolved_app_config.layer_by_id['out'].get('output_path') == (
            'output_path_1'
        )
        assert resolved_app_config.layer_by_id['branch_1_layer_1'].input_layer_ids == ['in']
        assert resolved_app_config.layer_by_id['out'].input_layer_ids == ['branches']

    def test_should_skip_disabled_layers(self):
        resolved_app_config = ResolvedAppConfig(AppConfig([
            LayerConfig({
                'id': 'in1',
                'enabled': True,
                'input_path': 'input_path_1'
            }),
            LayerConfig({
                'id': 'in2',
                'enabled': False,
                'input_path': 'input_path_1'
            }),
            LayerConfig({
                'id': 'out',
                'output_path': 'output_path_1'
            })
        ]))
        assert resolved_app_config.layer_by_id['in1']
        assert not resolved_app_config.layer_by_id.get('in2')
        assert resolved_app_config.layer_by_id['out'].input_layer_ids == ['in1']
