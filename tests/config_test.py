import pytest

from layered_vision.config import (
    AppConfig,
    LayerConfig,
    apply_config_override_map
)


class TestApplyConfigOverrideMap:
    def test_should_not_apply_empty_override_map(self):
        app_config = AppConfig(layers=[
            LayerConfig({'id': 'id1', 'prop': 'value1'})
        ])
        apply_config_override_map(app_config, {})
        assert app_config.layers[0].get('prop') == 'value1'

    def test_should_raise_exception_if_id_was_not_found(self):
        app_config = AppConfig(layers=[
            LayerConfig({'id': 'id1', 'prop': 'value1'})
        ])
        with pytest.raises(ValueError):
            apply_config_override_map(app_config, {'other-id': {'prop': 'new-value'}})

    def test_should_override_prop_of_root_layer(self):
        app_config = AppConfig(layers=[
            LayerConfig({'id': 'id1', 'prop': 'value1'})
        ])
        apply_config_override_map(app_config, {'id1': {'prop': 'new-value'}})
        assert app_config.layers[0].get('prop') == 'new-value'

    def test_should_add_prop_to_root_layer(self):
        app_config = AppConfig(layers=[
            LayerConfig({'id': 'id1', 'prop': 'value1'})
        ])
        apply_config_override_map(app_config, {'id1': {'new-prop': 'new-value'}})
        assert app_config.layers[0].get('prop') == 'value1'
        assert app_config.layers[0].get('new-prop') == 'new-value'

    def test_should_override_prop_of_nested_layer(self):
        app_config = AppConfig(layers=[
            LayerConfig({
                'id': 'id1',
                'branches': [{
                    'layers': [{
                        'id': 'nested',
                        'prop': 'value1'
                    }]
                }]
            })
        ])
        apply_config_override_map(app_config, {'nested': {'prop': 'new-value'}})
        assert app_config.layers[0].get('branches')[0]['layers'][0]['prop'] == 'new-value'
