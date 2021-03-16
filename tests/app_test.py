from io import StringIO
from pathlib import Path
from typing import Union

import pytest
import yaml

from layered_vision.app import LayeredVisionApp


def _quote_path(path: Union[str, Path]) -> str:
    return repr(str(path))


def _get_yaml_text(data) -> str:
    out = StringIO()
    yaml.safe_dump(data, out)
    return out.getvalue()


class TestMain:
    def test_should_configure_simple_input_output(self, temp_dir: Path):
        input_path = temp_dir / 'input.png'
        output_path = temp_dir / 'output.png'
        config_file = temp_dir / 'config.yml'
        config_file.write_text(
            '''
            layers:
            - id: in
              input_path: {input_path}
            - id: out
              output_path: {output_path}
            '''.format(
                input_path=_quote_path(input_path),
                output_path=_quote_path(output_path)
            )
        )
        with LayeredVisionApp(str(config_file)) as app:
            assert len(app.output_runtime_layers) == 1
            assert app.output_runtime_layers[0].layer_id == 'out'
            assert len(app.output_runtime_layers[0].source_layers) == 1
            assert app.output_runtime_layers[0].source_layers[0].layer_id == 'in'

    @pytest.mark.parametrize("no_source", (False, True))
    def test_should_ignore_extra_input_depending_on_no_source(
        self, temp_dir: Path, no_source: bool
    ):
        input_path = temp_dir / 'input.png'
        output_path = temp_dir / 'output.png'
        config_file = temp_dir / 'config.yml'
        config_file.write_text(
            '''
            layers:
            - id: in_ignored
              input_path: {input_path}
            - id: in
              input_path: {input_path}
              no_source: {no_source}
            - id: out
              output_path: {output_path}
            '''.format(
                input_path=_quote_path(input_path),
                output_path=_quote_path(output_path),
                no_source=no_source
            )
        )
        with LayeredVisionApp(str(config_file)) as app:
            assert len(app.output_runtime_layers) == 1
            assert app.output_runtime_layers[0].layer_id == 'out'
            assert len(app.output_runtime_layers[0].source_layers) == 1
            assert app.output_runtime_layers[0].source_layers[0].layer_id == 'in'
            if no_source:
                assert not app.output_runtime_layers[0].source_layers[0].source_layers
            else:
                assert app.output_runtime_layers[0].source_layers[0].source_layers

    def test_should_reload_input_source_if_changed(self, temp_dir: Path):
        input_path = temp_dir / 'input.png'
        input_path_2 = temp_dir / 'input_2.png'
        output_path = temp_dir / 'output.png'
        config_file = temp_dir / 'config.yml'
        config = {
            'layers': [{
                'id': 'in',
                'input_path': str(input_path)
            }, {
                'id': 'out',
                'output_path': str(output_path)
            }]
        }
        config_file.write_text(_get_yaml_text(config))
        with LayeredVisionApp(str(config_file)) as app:
            input_layer = app.layer_by_id['in']
            output_layer = app.layer_by_id['out']
            assert input_layer.layer_config.get('input_path') == str(input_path)
            config['layers'][0]['input_path'] = str(input_path_2)
            config_file.write_text(_get_yaml_text(config))
            app.reload_config()
            assert app.layer_by_id['in'].layer_config.get('input_path') == str(input_path_2)
            assert app.layer_by_id['out'] == output_layer
