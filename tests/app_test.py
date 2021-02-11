from pathlib import Path

import pytest

from layered_vision.app import LayeredVisionApp


def _quote_path(path: str) -> str:
    return repr(str(path))


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
