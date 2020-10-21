from pathlib import Path

from layered_vision.app import LayeredVisionApp


class TestMain:
    def test_should_configure_simple_input_output(self, temp_dir: Path):
        input_path = temp_dir / 'input.png'
        output_path = temp_dir / 'output.png'
        config_file = temp_dir / 'config.yml'
        config_file.write_text(
            '''
            layers:
            - id: in
              input_path: "{input_path}"
            - id: out
              output_path: "{output_path}"
            '''.format(
                input_path=input_path,
                output_path=output_path
            )
        )
        with LayeredVisionApp(str(config_file)) as app:
            assert len(app.output_runtime_layers) == 1
            assert app.output_runtime_layers[0].layer_id == 'out'
            assert len(app.output_runtime_layers[0].source_layers) == 1
            assert app.output_runtime_layers[0].source_layers[0].layer_id == 'in'

    def test_should_ignore_extra_input(self, temp_dir: Path):
        input_path = temp_dir / 'input.png'
        output_path = temp_dir / 'output.png'
        config_file = temp_dir / 'config.yml'
        config_file.write_text(
            '''
            layers:
            - id: in_ignored
              input_path: "{input_path}"
            - id: in
              input_path: "{input_path}"
            - id: out
              output_path: "{output_path}"
            '''.format(
                input_path=input_path,
                output_path=output_path
            )
        )
        with LayeredVisionApp(str(config_file)) as app:
            assert len(app.output_runtime_layers) == 1
            assert app.output_runtime_layers[0].layer_id == 'out'
            assert len(app.output_runtime_layers[0].source_layers) == 1
            assert app.output_runtime_layers[0].source_layers[0].layer_id == 'in'
