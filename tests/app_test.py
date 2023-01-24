from io import StringIO
from pathlib import Path
from typing import Tuple, Union

import pytest
import yaml
import cv2
import numpy as np

from layered_vision.utils.image import ImageSize, ImageArray
from layered_vision.app import LayeredVisionApp


DEFAULT_IMAGE_SIZE = ImageSize(height=4, width=6)
RGB_RED = (255, 0, 0)
RGB_GREEN = (0, 255, 0)
RGB_BLUE = (0, 0, 255)


def _quote_path(path: Union[str, Path]) -> str:
    return repr(str(path))


def _get_yaml_text(data) -> str:
    out = StringIO()
    yaml.safe_dump(data, out)
    return out.getvalue()


def create_solid_color_image(image_size: ImageSize, rgb: Tuple[int, int, int]) -> ImageArray:
    image = np.zeros((image_size.height, image_size.width, 3), np.uint8)
    image[:] = rgb
    return image


def read_image(path: Path) -> ImageArray:
    bgr_image = cv2.imread(str(path))
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return rgb_image


def save_image(path: Path, image: ImageArray):
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr_image)
    return path


RED_IMAGE = create_solid_color_image(DEFAULT_IMAGE_SIZE, RGB_RED)
GREEN_IMAGE = create_solid_color_image(DEFAULT_IMAGE_SIZE, RGB_GREEN)
BLUE_IMAGE = create_solid_color_image(DEFAULT_IMAGE_SIZE, RGB_BLUE)


@pytest.fixture(name='red_image_file')
def _red_image_file(tmp_path: Path) -> Path:
    return save_image(tmp_path / 'red.png', RED_IMAGE)


@pytest.fixture(name='green_image_file')
def _green_image_file(tmp_path: Path) -> Path:
    return save_image(tmp_path / 'green.png', GREEN_IMAGE)


@pytest.fixture(name='blue_image_file')
def _blue_image_file(tmp_path: Path) -> Path:
    return save_image(tmp_path / 'blue.png', BLUE_IMAGE)


class TestApp:
    def test_should_configure_simple_input_output(self, tmp_path: Path):
        input_path = tmp_path / 'input.png'
        output_path = tmp_path / 'output.png'
        config_file = tmp_path / 'config.yml'
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
        self, tmp_path: Path, no_source: bool
    ):
        input_path = tmp_path / 'input.png'
        output_path = tmp_path / 'output.png'
        config_file = tmp_path / 'config.yml'
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

    def test_should_reload_input_source_if_changed(self, tmp_path: Path):
        input_path = tmp_path / 'input.png'
        input_path_2 = tmp_path / 'input_2.png'
        output_path = tmp_path / 'output.png'
        config_file = tmp_path / 'config.yml'
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


class TestAppEndToEnd:
    def test_should_copy_image(
        self, tmp_path: Path,
        red_image_file: Path
    ):
        output_path = tmp_path / 'output.png'
        config_file = tmp_path / 'config.yml'
        config = {
            'layers': [{
                'id': 'in',
                'input_path': str(red_image_file)
            }, {
                'id': 'out',
                'output_path': str(output_path)
            }]
        }
        config_file.write_text(_get_yaml_text(config))
        with LayeredVisionApp(str(config_file)) as app:
            app.run()
            output_image = read_image(output_path)
            np.testing.assert_array_equal(output_image, RED_IMAGE)

    def test_should_reload_config_image_input(
        self, tmp_path: Path,
        red_image_file: Path,
        green_image_file: Path
    ):
        output_path = tmp_path / 'output.png'
        config_file = tmp_path / 'config.yml'
        config = {
            'layers': [{
                'id': 'in',
                'input_path': str(red_image_file)
            }, {
                'id': 'out',
                'output_path': str(output_path)
            }]
        }
        config_file.write_text(_get_yaml_text(config))
        with LayeredVisionApp(str(config_file)) as app:
            app.run()

            config['layers'][0]['input_path'] = str(green_image_file)
            config_file.write_text(_get_yaml_text(config))
            app.reload_config()
            app.run()

            output_image = read_image(output_path)
            np.testing.assert_array_equal(output_image, GREEN_IMAGE)

    def test_should_reload_config_enabled_image_input(
        self, tmp_path: Path,
        red_image_file: Path,
        green_image_file: Path
    ):
        output_path = tmp_path / 'output.png'
        config_file = tmp_path / 'config.yml'
        config: dict = {
            'layers': [{
                'id': 'in',
                'input_path': str(red_image_file)
            }, {
                'id': 'in2',
                'input_path': str(green_image_file),
                'enabled': False
            }, {
                'id': 'out',
                'output_path': str(output_path)
            }]
        }
        config_file.write_text(_get_yaml_text(config))
        with LayeredVisionApp(str(config_file)) as app:
            app.run()

            config['layers'][1]['enabled'] = True
            config_file.write_text(_get_yaml_text(config))
            app.reload_config()
            app.run()

            output_image = read_image(output_path)
            np.testing.assert_array_equal(output_image, GREEN_IMAGE)

    def test_should_fallback_to_error_image_for_input(
        self, tmp_path: Path,
        red_image_file: Path
    ):
        output_path = tmp_path / 'output.png'
        config_file = tmp_path / 'config.yml'
        config: dict = {
            'layers': [{
                'id': 'on_error',
                'input_path': str(red_image_file)
            }, {
                'id': 'in',
                'input_path': str(tmp_path / 'invalid')
            }, {
                'id': 'out',
                'output_path': str(output_path)
            }]
        }
        config_file.write_text(_get_yaml_text(config))
        with LayeredVisionApp(str(config_file)) as app:
            app.run()
            output_image = read_image(output_path)
            np.testing.assert_array_equal(output_image, RED_IMAGE)

    def test_should_fallback_to_error_image_for_filter(
        self, tmp_path: Path,
        red_image_file: Path,
        green_image_file: Path
    ):
        output_path = tmp_path / 'output.png'
        config_file = tmp_path / 'config.yml'
        config: dict = {
            'layers': [{
                'id': 'on_error',
                'input_path': str(red_image_file),
                'repeat': True
            }, {
                'id': 'in',
                'input_path': str(green_image_file)
            }, {
                'id': 'filter',
                'filter': 'invalid'
            }, {
                'id': 'out',
                'output_path': str(output_path)
            }]
        }
        config_file.write_text(_get_yaml_text(config))
        with LayeredVisionApp(str(config_file)) as app:
            app.run(1)
            output_image = read_image(output_path)
            np.testing.assert_array_equal(output_image, RED_IMAGE)
