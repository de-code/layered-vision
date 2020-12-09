from pathlib import Path

import cv2

import pytest

from layered_vision.cli import (
    parse_set_value,
    get_merged_set_values,
    main
)


EXAMPLE_IMAGE_URL = (
    r'https://github.com/numpy/numpy/raw/master/branding/logo/logomark/numpylogoicon.png'
)


def _quote_path(path: str) -> str:
    return repr(str(path))


def _load_image(path: str):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError('failed to load image: %r' % path)
    return image


class TestParseSetValue:
    def test_should_parse_simple_expression(self):
        assert parse_set_value('in.input_path=/path/to/input') == {
            'in': {
                'input_path': '/path/to/input'
            }
        }

    def test_should_fail_with_missing_value(self):
        with pytest.raises(ValueError):
            assert parse_set_value('in.input_path')

    def test_should_fail_with_missing_layer_id(self):
        with pytest.raises(ValueError):
            assert parse_set_value('input_path=/path/to/input')


class TestGetMergedSetValues:
    def test_should_merge_properties_with_same_layer_id(self):
        assert get_merged_set_values([
            {'id1': {'prop1': 'value1'}},
            {'id1': {'prop2': 'value2'}}
        ]) == {
            'id1': {
                'prop1': 'value1',
                'prop2': 'value2'
            }
        }

    def test_should_merge_properties_with_different_layer_id(self):
        assert get_merged_set_values([
            {'id1': {'prop1': 'value1'}},
            {'id2': {'prop2': 'value2'}}
        ]) == {
            'id1': {'prop1': 'value1'},
            'id2': {'prop2': 'value2'}
        }


class TestMain:
    def test_should_copy_source_to_target_image(self, temp_dir: Path):
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
                input_path=_quote_path(EXAMPLE_IMAGE_URL),
                output_path=_quote_path(output_path)
            )
        )
        main(['start', '--config-file=%s' % config_file])
        image = _load_image(output_path)
        height, width, *_ = image.shape
        assert width > 0
        assert height > 0

    def test_should_copy_and_resize_source_to_target_image(self, temp_dir: Path):
        output_path = temp_dir / 'output.png'
        config_file = temp_dir / 'config.yml'
        config_file.write_text(
            '''
            layers:
            - id: in
              input_path: {input_path}
              width: 320
              height: 200
            - id: out
              output_path: {output_path}
            '''.format(
                input_path=_quote_path(EXAMPLE_IMAGE_URL),
                output_path=_quote_path(output_path)
            )
        )
        main(['start', '--config-file=%s' % config_file])
        image = _load_image(output_path)
        height, width, *_ = image.shape
        assert (width, height) == (320, 200)

    def test_should_copy_to_multiple_outputs(self, temp_dir: Path):
        output_path_1 = temp_dir / 'output_1.png'
        output_path_2 = temp_dir / 'output_2.png'
        config_file = temp_dir / 'config.yml'
        config_file.write_text(
            '''
            layers:
            - id: in
              input_path: {input_path}
              width: 320
              height: 200
            - id: out_1
              output_path: {output_path_1}
            - id: out_2
              output_path: {output_path_2}
            '''.format(
                input_path=_quote_path(EXAMPLE_IMAGE_URL),
                output_path_1=_quote_path(output_path_1),
                output_path_2=_quote_path(output_path_2)
            )
        )
        main(['start', '--config-file=%s' % config_file])
        for output_path in [output_path_1, output_path_2]:
            image = _load_image(output_path)
            height, width, *_ = image.shape
            assert (width, height) == (320, 200)

    def test_should_be_able_to_replace_input_and_output_path(self, temp_dir: Path):
        output_path = temp_dir / 'output.png'
        config_file = temp_dir / 'config.yml'
        config_file.write_text(
            '''
            layers:
            - id: in
              input_path: "dummy"
            - id: out
              output_path: "dummy"
            '''
        )
        main([
            'start',
            '--config-file=%s' % config_file,
            '--set',
            'in.input_path=%s' % EXAMPLE_IMAGE_URL,
            '--set',
            'out.output_path=%s' % output_path
        ])
        image = _load_image(output_path)
        height, width, *_ = image.shape
        assert width > 0
        assert height > 0
