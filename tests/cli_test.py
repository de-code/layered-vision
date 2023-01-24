from pathlib import Path
from typing import Union

import cv2

import pytest

from layered_vision.cli import (
    parse_value_expression,
    parse_set_value,
    get_merged_set_values,
    main
)


EXAMPLE_IMAGE_URL = (
    r'https://raw.githubusercontent.com/numpy/numpy'
    r'/v1.20.1/branding/logo/logomark/numpylogoicon.png'
)


def _quote_path(path: Union[str, Path]) -> str:
    return repr(str(path))


def _load_image(path: Union[str, Path]):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError('failed to load image: %r' % path)
    return image


class TestParseValueExpression:
    def test_should_parse_str(self):
        assert parse_value_expression('abc') == 'abc'

    def test_should_parse_int(self):
        assert parse_value_expression('30') == 30

    def test_should_parse_float(self):
        assert parse_value_expression('30.1') == 30.1

    def test_should_parse_false(self):
        assert parse_value_expression('false') is False

    def test_should_parse_true(self):
        assert parse_value_expression('true') is True


class TestParseSetValue:
    def test_should_parse_simple_expression(self):
        assert parse_set_value('in.input_path=/path/to/input') == {
            'in': {
                'input_path': '/path/to/input'
            }
        }

    def test_should_parse_int_value(self):
        assert parse_set_value('in.fps=30') == {
            'in': {
                'fps': 30
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
    def test_should_copy_source_to_target_image(self, tmp_path: Path):
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
                input_path=_quote_path(EXAMPLE_IMAGE_URL),
                output_path=_quote_path(output_path)
            )
        )
        main(['start', '--config-file=%s' % config_file])
        image = _load_image(output_path)
        height, width, *_ = image.shape
        assert width > 0
        assert height > 0

    def test_should_copy_and_resize_source_to_target_image(self, tmp_path: Path):
        output_path = tmp_path / 'output.png'
        config_file = tmp_path / 'config.yml'
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

    def test_should_copy_to_multiple_outputs(self, tmp_path: Path):
        output_path_1 = tmp_path / 'output_1.png'
        output_path_2 = tmp_path / 'output_2.png'
        config_file = tmp_path / 'config.yml'
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

    def test_should_be_able_to_replace_input_and_output_path(self, tmp_path: Path):
        output_path = tmp_path / 'output.png'
        config_file = tmp_path / 'config.yml'
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
