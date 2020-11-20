from pathlib import Path

import cv2

from layered_vision.cli import main


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
