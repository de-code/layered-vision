from pathlib import Path
from unittest.mock import patch

import pytest

from layered_vision.sources import get_image_source_for_path
from layered_vision.utils.image import ImageArray
from layered_vision import app
from layered_vision.cli import main


EXAMPLE_CONFIG_DIR = './example-config'


class CapturingOutputSink:
    def __init__(self, limit: int = 1):
        self.images = []
        self.limit = limit

    def __enter__(self):
        return self

    def __exit__(self, *_, **__):
        pass

    def __call__(self, image_array: ImageArray):
        self.images.append(image_array)
        if len(self.images) >= self.limit:
            raise KeyboardInterrupt('limit reach')


class GetCapturingOutputSink:
    def __init__(self, **kwargs):
        self.sinks = []
        self.sink_kwargs = kwargs

    @property
    def last_sink(self):
        return self.sinks[-1]

    def __call__(self, path: str):
        self.sinks.append(CapturingOutputSink(
            **self.sink_kwargs
        ))
        return self.last_sink


@pytest.fixture(name='get_image_output_sink_for_path_mock', autouse=True)
def _get_image_output_sink_for_path_mock():
    target = GetCapturingOutputSink()
    with patch.object(app, 'get_image_output_sink_for_path', target) as mock:
        yield mock


def get_not_preloading_image_source_for_path_mock(path: str, **kwargs):
    kwargs['preload'] = False
    return get_image_source_for_path(path)


@pytest.fixture(name='get_image_source_for_path_mock', autouse=True)
def _get_image_source_for_path_mock():
    target = get_not_preloading_image_source_for_path_mock
    with patch.object(app, 'get_image_source_for_path', target) as mock:
        yield mock


class TestMain:
    def test_should_process_example(
            self,
            get_image_output_sink_for_path_mock: GetCapturingOutputSink):
        example_config_filename = 'display-video-bodypix-replace-background.yml'
        config_file = Path(EXAMPLE_CONFIG_DIR) / example_config_filename
        main(['start', '--config-file=%s' % config_file])
        sink = get_image_output_sink_for_path_mock.last_sink
        assert len(sink.images) > 0
