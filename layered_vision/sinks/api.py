import logging
import os
from contextlib import contextmanager
from functools import partial
from importlib import import_module
from typing import Callable, ContextManager, Dict, Iterator, Tuple

import cv2

from ..utils.path import parse_type_path
from ..utils.image import ImageArray, rgb_to_bgr
from ..utils.opencv import ShowImageSink


LOGGER = logging.getLogger(__name__)


T_OutputSink = Callable[[ImageArray], None]

T_OutputSinkFactory = Callable[..., ContextManager[T_OutputSink]]


class OutputTypes:
    WINDOW = 'window'
    V4L2 = 'v4l2'
    IMAGE_WRITER = 'image_writer'


def write_image_to(image_array: ImageArray, path: str):
    LOGGER.info('writing image to: %r', path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, rgb_to_bgr(image_array))


@contextmanager
def get_image_file_output_sink(path: str, **__) -> Iterator[T_OutputSink]:
    yield partial(write_image_to, path=path)


def is_v4l2_path(path: str) -> bool:
    return path.startswith("/dev/video")


def get_show_image_output_sink(
    *_, window_title: str = 'image', **__
) -> ContextManager[T_OutputSink]:
    return ShowImageSink(window_title)


def get_output_type_for_path(path: str) -> str:
    if path == 'window':
        return OutputTypes.WINDOW
    if is_v4l2_path(path):
        return OutputTypes.V4L2
    return OutputTypes.IMAGE_WRITER


def get_output_type_and_path(path: str, **kwargs) -> Tuple[str, str]:
    output_type = kwargs.get('type')
    if output_type:
        return output_type, path
    output_type, path = parse_type_path(path)
    if not output_type:
        output_type = get_output_type_for_path(path)
    return output_type, path


OUTPUT_SINK_FACTORY_BY_TYPE: Dict[str, T_OutputSinkFactory] = {
    OutputTypes.WINDOW: get_show_image_output_sink,
    OutputTypes.IMAGE_WRITER: get_image_file_output_sink
}


def get_image_output_sink_for_output_type_and_path(
    output_type: str, path: str, **kwargs
) -> ContextManager[T_OutputSink]:
    sink_factory = OUTPUT_SINK_FACTORY_BY_TYPE.get(output_type)
    if sink_factory is None:
        sink_module = import_module('layered_vision.sinks.%s' % output_type)
        sink_factory = getattr(sink_module, 'OUTPUT_SINK_FACTORY')
        OUTPUT_SINK_FACTORY_BY_TYPE[output_type] = sink_factory
    if sink_factory is not None:
        return sink_factory(path, **kwargs)
    raise ValueError('invalid output type: %r' % output_type)


def get_image_output_sink_for_path(path: str, **kwargs) -> ContextManager[T_OutputSink]:
    output_type, path = get_output_type_and_path(path, **kwargs)
    return get_image_output_sink_for_output_type_and_path(
        output_type, path, **kwargs
    )
