import os
import logging
from contextlib import contextmanager
from functools import partial
from typing import Callable

import cv2

from .utils.image import ImageArray, rgb_to_bgr
from .utils.opencv import ShowImageSink


LOGGER = logging.getLogger(__name__)


T_OutputSink = Callable[[ImageArray], None]


def write_image_to(image_array: ImageArray, path: str):
    LOGGER.info('writing image to: %r', path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, rgb_to_bgr(image_array))


@contextmanager
def get_image_file_output_sink(path: str) -> T_OutputSink:
    yield partial(write_image_to, path=path)


def get_show_image_output_sink() -> T_OutputSink:
    return ShowImageSink('image')


def get_image_output_sink_for_path(path: str) -> T_OutputSink:
    if path == 'window':
        return get_show_image_output_sink()
    return get_image_file_output_sink(path)
