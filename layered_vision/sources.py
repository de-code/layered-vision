import logging
import os
from contextlib import contextmanager
from itertools import cycle
from typing import ContextManager, Iterable

import cv2

from .utils.image import resize_image_to, ImageSize, ImageArray, bgr_to_rgb
from .utils.io import get_file, strip_url_suffix
from .utils.opencv import get_video_image_source, get_webcam_image_source


LOGGER = logging.getLogger(__name__)


T_ImageSource = ContextManager[Iterable[ImageArray]]


@contextmanager
def get_simple_image_source(
    path: str,
    image_size: ImageSize = None,
    repeat: bool = None,
    **_
) -> T_ImageSource:
    local_image_path = get_file(path)
    LOGGER.debug('local_image_path: %r', local_image_path)
    bgr_image_array = cv2.imread(local_image_path)
    if bgr_image_array is None:
        raise IOError('failed to load image: %r' % local_image_path)
    image_array = bgr_to_rgb(bgr_image_array)
    if image_size is not None:
        image_array = resize_image_to(image_array, image_size)
    image_array_iterable = [image_array]
    if repeat:
        image_array_iterable = cycle(image_array_iterable)
    yield image_array_iterable


def is_webcam_path(path: str) -> bool:
    return path.startswith('/dev/video')


def is_video_path(path: str) -> bool:
    ext = os.path.splitext(os.path.basename(strip_url_suffix(path)))[-1]
    LOGGER.debug('ext: %s', ext)
    return ext.lower() in {'.webm', '.mkv', '.mp4'}


def get_image_source_for_path(path: str, **kwargs) -> T_ImageSource:
    if is_webcam_path(path):
        return get_webcam_image_source(path, **kwargs)
    if is_video_path(path):
        return get_video_image_source(path, **kwargs)
    return get_simple_image_source(path, **kwargs)
