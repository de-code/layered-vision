import logging
import os
import re
from contextlib import contextmanager
from itertools import cycle
from typing import ContextManager, Iterable, Tuple

import cv2

from ..utils.image import resize_image_to, ImageSize, ImageArray, bgr_to_rgb
from ..utils.io import get_file, strip_url_suffix
from ..utils.opencv import (
    get_video_image_source,
    get_webcam_image_source,
    get_youtube_video_image_source
)


LOGGER = logging.getLogger(__name__)


T_ImageSource = ContextManager[Iterable[ImageArray]]


class SourceTypes:
    WEBCAM = 'webcam'
    VIDEO = 'video'
    YOUTUBE = 'youtube'
    IMAGE = 'image'


@contextmanager
def get_simple_image_source(
    path: str,
    image_size: ImageSize = None,
    repeat: bool = None,
    **_
) -> T_ImageSource:
    local_image_path = get_file(path)
    LOGGER.debug('local_image_path: %r', local_image_path)
    bgr_image_array = cv2.imread(local_image_path, cv2.IMREAD_UNCHANGED)
    if bgr_image_array is None:
        raise IOError('failed to load image: %r' % local_image_path)
    rgb_image_array = bgr_to_rgb(bgr_image_array)
    image_array = rgb_image_array
    LOGGER.info(
        'image loaded: %r (shape: %s, requested size: %s)',
        path, image_array.shape, image_size
    )
    if image_size is not None:
        image_array = resize_image_to(image_array, image_size)
    LOGGER.debug(
        'image loaded: %r (bgr: %s [%s] -> rgb: %s [%s] -> resized: %s [%s])',
        path, bgr_image_array.shape, bgr_image_array.dtype,
        rgb_image_array.shape, rgb_image_array.dtype,
        image_array.shape, image_array.dtype
    )
    image_array_iterable = [image_array]
    if repeat:
        image_array_iterable = cycle(image_array_iterable)
    yield image_array_iterable


def is_webcam_path(path: str) -> bool:
    return path.startswith('/dev/video')


def is_youtube_path(path: str) -> bool:
    return re.match(r'https?://([^/]*\.)?(youtu\.be|youtube\.com)/.*', path) is not None


def is_video_path(path: str) -> bool:
    ext = os.path.splitext(os.path.basename(strip_url_suffix(path)))[-1]
    LOGGER.debug('ext: %s', ext)
    return ext.lower() in {'.webm', '.mkv', '.mp4'}


def parse_source_type_path(path: str) -> Tuple[str, str]:
    m = re.match(r'([a-z]+):(([^/]|/[^/]).*)', path)
    if m:
        return m.group(1), m.group(2)
    return None, path


def get_source_type_for_path(path: str) -> str:
    if is_webcam_path(path):
        return SourceTypes.WEBCAM
    if is_youtube_path(path):
        return SourceTypes.YOUTUBE
    if is_video_path(path):
        return SourceTypes.VIDEO
    return SourceTypes.IMAGE


def get_source_type_and_path(path: str, **kwargs) -> Tuple[str, str]:
    source_type = kwargs.get('type')
    if source_type:
        return source_type, path
    source_type, path = parse_source_type_path(path)
    if not source_type:
        source_type = get_source_type_for_path(path)
    return source_type, path


def get_image_source_for_source_type_and_path(
    source_type: str, path: str, **kwargs
) -> T_ImageSource:
    source_type, path = get_source_type_and_path(path, **kwargs)
    if source_type == SourceTypes.WEBCAM:
        return get_webcam_image_source(path, **kwargs)
    if source_type == SourceTypes.YOUTUBE:
        return get_youtube_video_image_source(path, **kwargs)
    if source_type == SourceTypes.VIDEO:
        return get_video_image_source(path, **kwargs)
    if source_type == SourceTypes.IMAGE:
        return get_simple_image_source(path, **kwargs)
    raise ValueError('invalid source type: %r' % source_type)


def get_image_source_for_path(path: str, **kwargs) -> T_ImageSource:
    source_type, path = get_source_type_and_path(path, **kwargs)
    return get_image_source_for_source_type_and_path(
        source_type, path, **kwargs
    )
