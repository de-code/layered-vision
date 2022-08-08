import logging
import os
import re
from contextlib import contextmanager
from itertools import cycle
from importlib import import_module
from typing import Callable, ContextManager, Dict, Iterable, Iterator, Optional, Tuple

import cv2
import numpy as np

from ..utils.path import parse_type_path
from ..utils.image import (
    resize_image_to,
    ImageSize,
    ImageArray,
    bgr_to_rgb,
    apply_alpha,
    get_image_with_alpha
)
from ..utils.io import get_file, strip_url_suffix
from ..utils.opencv import (
    get_video_image_source,
    get_webcam_image_source
)


LOGGER = logging.getLogger(__name__)


T_ImageSource = Iterable[ImageArray]

T_ImageSourceFactory = Callable[..., ContextManager[T_ImageSource]]


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
    alpha: Optional[float] = None,
    **_
) -> Iterator[T_ImageSource]:
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
    if alpha is not None and alpha < 1:
        LOGGER.info('setting alpha of image to: %.2f', alpha)
        image_array = get_image_with_alpha(
            apply_alpha(image_array),
            np.full_like(image_array[:, :, 0], int(alpha * 255))
        )
    image_array_iterable: T_ImageSource = [image_array]
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
    source_type, path = parse_type_path(path)
    if not source_type:
        source_type = get_source_type_for_path(path)
    return source_type, path


IMAGE_SOURCE_FACTORY_BY_TYPE: Dict[str, T_ImageSourceFactory] = {
    SourceTypes.IMAGE: get_simple_image_source,
    SourceTypes.VIDEO: get_video_image_source,
    SourceTypes.WEBCAM: get_webcam_image_source
}


def get_image_source_for_source_type_and_path(
    source_type: str, path: str, **kwargs
) -> ContextManager[T_ImageSource]:
    image_source_factory = IMAGE_SOURCE_FACTORY_BY_TYPE.get(source_type)
    if image_source_factory is None:
        image_source_module = import_module('layered_vision.sources.%s' % source_type)
        image_source_factory = getattr(image_source_module, 'IMAGE_SOURCE_FACTORY')
        IMAGE_SOURCE_FACTORY_BY_TYPE[source_type] = image_source_factory
    if image_source_factory is not None:
        return image_source_factory(path, **kwargs)
    raise ValueError('invalid source type: %r' % source_type)


def get_image_source_for_path(path: str, **kwargs) -> ContextManager[T_ImageSource]:
    source_type, path = get_source_type_and_path(path, **kwargs)
    return get_image_source_for_source_type_and_path(
        source_type, path, **kwargs
    )
