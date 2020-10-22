import logging
from contextlib import contextmanager
from typing import ContextManager, Iterable

import cv2

from .utils.image import resize_image_to, ImageSize, ImageArray, bgr_to_rgb
from .utils.io import get_file


LOGGER = logging.getLogger(__name__)


T_ImageSource = ContextManager[Iterable[ImageArray]]


@contextmanager
def get_simple_image_source(
    path: str,
    image_size: ImageSize = None,
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
    yield [image_array]


def get_image_source_for_path(path: str, **kwargs) -> T_ImageSource:
    return get_simple_image_source(path, **kwargs)
