import logging
from contextlib import contextmanager
from itertools import cycle
from typing import Iterable, Iterator

import numpy as np

from layered_vision.utils.image import ImageArray, ImageSize
from layered_vision.sources.api import T_ImageSourceFactory, T_ImageSource
from layered_vision.utils.colors import get_color_numpy_array


LOGGER = logging.getLogger(__name__)


@contextmanager
def get_fill_image_source(
    path: str,  # pylint: disable=unused-argument
    *args,
    image_size: ImageSize = None,
    repeat: bool = False,
    color,
    **_
) -> Iterator[Iterable[ImageArray]]:
    color_value = get_color_numpy_array(color)
    if color_value is None:
        raise RuntimeError('color required')
    if image_size is None:
        raise RuntimeError('image size required')
    LOGGER.info('fill input: color=%r (image_size=%s)', color_value, image_size)
    image_array = np.full(
        (image_size.height, image_size.width, len(color_value)),
        color_value
    )
    image_array_iterable: T_ImageSource = [image_array]
    if repeat:
        image_array_iterable = cycle(image_array_iterable)
    yield image_array_iterable


IMAGE_SOURCE_FACTORY: T_ImageSourceFactory = get_fill_image_source
