import logging
from collections import namedtuple

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)


ImageSize = namedtuple('ImageSize', ('height', 'width'))

ImageArray = np.ndarray


def get_image_size(image: ImageArray):
    height, width, *_ = image.shape
    return ImageSize(height=height, width=width)


def resize_image_to(
    image: ImageArray,
    size: ImageSize,
    interpolation: int = cv2.INTER_LINEAR
) -> ImageArray:
    if get_image_size(image) == size:
        LOGGER.debug('image has already desired size: %s', size)
        return image

    return cv2.resize(
        image,
        (size.width, size.height),
        interpolation=interpolation
    )


def bgr_to_rgb(image: ImageArray) -> ImageArray:
    # see https://www.scivision.dev/numpy-image-bgr-to-rgb/
    return image[..., ::-1]


def rgb_to_bgr(image: ImageArray) -> ImageArray:
    return bgr_to_rgb(image)
