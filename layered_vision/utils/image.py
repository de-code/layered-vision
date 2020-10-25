import logging
from collections import namedtuple
from functools import reduce
from typing import List

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


def box_blur_image(image: ImageArray, blur_size: int) -> ImageArray:
    if not blur_size:
        return image
    if len(image.shape) == 4:
        image = image[0]
    result = cv2.blur(np.asarray(image), (blur_size, blur_size))
    if len(result.shape) == 2:
        result = np.expand_dims(result, axis=-1)
    result = result.astype(np.float32)
    return result


def dilate_image(image: ImageArray, size: int) -> ImageArray:
    kernel = np.ones((size, size), dtype=np.uint8)
    return cv2.dilate(image, kernel, iterations=None)


def erode_image(image: ImageArray, size: int) -> ImageArray:
    kernel = np.ones((size, size), dtype=np.uint8)
    return cv2.erode(image, kernel, iterations=None)


def bgr_to_rgb(image: ImageArray) -> ImageArray:
    # see https://www.scivision.dev/numpy-image-bgr-to-rgb/
    return image[..., ::-1]


def rgb_to_bgr(image: ImageArray) -> ImageArray:
    return bgr_to_rgb(image)


def get_image_with_alpha(image: ImageArray, alpha: ImageArray) -> ImageArray:
    color_channels = image.shape[-1]
    if color_channels == 3:
        if len(alpha.shape) == 2:
            alpha = np.expand_dims(alpha, -1)
        return np.concatenate(
            (image, alpha),
            axis=-1
        )
    raise ValueError('unsupported image')


def apply_alpha(image: ImageArray) -> ImageArray:
    color_channels = image.shape[-1]
    if color_channels == 3:
        return image
    if color_channels == 4:
        return image[:, :, :3] * (image[:, :, 3:] / 255)
    raise ValueError('unsupported image, shape=%s' % image.shape)


def combine_two_images(image1: ImageArray, image2: ImageArray) -> ImageArray:
    image1_size = get_image_size(image1)
    image2_size = get_image_size(image2)
    if image1_size != image2_size:
        raise ValueError('image size mismatch: %s != %s' % (image1_size, image2_size))
    image1_color_channels = image1.shape[-1]
    image2_color_channels = image2.shape[-1]
    if image2_color_channels <= 3:
        # image2 fully opaque
        return image2
    if image1_color_channels == 3:
        # no output alpha
        image2_alpha = image2[:, :, 3:] / 255
        return image1 * (1 - image2_alpha) + image2[:, :, :3] * image2_alpha
    raise ValueError('unsupported image')


def combine_images(images: List[ImageArray]) -> ImageArray:
    return reduce(
        combine_two_images,
        images
    )
