import logging
from collections import Counter, namedtuple
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)


ImageSize = namedtuple('ImageSize', ('height', 'width'))


class SimpleImageArray:
    shape: Tuple[int, ...]
    dtype: Any

    def __getitem__(self, *args) -> Union['SimpleImageArray', int, float]:
        pass

    def astype(self, dtype: Any) -> 'SimpleImageArray':
        pass


ImageArray = Union[np.ndarray, SimpleImageArray]


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
    LOGGER.debug('bgr_to_rgb, image: %s [%s]', image.shape, image.dtype)
    color_channels = image.shape[-1]
    if color_channels == 3:
        # see https://www.scivision.dev/numpy-image-bgr-to-rgb/
        return image[..., ::-1]
    # bgra to rgba
    return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)


def rgb_to_bgr(image: ImageArray) -> ImageArray:
    LOGGER.debug('rgb_to_bgr, image: %s [%s]', image.shape, image.dtype)
    color_channels = image.shape[-1]
    if color_channels == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # rgba to bgra
    return cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)


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


def has_alpha(image: ImageArray) -> bool:
    return image.shape[-1] == 4


def has_transparent_alpha(image: ImageArray) -> bool:
    if not has_alpha(image):
        return False
    return np.any(image[:, :, 3] != 255)


def apply_alpha(image: ImageArray) -> ImageArray:
    LOGGER.debug('apply_alpha, image: %s [%s]', image.shape, image.dtype)
    color_channels = image.shape[-1]
    if color_channels == 3:
        return image
    if color_channels == 4:
        result = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        LOGGER.debug('apply_alpha, result: %s [%s]', result.shape, result.dtype)
        return result
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
        image1 = np.asarray(image1)
        image2 = np.asarray(image2)
        image2_alpha = image2[:, :, 3:] / 255
        return image1 * (1 - image2_alpha) + image2[:, :, :3] * image2_alpha
    raise ValueError('unsupported image (channels %d + %d)' % (
        image1_color_channels, image2_color_channels
    ))


def safe_multiply(
    image1: ImageArray, image2: ImageArray,
    out: Optional[ImageArray] = None
) -> ImageArray:
    if out is not None and np.issubdtype(out.dtype, np.integer):
        LOGGER.debug('out image has integer type, which is not compatible')
        out = None
    return np.multiply(
        image1,
        image2,
        out=out
    )


def combine_two_images_with_alpha(
    image1: ImageArray,
    image2: ImageArray,
    out: Optional[ImageArray] = None,
    fixed_alpha_enabled: bool = True,
    reuse_image_buffer: bool = True
):
    image2 = np.asarray(image2)
    image2_raw_alpha = image2[:, :, 3]
    LOGGER.debug('out dtype: %s', out.dtype if out is not None else None)
    if fixed_alpha_enabled:
        image2_fixed_alpha = image2_raw_alpha[0, 0]
        if np.all(image2_fixed_alpha == image2_raw_alpha):
            # shortcut for where the alpha channel has the same value
            LOGGER.debug('same alpha: %s', image2_fixed_alpha)
            image2_fixed_alpha /= 255
            return cv2.addWeighted(
                src1=image1,
                alpha=1 - image2_fixed_alpha,
                src2=image2[:, :, :3],
                beta=image2_fixed_alpha,
                gamma=0,
                dtype=3,
                dst=out
            )
    image2_alpha = np.expand_dims(image2_raw_alpha, -1) / 255
    combined_image = safe_multiply(
        image1,
        1 - image2_alpha,
        out=out if reuse_image_buffer else None
    )
    return np.add(
        combined_image,
        image2[:, :, :3] * image2_alpha,
        out=combined_image if reuse_image_buffer else None
    )


def combine_images(
    images: List[ImageArray],
    fixed_alpha_enabled: bool = True,
    reuse_image_buffer: bool = True
) -> ImageArray:
    if not images:
        return None
    LOGGER.debug('images shapes: %s', [image.shape for image in images])
    visible_images: List[ImageArray] = []
    for image in images:
        if not has_alpha(image):
            visible_images = []
        visible_images.append(image)
    if len(visible_images) <= 1:
        # nothing to combine, return last image
        return images[-1]
    image_sizes = [get_image_size(image) for image in visible_images]
    image_size_counter = Counter(image_sizes)
    if len(image_size_counter) > 1:
        raise ValueError('image sizes mismatch: %s' % image_sizes)
    LOGGER.debug('visible_images shapes: %s', [image.shape for image in visible_images])
    visible_images[0] = apply_alpha(visible_images[0])
    combined_image: Optional[ImageArray] = None
    for image2 in visible_images[1:]:
        source_image = visible_images[0] if combined_image is None else combined_image
        combined_image = combine_two_images_with_alpha(
            source_image,
            image2,
            out=combined_image,
            fixed_alpha_enabled=fixed_alpha_enabled,
            reuse_image_buffer=reuse_image_buffer
        )
    return combined_image
