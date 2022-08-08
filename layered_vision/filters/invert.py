import logging

import cv2

from layered_vision.utils.image import ImageArray
from layered_vision.filters.api import AbstractOptionalChannelFilter


LOGGER = logging.getLogger(__name__)


class InvertFilter(AbstractOptionalChannelFilter):
    def do_channel_filter(self, image_array: ImageArray) -> ImageArray:
        LOGGER.debug('invert, image_array dtype: %s', image_array.dtype)
        return cv2.bitwise_not(image_array)


FILTER_CLASS = InvertFilter
