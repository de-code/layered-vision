import logging

import cv2

from layered_vision.utils.image import ImageArray
from layered_vision.filters.api import AbstractOptionalChannelFilter


LOGGER = logging.getLogger(__name__)


class BilateralFilter(AbstractOptionalChannelFilter):
    def do_channel_filter(self, image_array: ImageArray) -> ImageArray:
        return cv2.bilateralFilter(
            image_array,
            self.layer_config.get_int('d', 1),
            self.layer_config.get_int('sigma_color', 8),
            self.layer_config.get_int('sigma_space', 8)
        )


FILTER_CLASS = BilateralFilter
