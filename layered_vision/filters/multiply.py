import logging

from layered_vision.utils.image import ImageArray
from layered_vision.filters.api import AbstractOptionalChannelFilter


LOGGER = logging.getLogger(__name__)


class MultiplyFilter(AbstractOptionalChannelFilter):
    def do_channel_filter(self, image_array: ImageArray) -> ImageArray:
        LOGGER.debug('multiply, image_array dtype: %s', image_array.dtype)
        value = self.layer_config.get_float('value', 1.0)
        if value == 1.0:
            return image_array
        return image_array * value


FILTER_CLASS = MultiplyFilter
