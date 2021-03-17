import logging

from layered_vision.utils.image import ImageArray, combine_images
from layered_vision.utils.lazy_image import LazyImageList
from layered_vision.filters.api import AbstractLayerFilter


LOGGER = logging.getLogger(__name__)


class CompositeFilter(AbstractLayerFilter):
    def do_filter(self, image_array: ImageArray) -> ImageArray:
        assert isinstance(image_array, LazyImageList)
        lazy_images = image_array.images
        images = list(reversed([
            lazy_image.image
            for lazy_image in reversed(lazy_images)
        ]))
        with self.context.timer.enter_step(f'{self.filter_id}.combine'):
            return combine_images(images)

    def filter(self, image_array: ImageArray) -> ImageArray:
        # don't resolve lazy image
        return self.do_filter(image_array)


FILTER_CLASS = CompositeFilter
