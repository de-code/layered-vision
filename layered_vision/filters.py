import logging
from abc import ABC, abstractmethod

import numpy as np

from .utils.image import ImageArray, get_image_with_alpha
from .config import LayerConfig


LOGGER = logging.getLogger(__name__)


class LayerFilter(ABC):
    @abstractmethod
    def filter(self, image_array: ImageArray) -> ImageArray:
        pass


class AbstractLayerFilter(LayerFilter):
    def __init__(self, *_, **__):
        pass


class ChromaKeyFilter(AbstractLayerFilter):
    def __init__(self, layer_config: dict, **kwargs):
        self.rgb_key = (
            int(layer_config.get('red') or '0'),
            int(layer_config.get('green') or '0'),
            int(layer_config.get('blue') or '0')
        )
        self.threshold = int(layer_config.get('threshold') or '0')
        LOGGER.info('chroma key: %s', self.rgb_key)
        super().__init__(layer_config, **kwargs)

    def do_filter(self, image_array: ImageArray) -> ImageArray:
        if not self.threshold:
            mask = np.all(image_array[:, :, :3] != self.rgb_key, axis=-1).astype(np.uint8) * 255
        else:
            mask = (
                np.mean(np.abs(image_array[:, :, :3] - self.rgb_key), axis=-1)
                > self.threshold
            ).astype(np.uint8) * 255
        LOGGER.debug('mask.shape: %s', mask.shape)
        return get_image_with_alpha(
            image_array,
            mask
        )

    def filter(self, image_array: ImageArray) -> ImageArray:
        return self.do_filter(image_array)


FILTER_CLASS_BY_NAME_MAP = {
    'chroma_key': ChromaKeyFilter,
}


def create_filter(
    layer_config: LayerConfig
) -> LayerFilter:
    filter_name = layer_config.get('filter')
    filter_class = FILTER_CLASS_BY_NAME_MAP.get(filter_name)
    if filter_class:
        return filter_class(layer_config)
    raise RuntimeError('unrecognised filter: %r' % filter_name)
