import logging
from typing import NamedTuple, Optional

import numpy as np

from layered_vision.utils.image import ImageArray
from layered_vision.filters.api import AbstractOptionalChannelFilter
from layered_vision.config import LayerConfig
from layered_vision.utils.colors import get_color_numpy_array


LOGGER = logging.getLogger(__name__)


class FillFilter(AbstractOptionalChannelFilter):
    class Config(NamedTuple):
        color: Optional[np.ndarray]
        value: Optional[int]

    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__(layer_config, **kwargs)
        self.fill_config = self.parse_fill_config(layer_config)

    def parse_fill_config(self, layer_config: LayerConfig) -> Config:
        color = layer_config.get('color')
        color_value: Optional[np.ndarray] = get_color_numpy_array(color)
        value = layer_config.get_int('value')
        config = FillFilter.Config(
            color=color_value,
            value=value
        )
        LOGGER.info('fill config: %s', config)
        assert config.color is not None or config.value is not None
        return config

    def on_config_changed(self, layer_config: LayerConfig):
        super().on_config_changed(layer_config)
        self.fill_config = self.parse_fill_config(layer_config)

    def do_channel_filter(self, image_array: ImageArray) -> ImageArray:
        LOGGER.debug('fill, image_array dtype: %s', image_array.dtype)
        config = self.fill_config
        assert config.color
        image_shape = image_array.shape
        color_channels = image_shape[-1]
        if color_channels == 1:
            return np.full_like(image_array, config.value)
        return np.full(
            (image_shape[0], image_shape[1], len(config.color)),
            config.color
        )


FILTER_CLASS = FillFilter
