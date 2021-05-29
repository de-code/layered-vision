import logging
from typing import NamedTuple, Optional

import numpy as np
import cv2

from layered_vision.utils.image import ImageArray, get_image_size
from layered_vision.filters.api import AbstractOptionalChannelFilter
from layered_vision.config import LayerConfig
from layered_vision.utils.colors import get_color_numpy_array


LOGGER = logging.getLogger(__name__)


class FillFilter(AbstractOptionalChannelFilter):
    class Config(NamedTuple):
        color: Optional[np.ndarray]
        value: Optional[int]
        poly_points: Optional[np.ndarray]

    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__(layer_config, **kwargs)
        self.fill_config = self.parse_fill_config(layer_config)

    def parse_fill_config(self, layer_config: LayerConfig) -> Config:
        color = layer_config.get('color')
        color_value: Optional[np.ndarray] = get_color_numpy_array(color)
        value = layer_config.get_int('value')
        poly_points_list = layer_config.get_list('poly_points')
        poly_points = (
            np.float32(poly_points_list)
            if poly_points_list
            else None
        )
        config = FillFilter.Config(
            color=color_value,
            value=value,
            poly_points=poly_points
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
        assert config.color is not None
        image_shape = image_array.shape
        color_channels = image_shape[-1]
        if config.poly_points is not None:
            image_size = get_image_size(image_array)
            width_height_tuple = (image_size.width, image_size.height,)
            poly_points = (config.poly_points * width_height_tuple).astype(np.int)
            return cv2.fillPoly(
                np.array(image_array, dtype=np.float),
                pts=[poly_points],
                color=config.value if color_channels == 1 else config.color.tolist()
            )
        if color_channels == 1:
            return np.full_like(image_array, config.value)
        return np.full(
            (image_shape[0], image_shape[1], len(config.color)),
            config.color
        )


FILTER_CLASS = FillFilter
