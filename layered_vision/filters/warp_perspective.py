import logging
from typing import NamedTuple

import numpy as np
import cv2

from layered_vision.utils.image import ImageArray, get_image_size, has_alpha
from layered_vision.filters.api import AbstractOptionalChannelFilter
from layered_vision.config import LayerConfig


LOGGER = logging.getLogger(__name__)


DEFAULT_POINTS = [
    [0, 0], [1.0, 0],
    [0, 1.0], [1.0, 1.0]
]


class WarpPerspectiveFilter(AbstractOptionalChannelFilter):
    class Config(NamedTuple):
        target_points: np.ndarray
        source_points: np.ndarray
        add_alpha_channel: bool

    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__(layer_config, **kwargs)
        self.warp_perspective_config = self.parse_warp_perspective_config(layer_config)

    def parse_warp_perspective_config(self, layer_config: LayerConfig) -> Config:
        config = WarpPerspectiveFilter.Config(
            target_points=np.float32(
                layer_config.get_list('target_points', DEFAULT_POINTS)
            ),
            source_points=np.float32(
                layer_config.get_list('source_points', DEFAULT_POINTS)
            ),
            add_alpha_channel=self.layer_config.get_bool('add_alpha_channel', True)
        )
        LOGGER.info('warp perspective config: %s', config)
        assert config.source_points.shape == (4, 2)
        assert config.target_points.shape == (4, 2)
        return config

    def on_config_changed(self, layer_config: LayerConfig):
        super().on_config_changed(layer_config)
        self.warp_perspective_config = self.parse_warp_perspective_config(layer_config)

    def do_channel_filter(self, image_array: ImageArray) -> ImageArray:
        LOGGER.debug('warp perspective, image_array dtype: %s', image_array.dtype)
        config = self.warp_perspective_config
        if np.allclose(config.source_points, config.target_points):
            return image_array
        image_size = get_image_size(image_array)
        width_height_tuple = (image_size.width, image_size.height,)
        source_image_points = config.source_points * width_height_tuple
        target_image_points = config.target_points * width_height_tuple
        LOGGER.debug('source_image_points: %s', source_image_points)
        LOGGER.debug('target_image_points: %s', target_image_points)
        transformation_matrix = cv2.getPerspectiveTransform(
            np.float32(source_image_points),
            np.float32(target_image_points)
        )
        LOGGER.debug('transformation_matrix: %s', transformation_matrix)
        if not has_alpha(image_array):
            image_array = np.dstack((
                image_array,
                np.full_like(image_array[:, :, 0], 255),
            ))
        if np.issubdtype(image_array.dtype, np.integer):
            image_array = image_array.astype(np.float)
        return cv2.warpPerspective(
            image_array,
            transformation_matrix,
            width_height_tuple
        )


FILTER_CLASS = WarpPerspectiveFilter
