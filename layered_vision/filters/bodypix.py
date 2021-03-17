import logging
from time import time
from typing import List, NamedTuple, Optional

import numpy as np

from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
from tf_bodypix.model import BodyPixModelWrapper, BodyPixResultWrapper

from layered_vision.utils.image import (
    ImageArray,
    get_image_with_alpha
)

from layered_vision.filters.api import AbstractLayerFilter
from layered_vision.config import LayerConfig


LOGGER = logging.getLogger(__name__)


class BodyPixFilter(AbstractLayerFilter):
    class Config(NamedTuple):
        model_path: str
        threshold: float
        internal_resolution: float
        cache_model_result_secs: float
        parts: List[str]

    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__(layer_config, **kwargs)
        self.bodypix_config = self.parse_bodypix_config(layer_config)
        self._bodypix_model = None
        self._bodypix_result_cache = None
        self._bodypix_result_cache_time: Optional[float] = None

    def parse_bodypix_config(self, layer_config: LayerConfig) -> Config:
        config = BodyPixFilter.Config(
            model_path=(
                layer_config.get_str('model_path')
                or BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16
            ),
            threshold=layer_config.get_float('threshold') or 0.50,
            internal_resolution=layer_config.get_float('internal_resolution') or 0.50,
            cache_model_result_secs=(
                layer_config.get_float('cache_model_result_secs') or 0.0
            ),
            parts=list(
                layer_config.get_str_list('parts') or []
            )
        )
        LOGGER.info('bodypix filter config: %s', config)
        return config

    def on_config_changed(self, layer_config: LayerConfig):
        super().on_config_changed(layer_config)
        bodypix_config = self.parse_bodypix_config(layer_config)
        if (
            bodypix_config.model_path != self.bodypix_config.model_path
            or bodypix_config.internal_resolution != self.bodypix_config.internal_resolution
        ):
            # we need to reload the model
            self._bodypix_model = None
        self.bodypix_config = bodypix_config

    def load_bodypix_model(self) -> BodyPixModelWrapper:
        LOGGER.info('loading bodypix model: %s', self.bodypix_config.model_path)
        bodypix_model = load_model(
            download_model(self.bodypix_config.model_path),
            internal_resolution=self.bodypix_config.internal_resolution
        )
        LOGGER.info('bodypix internal resolution: %s', bodypix_model.internal_resolution)
        return bodypix_model

    @property
    def bodypix_model(self) -> BodyPixModelWrapper:
        if self._bodypix_model is None:
            self._bodypix_model = self.load_bodypix_model()
        return self._bodypix_model

    def get_bodypix_result(self, image_array: ImageArray) -> BodyPixResultWrapper:
        current_time = time()
        if (
            self._bodypix_result_cache is not None
            and current_time < (
                self._bodypix_result_cache_time + self.bodypix_config.cache_model_result_secs
            )
        ):
            return self._bodypix_result_cache
        self._bodypix_result_cache = self.bodypix_model.predict_single(image_array)
        self._bodypix_result_cache_time = current_time
        return self._bodypix_result_cache

    def do_filter(self, image_array: ImageArray) -> ImageArray:
        result = self.get_bodypix_result(image_array)
        mask = result.get_mask(threshold=self.bodypix_config.threshold, dtype=np.uint8)
        if self.bodypix_config.parts:
            mask = result.get_part_mask(mask, part_names=self.bodypix_config.parts)
        mask = np.multiply(mask, 255)
        LOGGER.debug('mask.shape: %s', mask.shape)
        return get_image_with_alpha(
            image_array,
            mask
        )


FILTER_CLASS = BodyPixFilter
