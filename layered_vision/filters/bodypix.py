import logging
from time import time

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
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__(layer_config, **kwargs)
        self.model_path = (
            layer_config.get('model_path')
            or BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16
        )
        self._bodypix_model = None
        self.threshold = float(layer_config.get('threshold') or 0.50)
        self.internal_resolution = float(layer_config.get('internal_resolution') or 0.50)
        self.cache_model_result_secs = float(
            layer_config.get('cache_model_result_secs') or 0.0
        )
        self.parts = list(
            layer_config.get('parts') or []
        )
        self._bodypix_result_cache = None
        self._bodypix_result_cache_time = None

    def load_bodypix_model(self) -> BodyPixModelWrapper:
        LOGGER.info('loading bodypix model: %s', self.model_path)
        bodypix_model = load_model(
            download_model(self.model_path),
            internal_resolution=self.internal_resolution
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
            and current_time < self._bodypix_result_cache_time + self.cache_model_result_secs
        ):
            return self._bodypix_result_cache
        self._bodypix_result_cache = self.bodypix_model.predict_single(image_array)
        self._bodypix_result_cache_time = current_time
        return self._bodypix_result_cache

    def filter(self, image_array: ImageArray) -> ImageArray:
        result = self.get_bodypix_result(image_array)
        mask = result.get_mask(threshold=self.threshold, dtype=np.uint8)
        if self.parts:
            mask = result.get_part_mask(mask, part_names=self.parts)
        mask = np.multiply(mask, 255)
        LOGGER.debug('mask.shape: %s', mask.shape)
        return get_image_with_alpha(
            image_array,
            mask
        )


FILTER_CLASS = BodyPixFilter
