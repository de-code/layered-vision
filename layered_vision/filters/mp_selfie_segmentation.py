import logging
from time import time
from typing import NamedTuple, Optional, cast

import numpy as np
import mediapipe as mp

from layered_vision.utils.image import (
    ImageArray,
    get_image_with_alpha
)

from layered_vision.filters.api import AbstractLayerFilter
from layered_vision.config import LayerConfig


LOGGER = logging.getLogger(__name__)


class BodyPixFilter(AbstractLayerFilter):
    class Config(NamedTuple):
        threshold: float
        model_selection: int
        cache_model_result_secs: float

    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__(layer_config, **kwargs)
        self.mp_selfie_segmentation_config = self.parse_mp_selfie_segmentation_config(
            layer_config
        )
        self._selfie_segmentation: Optional[
            mp.solutions.selfie_segmentation.SelfieSegmentation
        ] = None
        self._mask_cache: Optional[np.ndarray] = None
        self._mask_cache_time: float = 0

    def parse_mp_selfie_segmentation_config(self, layer_config: LayerConfig) -> Config:
        config = BodyPixFilter.Config(
            threshold=layer_config.get_float('threshold', 0.1),
            model_selection=layer_config.get_int('model_selection', 1),
            cache_model_result_secs=(
                layer_config.get_float('cache_model_result_secs', 0.0)
            )
        )
        LOGGER.info('mp selfie segmentation filter config: %s', config)
        return config

    def on_config_changed(self, layer_config: LayerConfig):
        super().on_config_changed(layer_config)
        mp_selfie_segmentation_config = self.parse_mp_selfie_segmentation_config(layer_config)
        if (
            mp_selfie_segmentation_config.model_selection
            != self.mp_selfie_segmentation_config.model_selection
        ):
            # we need to reload the model
            self.unload_selfie_segmentation()
        self.mp_selfie_segmentation_config = mp_selfie_segmentation_config

    def load_selfie_segmentation(self) -> mp.solutions.selfie_segmentation.SelfieSegmentation:
        LOGGER.info('loading selfie segmentation: %s', self.mp_selfie_segmentation_config)
        selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=self.mp_selfie_segmentation_config.model_selection
        )
        selfie_segmentation.__enter__()
        return selfie_segmentation

    @property
    def selfie_segmentation(self) -> mp.solutions.selfie_segmentation.SelfieSegmentation:
        if self._selfie_segmentation is None:
            self._selfie_segmentation = self.load_selfie_segmentation()
        return self._selfie_segmentation

    def unload_selfie_segmentation(self):
        if self._selfie_segmentation is not None:
            LOGGER.info('unloading selfie segmentation')
            self._selfie_segmentation.__exit__(None, None, None)
            self._selfie_segmentation = None
            self._mask_cache = None

    def close(self):
        self.unload_selfie_segmentation()
        super().close()

    def get_segmentation_mask(self, image_array: ImageArray) -> np.ndarray:
        current_time = time()
        if (
            self._mask_cache is not None
            and current_time < (
                self._mask_cache_time + self.mp_selfie_segmentation_config.cache_model_result_secs
            )
        ):
            return self._mask_cache
        segmentation_result: 'mp.solutions.SolutionOutputs' = (
            self.selfie_segmentation.process(image_array)
        )
        self._mask_cache = segmentation_result.segmentation_mask
        self._mask_cache_time = current_time
        return self._mask_cache

    def do_filter(self, image_array: ImageArray) -> ImageArray:
        mask = self.get_segmentation_mask(image_array)
        mask = cast(
            ImageArray,
            (mask >= self.mp_selfie_segmentation_config.threshold) * 255.0
        )
        LOGGER.debug('mask.shape: %s', mask.shape)
        return get_image_with_alpha(
            image_array,
            mask
        )


FILTER_CLASS = BodyPixFilter
