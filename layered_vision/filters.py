import logging
from abc import ABC, abstractmethod
from time import time

import numpy as np

import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
from tf_bodypix.model import BodyPixModelWrapper, BodyPixResultWrapper

from .utils.image import (
    ImageArray,
    get_image_with_alpha,
    box_blur_image,
    erode_image,
    dilate_image
)
from .config import LayerConfig


LOGGER = logging.getLogger(__name__)


class LayerFilter(ABC):
    @abstractmethod
    def filter(self, image_array: ImageArray) -> ImageArray:
        pass


class AbstractLayerFilter(LayerFilter):
    def __init__(self, layer_config: dict, **__):
        self.layer_config = layer_config


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
        mask = result.get_mask(threshold=self.threshold, dtype=tf.float32) * 255
        LOGGER.debug('mask.shape: %s', mask.shape)
        return get_image_with_alpha(
            image_array,
            mask
        )


CHANNEL_NAMES = ['red', 'green', 'blue', 'alpha']


class AbstractOptionalChannelFilter(AbstractLayerFilter):
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__(layer_config, **kwargs)
        self.channel = layer_config.get('channel')
        try:
            self.channel_index = CHANNEL_NAMES.index(self.channel) if self.channel else None
        except IndexError as exc:
            raise RuntimeError('invalid channel: %r (expected one of %s)' % (
                self.channel, CHANNEL_NAMES
            )) from exc

    @abstractmethod
    def do_channel_filter(self, image_array: ImageArray) -> ImageArray:
        pass

    def filter(self, image_array: ImageArray) -> ImageArray:
        if self.channel_index is None:
            return self.do_channel_filter(image_array)
        channel_count = image_array.shape[2]
        if self.channel_index >= channel_count:
            LOGGER.debug('image has no channel: %d (%r)', self.channel_index, self.channel)
            return self.do_channel_filter(image_array)
        image_channel = np.expand_dims(image_array[:, :, self.channel_index], axis=-1)
        filtered_image_channel = self.do_channel_filter(image_channel)
        if len(filtered_image_channel.shape) == 2:
            filtered_image_channel = np.expand_dims(filtered_image_channel, axis=-1)
        LOGGER.debug(
            'image_array shape: %s, filtered_image_channel.shape: %s',
            image_array.shape, filtered_image_channel.shape
        )
        return np.concatenate(
            (
                image_array[:, :, :self.channel_index],
                filtered_image_channel,
                image_array[:, :, (self.channel_index + 1):]
            ),
            axis=-1
        )


class BoxBlurFilter(AbstractOptionalChannelFilter):
    def do_channel_filter(self, image_array: ImageArray) -> ImageArray:
        return box_blur_image(image_array, int(self.layer_config.get('value')))


class ErodeFilter(AbstractOptionalChannelFilter):
    def do_channel_filter(self, image_array: ImageArray) -> ImageArray:
        return erode_image(image_array, int(self.layer_config.get('value')))


class DilateFilter(AbstractOptionalChannelFilter):
    def do_channel_filter(self, image_array: ImageArray) -> ImageArray:
        return dilate_image(image_array, int(self.layer_config.get('value')))


class MotionBlur(AbstractOptionalChannelFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_count = int(self.layer_config.get('frame_count') or 0)
        self.decay = float(self.layer_config.get('decay') or 0)
        self._frames = []

    def do_channel_filter(self, image_array: ImageArray) -> ImageArray:
        if self.frame_count < 2:
            return image_array
        self._frames.append(image_array)
        if len(self._frames) > self.frame_count:
            self._frames.pop(0)
        if len(self._frames) <= 1:
            return image_array
        if self.decay <= 0:
            return np.mean(self._frames, axis=0)
        decayed_frames = [self._frames[-1]]
        residue_total = 1
        current_residue = 1
        for frame in reversed(self._frames[:-1]):
            current_residue *= (1.0 - self.decay)
            residue_total += current_residue
            decayed_frames.append(frame * current_residue)
        output = np.sum(decayed_frames, axis=0)
        np.divide(output, residue_total, out=output)
        np.clip(output, 0, 255, out=output)
        return output


FILTER_CLASS_BY_NAME_MAP = {
    'bodypix': BodyPixFilter,
    'chroma_key': ChromaKeyFilter,
    'box_blur': BoxBlurFilter,
    'erode': ErodeFilter,
    'dilate': DilateFilter,
    'motion_blur': MotionBlur
}


def create_filter(
    layer_config: LayerConfig
) -> LayerFilter:
    filter_name = layer_config.get('filter')
    filter_class = FILTER_CLASS_BY_NAME_MAP.get(filter_name)
    if filter_class:
        return filter_class(layer_config)
    raise RuntimeError('unrecognised filter: %r' % filter_name)
