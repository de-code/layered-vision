import logging
from abc import ABC, abstractmethod
from importlib import import_module
from typing import NamedTuple, Optional, Tuple

import numpy as np

import cv2

from ..utils.timer import LoggingTimer
from ..utils.image import (
    ImageArray,
    ImageSize,
    resize_image_to,
    get_image_size,
    get_image_with_alpha,
    box_blur_image,
    erode_image,
    dilate_image
)
from ..utils.lazy_image import resolve_lazy_image
from ..config import LayerConfig


LOGGER = logging.getLogger(__name__)


class FilterContext(NamedTuple):
    timer: LoggingTimer


class LayerFilter(ABC):
    @abstractmethod
    def filter(self, image_array: ImageArray) -> ImageArray:
        pass

    @abstractmethod
    def set_config(self, layer_config: LayerConfig):
        pass

    def close(self):
        pass


class AbstractLayerFilter(LayerFilter):
    def __init__(
        self,
        layer_config: LayerConfig,
        filter_context: FilterContext,
        **__
    ):
        self.layer_config = layer_config
        self.context = filter_context

    @property
    def filter_id(self):
        return self.layer_config.get('id')

    @abstractmethod
    def do_filter(self, image_array: ImageArray) -> ImageArray:
        pass

    def filter(self, image_array: ImageArray) -> ImageArray:
        return self.do_filter(resolve_lazy_image(image_array))

    def on_config_changed(self, layer_config: LayerConfig):
        pass

    def set_config(self, layer_config: LayerConfig):
        if self.layer_config.props == layer_config.props:
            return
        self.layer_config = layer_config
        self.on_config_changed(layer_config)


class ChromaKeyFilter(AbstractLayerFilter):
    class Config(NamedTuple):
        rgb_key: Tuple[int, int, int]
        threshold: int

    def __init__(self, layer_config: LayerConfig, **kwargs):
        self.chroma_key_config = self.parse_chroma_key_config(layer_config)
        super().__init__(layer_config, **kwargs)

    def parse_chroma_key_config(self, layer_config: LayerConfig) -> Config:
        config = ChromaKeyFilter.Config(
            rgb_key=(
                layer_config.get_int('red', 0),
                layer_config.get_int('green', 0),
                layer_config.get_int('blue', 0)
            ),
            threshold=layer_config.get_int('threshold') or 0
        )
        LOGGER.info('chroma key: %s', config)
        return config

    def on_config_changed(self, layer_config: LayerConfig):
        super().on_config_changed(layer_config)
        self.chroma_key_config = self.parse_chroma_key_config(layer_config)

    def do_filter(self, image_array: ImageArray) -> ImageArray:
        rgb_key = self.chroma_key_config.rgb_key
        if not self.chroma_key_config.threshold:
            mask = np.all(image_array[:, :, :3] != rgb_key, axis=-1).astype(np.uint8) * 255
        else:
            mask = (
                np.mean(np.abs(np.asarray(image_array)[:, :, :3] - rgb_key), axis=-1)
                > self.chroma_key_config.threshold
            ).astype(np.uint8) * 255
        LOGGER.debug('mask.shape: %s', mask.shape)
        return get_image_with_alpha(
            image_array,
            mask
        )


CHANNEL_NAMES = ['red', 'green', 'blue', 'alpha']


class AbstractOptionalChannelFilter(AbstractLayerFilter):
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__(layer_config, **kwargs)
        self.channel_index = self.parse_channel_index(layer_config)

    def parse_channel_index(self, layer_config: LayerConfig) -> Optional[int]:
        channel = layer_config.get_str('channel')
        try:
            return CHANNEL_NAMES.index(channel) if channel else None
        except IndexError as exc:
            raise RuntimeError('invalid channel: %r (expected one of %s)' % (
                channel, CHANNEL_NAMES
            )) from exc

    def on_config_changed(self, layer_config: LayerConfig):
        super().on_config_changed(layer_config)
        self.channel_index = self.parse_channel_index(layer_config)

    @abstractmethod
    def do_channel_filter(self, image_array: ImageArray) -> ImageArray:
        pass

    def do_filter(self, image_array: ImageArray) -> ImageArray:
        if self.channel_index is None:
            return self.do_channel_filter(image_array)
        channel_count = image_array.shape[2]
        if self.channel_index >= channel_count:
            LOGGER.debug('image has no channel: %d', self.channel_index)
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
        return box_blur_image(image_array, self.layer_config.get_int('value'))


class ErodeFilter(AbstractOptionalChannelFilter):
    def do_channel_filter(self, image_array: ImageArray) -> ImageArray:
        return erode_image(image_array, self.layer_config.get_int('value'))


class DilateFilter(AbstractOptionalChannelFilter):
    def do_channel_filter(self, image_array: ImageArray) -> ImageArray:
        return dilate_image(image_array, self.layer_config.get_int('value'))


class MotionBlur(AbstractOptionalChannelFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_count = self.layer_config.get_int('frame_count') or 0
        self.decay = self.layer_config.get_float('decay') or 0.0
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


class PixelateFilter(AbstractOptionalChannelFilter):
    def do_channel_filter(self, image_array: ImageArray) -> ImageArray:
        image_size = get_image_size(image_array)
        resolution = self.layer_config.get_float('value') or 0.1
        target_image_size = ImageSize(
            width=max(1, int(image_size.width * resolution)),
            height=max(1, int(image_size.height * resolution))
        )
        return resize_image_to(
            resize_image_to(image_array, target_image_size, interpolation=cv2.INTER_LINEAR),
            image_size,
            interpolation=cv2.INTER_NEAREST
        )


class CopyFilter(AbstractOptionalChannelFilter):
    def do_channel_filter(self, image_array: ImageArray) -> ImageArray:
        return image_array


FILTER_CLASS_BY_NAME_MAP = {
    'chroma_key': ChromaKeyFilter,
    'box_blur': BoxBlurFilter,
    'erode': ErodeFilter,
    'dilate': DilateFilter,
    'motion_blur': MotionBlur,
    'pixelate': PixelateFilter,
    'copy': CopyFilter
}


def create_filter(
    layer_config: LayerConfig,
    filter_context: FilterContext
) -> LayerFilter:
    filter_name = layer_config.get_str('filter')
    filter_class = FILTER_CLASS_BY_NAME_MAP.get(filter_name)
    if not filter_class:
        filter_module = import_module('layered_vision.filters.%s' % filter_name)
        _filter_class = getattr(filter_module, 'FILTER_CLASS')
        filter_class = _filter_class
        FILTER_CLASS_BY_NAME_MAP[filter_name] = _filter_class
    if filter_class:
        return filter_class(layer_config, filter_context=filter_context)
    raise RuntimeError('unrecognised filter: %r' % filter_name)
