import logging
import os
from abc import ABC, abstractmethod
from time import monotonic
from contextlib import ExitStack
from functools import partial
from threading import Event
from typing import ContextManager, Dict, Optional, List

from .utils.timer import LoggingTimer
from .utils.image import (
    ImageArray,
    ImageSize,
    get_image_size,
    resize_image_to,
    has_transparent_alpha,
    apply_alpha,
    combine_images
)
from .utils.lazy_image import LazyImageList

from .sinks.api import (
    T_OutputSink,
    get_image_output_sink_for_path
)

from .config import load_config, apply_config_override_map, AppConfig, LayerConfig, T_Value
from .resolved_config import ResolvedAppConfig, ResolvedLayerConfig, ResolvedLayerType
from .sources.api import get_image_source_for_path, T_ImageSource
from .filters.api import FilterContext, LayerFilter, create_filter


LOGGER = logging.getLogger(__name__)


CORE_LAYER_PROPS = {
    'id',
    'enabled',
    'no_source',
    'input_path',
    'output_path',
    'filter',
    'width',
    'height',
    'resize_like_id',
    'type'
}


def get_custom_layer_props(layer_config: LayerConfig) -> dict:
    return {
        key: value
        for key, value in layer_config.props.items()
        if key not in CORE_LAYER_PROPS
    }


class LayerException(RuntimeError):
    pass


class LayerIds:
    ON_ERROR = 'on_error'


class ErrorHandler(ABC):
    @abstractmethod
    def on_error(
        self,
        layer: 'RuntimeLayer',
        error: Exception,
        source_image: Optional[ImageArray],
    ) -> Optional[ImageArray]:
        pass


class RuntimeContext:
    def __init__(
        self,
        layer_by_id: Dict[str, 'RuntimeLayer'],
        timer: LoggingTimer,
        error_handler: ErrorHandler,
        preferred_image_size: ImageSize = None,
        application_stopped_event: Event = None
    ):
        self.layer_by_id = layer_by_id
        self.timer = timer
        self.preferred_image_size = preferred_image_size
        self.frame_cache: dict = {}
        self.application_stopped_event = application_stopped_event
        self.on_error_layer: Optional[RuntimeLayer] = None
        self.error_handler: ErrorHandler = error_handler


class DelegateErrorHandler(ErrorHandler):
    def on_error(
        self,
        layer: 'RuntimeLayer',
        error: Exception,
        source_image: Optional[ImageArray],
    ) -> Optional[ImageArray]:
        on_error_layer = layer.context.on_error_layer
        result: Optional[ImageArray] = None
        if on_error_layer is not None and on_error_layer != layer:
            result = next(on_error_layer)
        if source_image is not None and result is not None:
            result = resize_image_to(result, get_image_size(source_image))
        if result is None:
            raise error
        return result


class OutputSinkWrapper:
    def __init__(
        self,
        output_sink_generator: ContextManager[T_OutputSink]
    ):
        self.output_sink_generator = output_sink_generator
        self._output_sink: Optional[T_OutputSink] = None

    def close(self):
        self.__exit__(None, None, None)

    @property
    def output_sink(self):
        if self._output_sink is None:
            self.__enter__()
        assert self._output_sink is not None
        return self._output_sink

    def __enter__(self):
        self._output_sink = self.output_sink_generator.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        self.output_sink_generator.__exit__(*args, **kwargs)


class ImageSourceWrapper:
    def __init__(
        self,
        image_source_generator: ContextManager[T_ImageSource],
        stopped_event: Event
    ):
        self.image_source_generator = image_source_generator
        self._image_iterator: Optional[T_ImageSource] = None
        self.stopped_event = stopped_event

    def close(self):
        self.__exit__(None, None, None)

    @property
    def image_iterator(self):
        if self._image_iterator is None:
            self.__enter__()
        assert self._image_iterator is not None
        return self._image_iterator

    def __enter__(self):
        self._image_iterator = iter(self.image_source_generator.__enter__())
        return self

    def __exit__(self, *args, **kwargs):
        self.stopped_event.set()
        self.image_source_generator.__exit__(*args, **kwargs)


def get_image_source_for_layer_config(
    layer_config: LayerConfig,
    preferred_image_size: Optional[ImageSize]
) -> ImageSourceWrapper:
    width = layer_config.get_int('width')
    height = layer_config.get_int('height')
    image_size: Optional[ImageSize]
    if width and height:
        image_size = ImageSize(width=width, height=height)
    else:
        image_size = preferred_image_size
    stopped_event = Event()
    return ImageSourceWrapper(
        get_image_source_for_path(
            layer_config.get_str('input_path'),
            image_size=image_size,
            stopped_event=stopped_event,
            **get_custom_layer_props(layer_config)
        ),
        stopped_event=stopped_event
    )


def safe_close(closable):
    if closable is not None:
        closable.close()
        closable = None
    return closable


class RuntimeLayer:
    def __init__(
        self,
        layer_index: int,
        layer_config: ResolvedLayerConfig,
        layer_id: str,
        context: RuntimeContext,
        source_layers: List['RuntimeLayer'] = None
    ):
        self.layer_index = layer_index
        self.layer_config = layer_config
        self.layer_id = layer_id
        self.exit_stack = ExitStack()
        self.source_layers = (source_layers or []).copy()
        self.context = context
        self.previous_error: Optional[Exception] = None
        self.is_output_layer = False

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()
        self.exit_stack.__exit__(*args, **kwargs)

    def close(self):
        pass

    def __repr__(self):
        return '%s(layer_config=%r, ...)' % (
            type(self).__name__,
            self.layer_config
        )

    def on_layer_config_changed(self):
        pass

    def set_layer_config(self, layer_config: ResolvedLayerConfig):
        self.previous_error = None
        if self.layer_config.props == layer_config.props:
            # replace layer config anyway, as it may include other resolved config information,
            # such as the source layers
            self.layer_config = layer_config
            return
        LOGGER.info(
            'updating layer config: id=%r',
            layer_config.layer_id
        )
        self.layer_config = layer_config
        self.on_layer_config_changed()

    def __iter__(self):
        return self

    @property
    def source_lazy_image_list(self):
        return LazyImageList([
            partial(next, source_layer)
            for source_layer in self.source_layers
        ])

    def get_error_image(
        self, error: Exception, source_image: Optional[ImageArray]
    ) -> Optional[ImageArray]:
        result = self.context.error_handler.on_error(
            self, error, source_image
        )
        if self.previous_error is None:
            LOGGER.warning(
                'failed to process filter for %r due to %s',
                self.layer_config.layer_id,
                error,
                exc_info=error
            )
            self.previous_error = error
        return result

    def get_next_image(self):
        raise LayerException('input not supported for layer: %s' % self)

    def write(self, image_array: ImageArray):
        raise LayerException('output not supported for layer: %s' % self)

    def __next__(self):
        try:
            if not self.is_enabled:
                return next(self.source_layers[0])
            return self.get_next_image()
        except (StopIteration, LayerException):
            raise
        except Exception as exc:
            raise LayerException('failed to process layer %r due to %r' % (
                self.layer_id, exc
            )) from exc

    @property
    def is_enabled(self) -> bool:
        return self.layer_config.get_bool('enabled', True)

    @property
    def resize_like_id(self) -> Optional[str]:
        return self.layer_config.get_str('resize_like_id')


class InputSourceRuntimeLayer(RuntimeLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_source: Optional[ImageSourceWrapper] = None

    def close(self):
        self.image_source = safe_close(self.image_source)

    def on_layer_config_changed(self):
        super().on_layer_config_changed()
        self.close()

    def get_image_iterator(self) -> T_ImageSource:
        if self.image_source is not None:
            return self.image_source.image_iterator
        preferred_image_size = self.context.preferred_image_size
        resize_like_id = self.resize_like_id
        if resize_like_id:
            image = next(self.context.layer_by_id[resize_like_id])
            preferred_image_size = get_image_size(image)
        image_source = get_image_source_for_layer_config(
            self.layer_config,
            preferred_image_size=preferred_image_size
        )
        self.image_source = image_source
        return image_source.image_iterator

    def get_next_input_image(self):
        image_array = self.context.frame_cache.get(self.layer_id)
        if image_array is None:
            image_array = next(self.get_image_iterator())
            if has_transparent_alpha(image_array) and self.source_layers:
                source_image = next(self.source_layers[0])
                self.context.timer.on_step_start(self.layer_id + '.combine')
                image_array = combine_images([source_image] + [image_array])
            self.context.frame_cache[self.layer_id] = image_array
        if self.context.preferred_image_size is None:
            image_size = get_image_size(image_array)
            LOGGER.info('setting preferred image size to: %s', image_size)
            self.context.preferred_image_size = image_size
        return image_array

    def get_next_image(self):
        try:
            with self.context.timer.enter_step(self.layer_id):
                if self.previous_error:
                    raise LayerException(
                        'layer %r failed before' % self.layer_config.layer_id
                    ) from self.previous_error
                return self.get_next_input_image()
        except Exception as exc:  # pylint: disable=broad-except
            return self.get_error_image(exc, source_image=None)


class FilterRuntimeLayer(RuntimeLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filter: Optional[LayerFilter] = None

    def close(self):
        self.filter = safe_close(self.filter)

    def on_layer_config_changed(self):
        super().on_layer_config_changed()
        try:
            if self.filter:
                self.filter.set_config(self.layer_config)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning(
                'failed to update filter config for %r due to %s',
                self.layer_config.layer_id,
                exc,
                exc_info=exc
            )
            self.filter = safe_close(self.filter)

    def get_filter(self) -> LayerFilter:
        if self.filter is not None:
            return self.filter
        _filter = create_filter(
            self.layer_config,
            filter_context=FilterContext(timer=self.context.timer)
        )
        self.filter = _filter
        return _filter

    def get_filter_image(self, source_image):
        try:
            if self.previous_error:
                raise LayerException(
                    'layer %r failed before' % self.layer_config.layer_id
                ) from self.previous_error
            return self.get_filter().filter(source_image)
        except Exception as exc:  # pylint: disable=broad-except
            return self.get_error_image(exc, source_image)

    def get_next_image(self):
        with self.context.timer.enter_step(self.layer_id):
            return self.get_filter_image(self.source_lazy_image_list)


class OutputSinkRuntimeLayer(RuntimeLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_sink_wrapper: Optional[OutputSinkWrapper] = None
        self.is_output_layer = True

    def close(self):
        self.output_sink_wrapper = safe_close(self.output_sink_wrapper)

    def write(self, image_array: ImageArray):
        image_array = apply_alpha(image_array)
        LOGGER.debug('output shape: %s', image_array.shape)
        self.get_output_sink()(image_array)

    def on_layer_config_changed(self):
        self.close()

    def get_output_sink(self) -> T_OutputSink:
        if not self.is_output_layer:
            raise RuntimeError('not an output layer: %r' % self)
        output_path = self.layer_config.get_str('output_path')
        assert output_path, "output_path required"
        if self.output_sink_wrapper is None:
            self.output_sink_wrapper = OutputSinkWrapper(
                get_image_output_sink_for_path(
                    output_path,
                    **get_custom_layer_props(self.layer_config)
                )
            )
        return self.output_sink_wrapper.output_sink


def create_runtime_layer(
    layer_config: ResolvedLayerConfig,
    **kwargs
) -> RuntimeLayer:
    layer_type = layer_config.resolved_layer_type
    if layer_type == ResolvedLayerType.INPUT_SOURCE:
        return InputSourceRuntimeLayer(layer_config=layer_config, **kwargs)
    if layer_type == ResolvedLayerType.FILTER:
        return FilterRuntimeLayer(layer_config=layer_config, **kwargs)
    if layer_type == ResolvedLayerType.OUTPUT_SINK:
        return OutputSinkRuntimeLayer(layer_config=layer_config, **kwargs)
    raise RuntimeError('unrecognised layer type: %s' % layer_type)


class LayeredVisionApp:
    def __init__(
        self,
        config_path: str,
        override_map: Dict[str, Dict[str, T_Value]] = None,
        min_config_reload_secs: float = 1.0,
        error_handler: ErrorHandler = None
    ):
        if error_handler is None:
            error_handler = DelegateErrorHandler()
        self.config_path = config_path
        self.config_modified_time = 0
        self.override_map = override_map
        self.exit_stack = ExitStack()
        self.timer = LoggingTimer()
        self.output_runtime_layers: Optional[List[RuntimeLayer]] = None
        self.application_stopped_event = Event()
        self.layer_by_id: Dict[str, RuntimeLayer] = {}
        self.context = RuntimeContext(
            layer_by_id=self.layer_by_id,
            timer=self.timer,
            application_stopped_event=self.application_stopped_event,
            error_handler=error_handler
        )
        self.min_config_reload_secs = min_config_reload_secs
        self.config_last_checked_time = monotonic()

    def __enter__(self):
        try:
            self.load()
            return self
        except Exception as exc:
            self.exit_stack.__exit__(type(exc), exc, None)
            raise exc

    def __exit__(self, *args, **kwargs):
        self.application_stopped_event.set()
        self.exit_stack.__exit__(*args, **kwargs)
        self.close()

    def close(self):
        for layer in self.layer_by_id.values():
            layer.close()
        self.layer_by_id = {}

    def get_config_modified_timestamp(self):
        if os.path.isfile(self.config_path):
            return os.path.getmtime(self.config_path)
        return 0

    def check_reload_config(self):
        if monotonic() < self.config_last_checked_time + self.min_config_reload_secs:
            return
        if self.get_config_modified_timestamp() > self.config_modified_time:
            self.reload_config()
        self.config_last_checked_time = monotonic()

    def reload_config(self):
        self.load()

    def remove_layer_by_id(self, layer_id: str):
        LOGGER.info('removing layer: id=%r', layer_id)
        self.layer_by_id[layer_id].close()
        del self.layer_by_id[layer_id]

    def set_resolved_app_config(self, resolved_app_config: ResolvedAppConfig):
        had_layers = bool(self.layer_by_id)
        layers = resolved_app_config.layers
        assert len(layers) >= 2
        runtime_layers: List[RuntimeLayer] = []
        for layer_index, layer_config in enumerate(layers):
            layer_id = layer_config.layer_id
            runtime_layer = self.layer_by_id.get(layer_id)

            if (
                runtime_layer
                and (
                    runtime_layer.layer_config.resolved_layer_type
                    != layer_config.resolved_layer_type
                )
            ):
                self.remove_layer_by_id(layer_id)
                runtime_layer = None

            if runtime_layer:
                runtime_layer.set_layer_config(layer_config)
            else:
                if had_layers:
                    LOGGER.info('adding layer: id=%r', layer_id)
                runtime_layer = create_runtime_layer(
                    layer_index=layer_index,
                    layer_config=layer_config,
                    layer_id=layer_id,
                    context=self.context
                )
                assert runtime_layer
                self.layer_by_id[layer_id] = runtime_layer
            runtime_layers.append(runtime_layer)
        for runtime_layer in runtime_layers:
            runtime_layer.source_layers = [
                self.layer_by_id[source_layer_config.layer_id]
                for source_layer_config in runtime_layer.layer_config.input_layers
            ]
        seen_layer_ids = {runtime_layer.layer_id for runtime_layer in runtime_layers}
        removed_layer_ids = set(self.layer_by_id.keys()) - seen_layer_ids
        for layer_id in removed_layer_ids:
            self.remove_layer_by_id(layer_id)
        self.output_runtime_layers = [
            runtime_layer
            for runtime_layer in runtime_layers
            if runtime_layer.is_output_layer and runtime_layer.is_enabled
        ]
        LOGGER.debug('output layers: %s', [
            '%s -> %s' % (
                output_runtime_layer.layer_id,
                ', '.join([
                    source_runtime_layer.layer_id
                    for source_runtime_layer in output_runtime_layer.source_layers
                ])
            )
            for output_runtime_layer in self.output_runtime_layers
        ])
        self.context.on_error_layer = self.layer_by_id.get(LayerIds.ON_ERROR)

    def set_app_config(self, app_config: AppConfig):
        LOGGER.info('config: %s', app_config)
        resolved_config = ResolvedAppConfig(app_config)
        LOGGER.info('resolved_config: %s', resolved_config)
        self.set_resolved_app_config(resolved_config)

    def load(self):
        app_config = load_config(self.config_path)
        apply_config_override_map(app_config, self.override_map)
        self.config_modified_time = self.get_config_modified_timestamp()
        self.set_app_config(app_config)

    def get_frame_for_layer(self, runtime_layer: RuntimeLayer):
        source_layers = runtime_layer.source_layers
        assert len(source_layers) == 1
        return next(source_layers[0])

    def next_frame(self):
        self.timer.on_frame_start(initial_step_name='other')
        self.context.frame_cache.clear()
        self.check_reload_config()
        try:
            for output_runtime_layer in self.output_runtime_layers:
                self.timer.on_step_start('other')
                image_array = self.get_frame_for_layer(output_runtime_layer)
                self.timer.on_step_start('out')
                output_runtime_layer.write(image_array)
        except StopIteration:
            return False
        self.timer.on_frame_end()
        return True

    def run(self, max_iterations: int = None):
        try:
            self.timer.start()
            iteration = 0
            while max_iterations is None or iteration < max_iterations:
                if not self.next_frame():
                    break
                iteration += 1
        except KeyboardInterrupt:
            LOGGER.info('exiting')
            self.application_stopped_event.set()
