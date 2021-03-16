import logging
from contextlib import ExitStack
from functools import partial
from threading import Event
from typing import ContextManager, Dict, Iterable, Optional, List

from .utils.timer import LoggingTimer
from .utils.image import (
    ImageArray,
    ImageSize,
    get_image_size,
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
from .resolved_config import ResolvedAppConfig, ResolvedLayerConfig
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


class RuntimeContext:
    def __init__(
        self,
        layer_by_id: Dict[str, 'RuntimeLayer'],
        timer: LoggingTimer,
        preferred_image_size: ImageSize = None,
        application_stopped_event: Event = None
    ):
        self.layer_by_id = layer_by_id
        self.timer = timer
        self.preferred_image_size = preferred_image_size
        self.frame_cache: dict = {}
        self.application_stopped_event = application_stopped_event


def get_image_source_for_layer_config(
    layer_config: LayerConfig,
    preferred_image_size: Optional[ImageSize],
    stopped_event: Optional[Event]
) -> ContextManager[Iterable[ImageArray]]:
    width = layer_config.get_int('width')
    height = layer_config.get_int('height')
    image_size: Optional[ImageSize]
    if width and height:
        image_size = ImageSize(width=width, height=height)
    else:
        image_size = preferred_image_size
    return get_image_source_for_path(
        layer_config.get_str('input_path'),
        image_size=image_size,
        stopped_event=stopped_event,
        **get_custom_layer_props(layer_config)
    )


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
        self.image_iterator: Optional[T_ImageSource] = None
        self.output_sink: Optional[T_OutputSink] = None
        self.filter: Optional[LayerFilter] = None
        self.context = context

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.exit_stack.__exit__(*args, **kwargs)

    def __repr__(self):
        return '%s(layer_config=%r, ...)' % (
            type(self).__name__,
            self.layer_config
        )

    def get_image_iterator(self) -> T_ImageSource:
        if not self.is_input_layer:
            raise RuntimeError('not an input layer: %r' % self)
        if self.image_iterator is not None:
            return self.image_iterator
        preferred_image_size = self.context.preferred_image_size
        resize_like_id = self.resize_like_id
        if resize_like_id:
            image = next(self.context.layer_by_id[resize_like_id])
            preferred_image_size = get_image_size(image)
        _image_iterator = iter(self.exit_stack.enter_context(
            get_image_source_for_layer_config(
                self.layer_config,
                preferred_image_size=preferred_image_size,
                stopped_event=self.context.application_stopped_event
            )
        ))
        self.image_iterator = _image_iterator
        return _image_iterator

    def get_output_sink(self) -> T_OutputSink:
        if not self.is_output_layer:
            raise RuntimeError('not an output layer: %r' % self)
        output_path = self.layer_config.get_str('output_path')
        assert output_path, "output_path required"
        if self.output_sink is None:
            self.output_sink = self.exit_stack.enter_context(
                get_image_output_sink_for_path(
                    output_path,
                    **get_custom_layer_props(self.layer_config)
                )
            )
        return self.output_sink

    def get_filter(self) -> LayerFilter:
        if self.filter is not None:
            return self.filter
        if not self.is_filter_layer:
            raise RuntimeError('not an output layer')
        _filter = create_filter(
            self.layer_config,
            filter_context=FilterContext(timer=self.context.timer)
        )
        self.filter = _filter
        return _filter

    def __iter__(self):
        return self

    @property
    def source_lazy_image_list(self):
        return LazyImageList([
            partial(next, source_layer)
            for source_layer in self.source_layers
        ])

    def __next__(self):
        try:
            if not self.is_enabled:
                return next(self.source_layers[0])
            if self.is_filter_layer:
                with self.context.timer.enter_step(self.layer_id):
                    return self.get_filter().filter(self.source_lazy_image_list)
            with self.context.timer.enter_step(self.layer_id):
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
        except (StopIteration, LayerException):
            raise
        except Exception as exc:
            raise LayerException('failed to process layer %r due to %r' % (
                self.layer_id, exc
            )) from exc

    def write(self, image_array: ImageArray):
        image_array = apply_alpha(image_array)
        LOGGER.debug('output shape: %s', image_array.shape)
        self.get_output_sink()(image_array)

    @property
    def is_enabled(self) -> bool:
        return self.layer_config.get_bool('enabled', True)

    @property
    def is_no_source(self) -> bool:
        return self.layer_config.get_bool('no_source', False)

    @property
    def is_output_layer(self) -> bool:
        return bool(self.layer_config.props.get('output_path'))

    @property
    def is_input_layer(self) -> bool:
        return bool(self.layer_config.props.get('input_path'))

    @property
    def is_filter_layer(self) -> bool:
        return bool(self.layer_config.props.get('filter'))

    @property
    def resize_like_id(self) -> Optional[str]:
        return self.layer_config.get_str('resize_like_id')

    def add_source_layer(self, source_layer: 'RuntimeLayer'):
        self.source_layers.append(source_layer)


class LayeredVisionApp:
    def __init__(self, config_path: str, override_map: Dict[str, Dict[str, T_Value]] = None):
        self.config_path = config_path
        self.override_map = override_map
        self.exit_stack = ExitStack()
        self.timer = LoggingTimer()
        self.output_sink = None
        self.image_iterator = None
        self.output_runtime_layers: Optional[List[RuntimeLayer]] = None
        self.application_stopped_event = Event()
        self.layer_by_id: Dict[str, RuntimeLayer] = {}
        self.context = RuntimeContext(
            layer_by_id=self.layer_by_id,
            timer=self.timer,
            application_stopped_event=self.application_stopped_event
        )

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

    def reload_config(self):
        self.load()

    def set_resolved_app_config(self, resolved_app_config: ResolvedAppConfig):
        layers = resolved_app_config.layers
        assert len(layers) >= 2
        runtime_layers: List[RuntimeLayer] = []
        for layer_index, layer_config in enumerate(layers):
            layer_id = layer_config.layer_id
            runtime_layer = self.layer_by_id.get(layer_id)
            if not runtime_layer:
                runtime_layer = self.exit_stack.enter_context(RuntimeLayer(
                    layer_index,
                    layer_config,
                    layer_id=layer_id,
                    context=self.context
                ))
                assert runtime_layer
                self.layer_by_id[layer_id] = runtime_layer
            runtime_layers.append(runtime_layer)
        for runtime_layer in runtime_layers:
            runtime_layer.source_layers = [
                self.layer_by_id[source_layer_config.layer_id]
                for source_layer_config in runtime_layer.layer_config.input_layers
            ]
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

    def set_app_config(self, app_config: AppConfig):
        LOGGER.info('config: %s', app_config)
        resolved_config = ResolvedAppConfig(app_config)
        self.set_resolved_app_config(resolved_config)

    def load(self):
        app_config = load_config(self.config_path)
        apply_config_override_map(app_config, self.override_map)
        self.set_app_config(app_config)

    def get_frame_for_layer(self, runtime_layer: RuntimeLayer):
        source_layers = runtime_layer.source_layers
        assert len(source_layers) == 1
        return next(source_layers[0])

    def next_frame(self):
        self.timer.on_frame_start(initial_step_name='other')
        self.context.frame_cache.clear()
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

    def run(self):
        try:
            self.timer.start()
            while self.next_frame():
                pass
        except KeyboardInterrupt:
            LOGGER.info('exiting')
            self.application_stopped_event.set()
