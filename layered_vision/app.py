import logging
from contextlib import ExitStack
from typing import List

from .utils.timer import LoggingTimer
from .utils.image import (
    ImageArray,
    ImageSize
)

from .sinks import (
    T_OutputSink,
    get_image_output_sink_for_path
)

from .config import load_config, LayerConfig
from .sources import get_image_source_for_path, T_ImageSource


LOGGER = logging.getLogger(__name__)


class RuntimeContext:
    def __init__(
        self,
        timer: LoggingTimer
    ):
        self.timer = timer
        self.frame_cache = {}


class RuntimeLayer:
    def __init__(
        self,
        layer_index: int,
        layer_config: LayerConfig,
        layer_id: str,
        context: RuntimeContext,
        source_layers: List['RuntimeLayer'] = None
    ):
        self.layer_index = layer_index
        self.layer_config = layer_config
        self.layer_id = layer_id
        self.exit_stack = ExitStack()
        self.source_layers = (source_layers or []).copy()
        self.image_iterator = None
        self.output_sink = None
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
        if self.image_iterator is None:
            width = self.layer_config.get('width')
            height = self.layer_config.get('height')
            if width and height:
                image_size = ImageSize(width=width, height=height)
            else:
                image_size = None
            self.image_iterator = iter(self.exit_stack.enter_context(
                get_image_source_for_path(
                    self.layer_config.get('input_path'),
                    image_size=image_size,
                    repeat=self.layer_config.get('repeat'),
                    fps=self.layer_config.get('fps')
                )
            ))
        return self.image_iterator

    def get_output_sink(self) -> T_OutputSink:
        if not self.is_output_layer:
            raise RuntimeError('not an output layer: %r' % self)
        if self.output_sink is None:
            self.output_sink = self.exit_stack.enter_context(
                get_image_output_sink_for_path(self.layer_config.get('output_path'))
            )
        return self.output_sink

    def __next__(self):
        self.context.timer.on_step_start(self.layer_id)
        image_array = self.context.frame_cache.get(self.layer_id)
        if image_array is None:
            image_array = next(self.get_image_iterator())
            self.context.frame_cache[self.layer_id] = image_array
        return image_array

    def write(self, image_array: ImageArray):
        LOGGER.debug('output shape: %s', image_array.shape)
        self.get_output_sink()(image_array)

    @property
    def is_output_layer(self) -> bool:
        return bool(self.layer_config.props.get('output_path'))

    @property
    def is_input_layer(self) -> bool:
        return bool(self.layer_config.props.get('input_path'))

    def add_source_layer(self, source_layer: 'RuntimeLayer'):
        self.source_layers.append(source_layer)


def get_source_layer_index(
    all_runtime_layers: List[RuntimeLayer],
    target_layer: RuntimeLayer
):
    source_index = target_layer.layer_index - 1
    while all_runtime_layers[source_index].is_output_layer:
        source_index -= 1
    assert source_index >= 0
    return source_index


def add_source_layers_recursively(
    all_runtime_layers: List[RuntimeLayer],
    target_layer: RuntimeLayer
):
    if target_layer.source_layers:
        return
    source_layer_index = get_source_layer_index(
        all_runtime_layers,
        target_layer
    )
    source_layer = all_runtime_layers[source_layer_index]
    target_layer.add_source_layer(source_layer)
    if not source_layer.is_input_layer:
        add_source_layers_recursively(
            all_runtime_layers,
            source_layer
        )


class LayeredVisionApp:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.exit_stack = ExitStack()
        self.timer = LoggingTimer()
        self.config = None
        self.output_sink = None
        self.image_iterator = None
        self.output_runtime_layers = None
        self.context = RuntimeContext(
            timer=self.timer
        )

    def __enter__(self):
        try:
            self.load()
            return self
        except Exception as exc:
            self.exit_stack.__exit__(type(exc), exc, None)
            raise exc

    def __exit__(self, *args, **kwargs):
        self.exit_stack.__exit__(*args, **kwargs)

    def load(self):
        self.config = load_config(self.config_path)
        LOGGER.info('config: %s', self.config)
        layers = self.config.layers
        assert len(layers) >= 2
        runtime_layers = [
            self.exit_stack.enter_context(RuntimeLayer(
                layer_index, layer_config,
                layer_id=layer_config.get('id') or 'l%d' % layer_index,
                context=self.context
            ))
            for layer_index, layer_config in enumerate(layers)
        ]
        self.output_runtime_layers = [
            runtime_layer
            for runtime_layer in runtime_layers
            if runtime_layer.is_output_layer
        ]
        for output_layer in self.output_runtime_layers:
            add_source_layers_recursively(runtime_layers, output_layer)
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
