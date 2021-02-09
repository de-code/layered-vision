from tf_bodypix.utils.v4l2 import VideoLoopbackImageSink

from .api import T_OutputSink


def get_v4l2_output_sink(device_name: str, **__) -> T_OutputSink:
    return VideoLoopbackImageSink(device_name)


OUTPUT_SINK_FACTORY = get_v4l2_output_sink
