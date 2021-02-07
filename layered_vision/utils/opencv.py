import logging
from collections import deque
from contextlib import contextmanager
from itertools import cycle
from time import monotonic, sleep
from threading import Event, Thread
from typing import Callable, ContextManager, Iterable, List, T

import cv2
import numpy as np

from .io import get_file
from .image import ImageArray, ImageSize, rgb_to_bgr, bgr_to_rgb, get_image_size


LOGGER = logging.getLogger(__name__)


DEFAULT_WEBCAM_FOURCC = 'MJPG'


class WaitingDeque:
    def __init__(self, max_length: int):
        self.deque = deque(maxlen=max_length)
        self.changed_event = Event()

    def append(self, data: T):
        self.deque.append(data)
        self.changed_event.set()

    def peek(self, default_value: T = None) -> T:
        try:
            return self.deque[-1]
        except IndexError:
            return default_value

    def pop(self, timeout: float = None) -> T:
        self.changed_event.clear()
        try:
            return self.deque.pop()
        except IndexError:
            pass
        self.changed_event.wait(timeout=timeout)
        return self.deque.pop()


class ReadLatestThreadedReader:
    def __init__(
        self,
        iterable: Iterable[ImageArray],
        stopped_event: Event = None,
        wait_for_data: bool = False
    ):
        self.iterable = iterable
        self.thread = Thread(target=self.read_all_loop, daemon=False)
        self.data_deque = WaitingDeque(max_length=1)
        if stopped_event is None:
            stopped_event = Event()
        self.stopped_event = stopped_event
        self.wait_for_data = wait_for_data

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_, **__):
        self.stop()
        return False

    def __iter__(self):
        return self

    def __next__(self):
        if self.wait_for_data:
            return self.pop()
        return self.peek()

    def start(self):
        self.thread.start()

    def stop(self):
        self.stopped_event.set()

    def peek(self) -> ImageArray:
        while True:
            data = self.data_deque.peek()
            if data is not None:
                return data
            if self.stopped_event.is_set():
                return None
            # wait for first frame (subsequent frames will always be available)
            sleep(0.01)

    def pop(self, timeout: float = None) -> ImageArray:
        LOGGER.debug('waiting for data..')
        return self.data_deque.pop(timeout=timeout)

    def read_all_loop(self):
        while not self.stopped_event.is_set():
            try:
                self.data_deque.append(next(self.iterable))
                LOGGER.debug('read data')
            except StopIteration:
                LOGGER.info('read thread stopped, due to exhausted iterable')
                self.stopped_event.set()
                return
        LOGGER.info('reader thread stopped, due to event')
        self.stopped_event.set()


def iter_read_threaded(iterable: Iterable[T], **kwargs) -> Iterable[T]:
    with ReadLatestThreadedReader(iterable, **kwargs) as reader:
        yield from reader


def iter_read_raw_video_images(
    video_capture: cv2.VideoCapture,
    repeat: bool = False,
    is_stopped: Callable[[], bool] = None
) -> Iterable[ImageArray]:
    while is_stopped is None or not is_stopped():
        grabbed, image_array = video_capture.read()
        if not grabbed:
            LOGGER.info('video end reached')
            if not repeat:
                return
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            grabbed, image_array = video_capture.read()
            if not grabbed:
                LOGGER.info('unable to rewind video')
                return
        yield image_array


def iter_read_raw_video_images_threaded(
    video_capture: cv2.VideoCapture,
    repeat: bool = False,
    is_stopped: Callable[[], bool] = None
) -> Iterable[ImageArray]:
    iterable = iter_read_raw_video_images(
        video_capture=video_capture,
        repeat=repeat,
        is_stopped=is_stopped
    )
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    if fps:
        LOGGER.info('fps: %s', fps)
        iterable = iter_delay_video_images_to_fps(iterable, fps=fps)
    with ReadLatestThreadedReader(iterable) as reader:
        yield from reader


def iter_resize_video_images(
    video_images: Iterable[ImageArray],
    image_size: ImageSize = None,
    interpolation: int = cv2.INTER_LINEAR
) -> Iterable[ImageArray]:
    is_first = True
    for image_array in video_images:
        LOGGER.debug('video image_array.shape: %s', image_array.shape)
        if is_first:
            LOGGER.info(
                'received video image shape: %s (requested: %s)',
                image_array.shape, image_size
            )
            is_first = False
        if image_size and get_image_size(image_array) != image_size:
            image_array = cv2.resize(
                image_array,
                (image_size.width, image_size.height),
                interpolation=interpolation
            )
        yield image_array


def iter_convert_video_images_to_rgb(
    video_images: Iterable[ImageArray]
) -> Iterable[ImageArray]:
    return (bgr_to_rgb(image_array) for image_array in video_images)


def iter_delay_video_images_to_fps(
    video_images: Iterable[ImageArray],
    fps: float = None
) -> Iterable[np.ndarray]:
    if not fps or fps <= 0:
        yield from video_images
        return
    desired_frame_time = 1 / fps
    last_frame_time = None
    frame_times = deque(maxlen=10)
    current_fps = 0
    additional_frame_adjustment = 0
    end_frame_time = monotonic()
    while True:
        start_frame_time = end_frame_time
        # attempt to retrieve the next frame (that may vary in time)
        try:
            image_array = next(video_images)
        except StopIteration:
            return
        # wait time until delivery in order to achieve a similar fps
        current_time = monotonic()
        if last_frame_time:
            desired_wait_time = (
                desired_frame_time
                - (current_time - last_frame_time)
                + additional_frame_adjustment
            )
            if desired_wait_time > 0:
                LOGGER.debug(
                    'sleeping for desired fps: %s (desired_frame_time: %s, fps: %.3f)',
                    desired_wait_time, desired_frame_time, current_fps
                )
                sleep(desired_wait_time)
        last_frame_time = monotonic()
        # emit the frame (post processing may add to the overall)
        yield image_array
        end_frame_time = monotonic()
        frame_time = end_frame_time - start_frame_time
        additional_frame_adjustment = desired_frame_time - frame_time
        frame_times.append(frame_time)
        current_fps = 1 / (sum(frame_times) / len(frame_times))


def iter_read_video_images(
    video_capture: cv2.VideoCapture,
    image_size: ImageSize = None,
    interpolation: int = cv2.INTER_LINEAR,
    repeat: bool = False,
    preload: bool = False,
    read_in_thread: bool = True,
    fps: float = None,
    stopped_event: Event = None
) -> Iterable[np.ndarray]:
    if preload:
        LOGGER.info('preloading video images')
        preloaded_video_images = list(
            iter_convert_video_images_to_rgb(iter_resize_video_images(
                iter_read_raw_video_images(video_capture, repeat=False),
                image_size=image_size, interpolation=interpolation
            ))
        )
        if repeat:
            video_images = cycle(preloaded_video_images)
        return iter_delay_video_images_to_fps(video_images, fps)
    video_images = iter_read_raw_video_images(video_capture, repeat=repeat)
    video_images = iter_delay_video_images_to_fps(video_images, fps)
    if read_in_thread:
        video_images = iter_read_threaded(
            video_images, stopped_event=stopped_event
        )
    video_images = iter_resize_video_images(
        video_images, image_size=image_size, interpolation=interpolation
    )
    video_images = iter_convert_video_images_to_rgb(video_images)
    return video_images


@contextmanager
def get_video_image_source(
    path: str,
    image_size: ImageSize = None,
    repeat: bool = False,
    preload: bool = False,
    download: bool = True,
    fps: float = None,
    fourcc: str = None,
    buffer_size: int = None,
    stopped_event: Event = None,
    **_
) -> ContextManager[Iterable[ImageArray]]:
    local_path = get_file(path, download=download)
    if local_path != path:
        LOGGER.info('loading video: %r (downloaded from %r)', local_path, path)
    else:
        LOGGER.info('loading video: %r', path)
    video_capture = cv2.VideoCapture(local_path)
    if fourcc:
        LOGGER.info('setting video fourcc to %r', fourcc)
        video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if buffer_size:
        video_capture.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
    if image_size:
        LOGGER.info('attempting to set video image size to: %s', image_size)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, image_size.width)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, image_size.height)
    if fps:
        LOGGER.info('attempting to set video fps to %r', fps)
        video_capture.set(cv2.CAP_PROP_FPS, fps)
    actual_image_size = ImageSize(
        width=video_capture.get(cv2.CAP_PROP_FRAME_WIDTH),
        height=video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    actual_fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    LOGGER.info(
        'video reported image size: %s (%s fps, %s frames)',
        actual_image_size, actual_fps, frame_count
    )
    if preload and frame_count <= 0:
        LOGGER.info('disabling preload for video source with unknown frame count')
        preload = False
    try:
        yield iter_read_video_images(
            video_capture,
            image_size=image_size,
            repeat=repeat,
            preload=preload,
            fps=fps if fps is not None else actual_fps,
            stopped_event=stopped_event
        )
    finally:
        LOGGER.debug('releasing video capture: %s', path)
        video_capture.release()


def get_webcam_image_source(
    *args,
    fourcc: str = None,
    buffer_size: int = 1,
    **kwargs
) -> ContextManager[Iterable[ImageArray]]:
    if fourcc is None:
        fourcc = DEFAULT_WEBCAM_FOURCC
    return get_video_image_source(*args, fourcc=fourcc, buffer_size=buffer_size, **kwargs)


def get_pafy_video(url: str) -> 'pafy.BasePafy':
    # lazy import to make it optional
    import pafy  # pylint: disable=import-outside-toplevel

    return pafy.new(url)


def get_best_matching_video_stream(
    streams: List['pafy.BaseStream'],
    preferred_type: str,
    image_size: ImageSize = None
) -> 'pafy.BaseStream':
    streams_with_dimensions_difference = []
    for stream in streams:
        if not image_size:
            error = 0
        else:
            stream_width, stream_height = stream.dimensions
            error = (
                abs(stream_width - image_size.width) ** 2
                + abs(stream_height - image_size.height) ** 2
            )
        streams_with_dimensions_difference.append((error, stream,))

    sorted_streams_with_dimensions_difference = sorted(streams_with_dimensions_difference)
    for error, stream in sorted_streams_with_dimensions_difference:
        if preferred_type and stream.extension != preferred_type:
            LOGGER.debug(
                'skipping stream with not prefeered type: %r (preferred: %r)',
                stream.extension, preferred_type
            )
            continue
        LOGGER.debug('choosing stream with error=%r (%r)', error, stream)
        return stream
    return None


def get_youtube_stream_url(
    url: str,
    preferred_type: str,
    image_size: ImageSize = None
) -> str:
    video = get_pafy_video(url)
    LOGGER.info('found video (%r): %r', url, video.title)
    LOGGER.info('available video streams: %s', [
        '%s (%s)' % (stream.resolution, stream.extension)
        for stream in video.streams
    ])
    stream = get_best_matching_video_stream(
        video.streams,
        preferred_type=preferred_type,
        image_size=image_size
    )
    if not stream:
        raise ValueError('no stream found for preferred_type=%r' % preferred_type)
    LOGGER.info(
        'video stream: %r (%s) (requested: %s)',
        stream.resolution, stream.extension,
        image_size
    )
    LOGGER.debug('video stream url: %r', stream.url)
    return stream.url


def get_youtube_video_image_source(
    path: str,
    *args,
    image_size: ImageSize = None,
    download: bool = False,
    preload: bool = False,
    buffer_size: int = 20,
    preferred_type: str = 'mp4',
    **kwargs
) -> ContextManager[Iterable[ImageArray]]:
    # no download support for youtube, also don't preload as those might be streams
    download = False
    preload = False
    return get_video_image_source(
        get_youtube_stream_url(path, preferred_type=preferred_type, image_size=image_size),
        *args,
        image_size=image_size,
        download=download,
        preload=preload,
        buffer_size=buffer_size,
        **kwargs
    )


class ShowImageSink:
    def __init__(self, window_name: str):
        self.window_name = window_name

    def __enter__(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 600, 600)
        return self

    def __exit__(self, *_, **__):
        cv2.destroyAllWindows()

    def __call__(self, image_array: ImageArray):
        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) <= 0:
            LOGGER.info('window closed')
            raise KeyboardInterrupt('window closed')
        image_array = np.asarray(image_array).astype(np.uint8)
        cv2.imshow(self.window_name, rgb_to_bgr(image_array))
        cv2.waitKey(1)
