import logging
from contextlib import contextmanager
from itertools import cycle
from time import time, sleep
from typing import ContextManager, Iterable

import cv2
import numpy as np

from .io import get_file
from .image import ImageArray, ImageSize, rgb_to_bgr, bgr_to_rgb, get_image_size


LOGGER = logging.getLogger(__name__)


DEFAULT_WEBCAM_FOURCC = 'MJPG'


def iter_read_raw_video_images(
    video_capture: cv2.VideoCapture,
    repeat: bool = False
) -> Iterable[ImageArray]:
    while True:
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
    for image_array in video_images:
        current_time = time()
        if last_frame_time:
            desired_wait_time = desired_frame_time - (current_time - last_frame_time)
            if desired_wait_time > 0:
                LOGGER.debug('sleeping for desired fps: %s', desired_wait_time)
                sleep(desired_wait_time)
        last_frame_time = time()
        yield image_array


def iter_read_video_images(
    video_capture: cv2.VideoCapture,
    image_size: ImageSize = None,
    interpolation: int = cv2.INTER_LINEAR,
    repeat: bool = False,
    preload: bool = False,
    fps: float = None
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
    else:
        video_images = iter_convert_video_images_to_rgb(iter_resize_video_images(
            iter_read_raw_video_images(video_capture, repeat=repeat),
            image_size=image_size, interpolation=interpolation
        ))
    return iter_delay_video_images_to_fps(video_images, fps)


@contextmanager
def get_video_image_source(
    path: str,
    image_size: ImageSize = None,
    repeat: bool = False,
    preload: bool = False,
    fps: float = None,
    fourcc: str = None,
    **_
) -> ContextManager[Iterable[ImageArray]]:
    local_path = get_file(path)
    if local_path != path:
        LOGGER.info('loading video: %r (downloaded from %r)', local_path, path)
    else:
        LOGGER.info('loading video: %r', path)
    video_capture = cv2.VideoCapture(path)
    if fourcc:
        LOGGER.info('setting video fourcc to %r', fourcc)
        video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
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
    LOGGER.info(
        'video reported image size: %s (%s fps)',
        actual_image_size, actual_fps
    )
    try:
        yield iter_read_video_images(
            video_capture,
            image_size=image_size,
            repeat=repeat,
            preload=preload,
            fps=fps if fps is not None else actual_fps,
        )
    finally:
        LOGGER.debug('releasing video capture: %s', path)
        video_capture.release()


def get_webcam_image_source(
    *args,
    fourcc: str = None,
    **kwargs
) -> ContextManager[Iterable[ImageArray]]:
    if fourcc is None:
        fourcc = DEFAULT_WEBCAM_FOURCC
    return get_video_image_source(*args, fourcc=fourcc, **kwargs)


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
