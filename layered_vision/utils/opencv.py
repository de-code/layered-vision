import logging
from time import time, sleep
from contextlib import contextmanager
from typing import ContextManager, Iterable

import cv2
import numpy as np

from .image import ImageArray, ImageSize, rgb_to_bgr, bgr_to_rgb, get_image_size


LOGGER = logging.getLogger(__name__)


def iter_read_video_images(
    video_capture: cv2.VideoCapture,
    image_size: ImageSize = None,
    interpolation: int = cv2.INTER_LINEAR,
    fps: float = None,
    repeat: bool = None
) -> Iterable[np.ndarray]:
    is_first = True
    desired_frame_time = 1 / fps if fps else 0
    last_frame_time = None
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
        LOGGER.debug('video image_array.shape: %s', image_array.shape)
        if is_first:
            LOGGER.info(
                'received video image shape: %s (requested: %s)',
                image_array.shape, image_size
            )
        if image_size and get_image_size(image_array) != image_size:
            image_array = cv2.resize(
                image_array,
                (image_size.width, image_size.height),
                interpolation=interpolation
            )
        rgb_image_array = bgr_to_rgb(image_array)
        current_time = time()
        if last_frame_time:
            desired_wait_time = desired_frame_time - (current_time - last_frame_time)
            if desired_wait_time > 0:
                LOGGER.debug('sleeping for desired fps: %s', desired_wait_time)
                sleep(desired_wait_time)
        last_frame_time = time()
        yield rgb_image_array
        is_first = False


@contextmanager
def get_video_image_source(
    path: str,
    image_size: ImageSize = None,
    repeat: bool = None
) -> ContextManager[Iterable[np.ndarray]]:
    LOGGER.info('loading video: %r', path)
    video_capture = cv2.VideoCapture(path)
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
            fps=actual_fps,
            repeat=repeat
        )
    finally:
        LOGGER.debug('releasing video capture: %s', path)
        video_capture.release()


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
