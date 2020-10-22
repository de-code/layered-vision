import logging
from contextlib import contextmanager
from typing import ContextManager, Iterable

import cv2
import numpy as np

from .image import ImageArray, ImageSize, rgb_to_bgr, bgr_to_rgb, get_image_size


LOGGER = logging.getLogger(__name__)


def iter_read_video_images(
    video_capture: cv2.VideoCapture,
    image_size: ImageSize = None,
    interpolation: int = cv2.INTER_LINEAR
) -> Iterable[np.ndarray]:
    is_first = True
    while True:
        grabbed, image_array = video_capture.read()
        if not grabbed:
            LOGGER.info('video end reached')
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
        yield bgr_to_rgb(image_array)
        is_first = False


@contextmanager
def get_video_image_source(
    path: str,
    image_size: ImageSize = None
) -> ContextManager[Iterable[np.ndarray]]:
    video_capture = cv2.VideoCapture(path)
    try:
        yield iter_read_video_images(video_capture, image_size=image_size)
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
