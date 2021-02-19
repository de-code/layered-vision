import logging
from contextlib import contextmanager
from threading import Event
from typing import Iterable, Iterator

import mss
from mss.base import MSSBase
import numpy as np

from layered_vision.utils.image import ImageArray, ImageSize, apply_alpha, bgr_to_rgb
from layered_vision.utils.opencv import iter_resize_video_images
from layered_vision.sources.api import T_ImageSourceFactory


LOGGER = logging.getLogger(__name__)


def iter_screen_grab(
    sct: MSSBase,
    grab_params: dict,
    stopped_event: Event = None
) -> Iterable[ImageArray]:
    if stopped_event is None:
        stopped_event = Event()
    while not stopped_event.is_set():
        yield bgr_to_rgb(apply_alpha(np.asarray(sct.grab(grab_params))))


@contextmanager
def get_mss_video_image_source(
    path: str,  # pylint: disable=unused-argument
    *args,
    image_size: ImageSize = None,
    stopped_event: Event = None,
    init_params: dict = None,
    grab_params: dict = None,
    **_
) -> Iterator[Iterable[ImageArray]]:
    LOGGER.info('constructing mss with %r', init_params or {})
    with mss.mss(**(init_params or {})) as sct:
        _grab_params = grab_params or {}
        _grab_params = {
            **sct.monitors[_grab_params.get('mon', 1)],
            **_grab_params
        }
        LOGGER.info('mss grab_params: %r', _grab_params)
        yield iter_resize_video_images(
            iter_screen_grab(sct, _grab_params, stopped_event=stopped_event),
            image_size=image_size
        )


IMAGE_SOURCE_FACTORY: T_ImageSourceFactory = get_mss_video_image_source
