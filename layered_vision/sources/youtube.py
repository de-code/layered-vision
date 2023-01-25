import logging
import time
from typing import ContextManager, Iterable, List, Optional

import youtube_dl

import pafy
import pafy.g
from pafy.backend_youtube_dl import YtdlPafy

from layered_vision.utils.image import ImageArray, ImageSize
from layered_vision.utils.opencv import get_video_image_source


LOGGER = logging.getLogger(__name__)


class PatchedYtdlPafy(YtdlPafy):
    # patch _fetch_basic function due to missing dislike_count
    def _fetch_basic(self):
        """ Fetch basic data and streams. """
        if self._have_basic:
            return

        with youtube_dl.YoutubeDL(self._ydl_opts) as ydl:
            try:
                self._ydl_info = ydl.extract_info(self.videoid, download=False)
            # Turn into an IOError since that is what pafy previously raised
            except youtube_dl.utils.DownloadError as exc:
                raise IOError(str(exc).replace('YouTube said', 'Youtube says')) from exc

        if self.callback:
            self.callback("Fetched video info")

        self._title = self._ydl_info['title']
        self._author = self._ydl_info['uploader']
        self._rating = self._ydl_info['average_rating']
        self._length = self._ydl_info['duration']
        self._viewcount = self._ydl_info['view_count']
        self._likes = self._ydl_info.get('like_count', 0)
        self._dislikes = self._ydl_info.get('dislike_count', 0)
        self._username = self._ydl_info['uploader_id']
        self._category = self._ydl_info['categories'][0] if self._ydl_info['categories'] else ''
        self._bestthumb = self._ydl_info['thumbnails'][0]['url']
        self._bigthumb = pafy.g.urls['bigthumb'] % self.videoid
        self._bigthumbhd = pafy.g.urls['bigthumbhd'] % self.videoid
        self.expiry = time.time() + pafy.g.lifespan

        self._have_basic = True


def get_pafy_video(url: str) -> pafy.backend_shared.BasePafy:
    return PatchedYtdlPafy(url)


def get_best_matching_video_stream(
    streams: List[pafy.backend_shared.BaseStream],
    preferred_type: str,
    image_size: Optional[ImageSize] = None
) -> pafy.backend_shared.BaseStream:
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

    sorted_streams_with_dimensions_difference = sorted(
        streams_with_dimensions_difference,
        key=lambda t: t[0]
    )
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
    image_size: Optional[ImageSize] = None
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
    image_size: Optional[ImageSize] = None,
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
        image_size=image_size,
        download=download,
        preload=preload,
        buffer_size=buffer_size,
        **kwargs
    )


IMAGE_SOURCE_FACTORY = get_youtube_video_image_source
