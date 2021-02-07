from time import sleep
from typing import Tuple

import pafy

from layered_vision.utils.image import ImageSize

from layered_vision.utils.opencv import (
    ReadLatestThreadedReader,
    get_best_matching_video_stream
)


def get_pafy_stream(
    dimensions: Tuple[int, int],
    extension: str
) -> 'pafy.backend_shared.BaseStream':
    stream = pafy.backend_shared.BaseStream('parent')
    stream._dimensions = dimensions  # pylint: disable=protected-access
    stream._extension = extension  # pylint: disable=protected-access
    return stream


class TestReadLatestThreadedReader:
    def test_should_return_last_read_item_using_peek(self):
        data_list = ['abc', 'def']
        with ReadLatestThreadedReader(iter(data_list)) as reader:
            # adding delay so that it reads to the last item
            sleep(0.01)
            peeked_data = reader.peek()
        assert peeked_data == data_list[-1]

    def test_should_return_last_read_item_using_pop(self):
        data_list = ['abc', 'def']
        with ReadLatestThreadedReader(iter(data_list)) as reader:
            # adding delay so that it reads to the last item
            sleep(0.01)
            peeked_data = reader.pop()
        assert peeked_data == data_list[-1]


class TestGetBestMatchingVideoStream:
    def test_should_select_stream_with_close_resolution(self):
        streams = [
            get_pafy_stream(dimensions=(100, 100), extension='mp4'),
            get_pafy_stream(dimensions=(200, 200), extension='mp4'),
            get_pafy_stream(dimensions=(400, 400), extension='mp4')
        ]
        result = get_best_matching_video_stream(
            streams,
            preferred_type='mp4',
            image_size=ImageSize(190, 190)
        )
        assert result == streams[1]

    def test_should_select_stream_with_close_resolution_and_extension(self):
        streams = [
            get_pafy_stream(dimensions=(100, 100), extension='mp4'),
            get_pafy_stream(dimensions=(190, 190), extension='other'),
            get_pafy_stream(dimensions=(200, 200), extension='mp4'),
            get_pafy_stream(dimensions=(400, 400), extension='mp4')
        ]
        result = get_best_matching_video_stream(
            streams,
            preferred_type='mp4',
            image_size=ImageSize(190, 190)
        )
        assert result == streams[2]
