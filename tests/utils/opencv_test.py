from time import sleep

from layered_vision.utils.opencv import (
    ReadLatestThreadedReader
)


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
