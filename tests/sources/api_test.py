from layered_vision.sources.api import (
    is_youtube_path,
    parse_source_type_path
)


class TestIsYoutubePath:
    def test_should_return_false_for_local_path_containing_youtube(self):
        assert is_youtube_path('/path/to/youtube/here') is False

    def test_should_return_false_for_https_path_containing_youtube_in_path(self):
        assert is_youtube_path('https://host/path/to/youtube/here') is False

    def test_should_return_true_for_https_www_youtube_com_host(self):
        assert is_youtube_path('https://www.youtube.com/watch?v=12345') is True

    def test_should_return_true_for_https_youtu_be_host(self):
        assert is_youtube_path('https://youtu.be/watch?v=12345') is True


class TestParseSourceTypePath:
    def test_should_should_parse_relative_video_path(self):
        assert parse_source_type_path('video:relative/path') == ('video', 'relative/path')

    def test_should_should_parse_absolute_video_path(self):
        assert parse_source_type_path('video:/relative/path') == ('video', '/relative/path')

    def test_should_should_parse_remote_video_url(self):
        assert parse_source_type_path('video:https://host/path') == ('video', 'https://host/path')

    def test_should_should_parse_source_type_with_colon_and_no_path(self):
        assert parse_source_type_path('video:') == ('video', '')

    def test_should_should_parse_source_type_only(self):
        assert parse_source_type_path('video') == ('video', '')
