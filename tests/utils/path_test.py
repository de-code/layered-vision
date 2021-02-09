from layered_vision.utils.path import parse_type_path


class TestParseTypePath:
    def test_should_should_parse_relative_video_path(self):
        assert parse_type_path('video:relative/path') == ('video', 'relative/path')

    def test_should_should_parse_absolute_video_path(self):
        assert parse_type_path('video:/relative/path') == ('video', '/relative/path')

    def test_should_should_parse_remote_video_url(self):
        assert parse_type_path('video:https://host/path') == ('video', 'https://host/path')

    def test_should_should_parse_source_type_with_colon_and_no_path(self):
        assert parse_type_path('video:') == ('video', '')

    def test_should_should_parse_source_type_only(self):
        assert parse_type_path('video') == ('video', '')
