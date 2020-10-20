from layered_vision.cli import main


class TestMain:
    def test_should_not_fail(self):
        main(['start'])
