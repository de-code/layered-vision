import logging
from pathlib import Path

import pytest


@pytest.fixture(scope='session', autouse=True)
def setup_logging():
    for name in ['tests', 'layered_vision']:
        logging.getLogger(name).setLevel('DEBUG')


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    # alias for tmp_path
    return tmp_path
