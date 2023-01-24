import logging

import pytest


@pytest.fixture(scope='session', autouse=True)
def setup_logging():
    for name in ['tests', 'layered_vision']:
        logging.getLogger(name).setLevel('DEBUG')
