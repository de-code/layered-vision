import pytest

from layered_vision.utils.timer import LoggingTimer
from layered_vision.filters.api import FilterContext


@pytest.fixture()
def filter_context() -> FilterContext:
    return FilterContext(LoggingTimer())
