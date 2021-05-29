import logging

import numpy as np
import numpy.ma as ma

from layered_vision.config import LayerConfig
from layered_vision.filters.api import FilterContext
from layered_vision.filters.fill import FillFilter


LOGGER = logging.getLogger(__name__)


class TestFillFilter:
    def test_should_fill_whole_image(
        self,
        filter_context: FilterContext
    ):
        layer_config = LayerConfig.from_json({
            'color': 'red'
        })
        image_filter = FillFilter(layer_config=layer_config, filter_context=filter_context)
        source_image = np.zeros((5, 4, 3), dtype=np.float32)
        result_image = image_filter.filter(source_image)
        LOGGER.debug('result_image:\n%s', result_image)
        assert result_image.shape == source_image.shape
        assert np.all(result_image == (255, 0, 0))

    def test_should_fill_area_using_poly_points(
        self,
        filter_context: FilterContext
    ):
        layer_config = LayerConfig.from_json({
            'color': 'red',
            'poly_points': [
                (0.1, 0.1), (0.6, 0.1),  # top left, top right
                (0.6, 0.4), (0.1, 0.4)  # bottom right, bottom left
            ]
        })
        image_filter = FillFilter(layer_config=layer_config, filter_context=filter_context)
        source_image = np.zeros((10, 10, 3), dtype=np.float32)
        result_image = image_filter.filter(source_image)
        LOGGER.debug('result_image:\n%s', result_image)
        mask = np.zeros(source_image.shape)
        mask[1:5, 1:7] = 1
        masked_result_image = ma.masked_array(result_image, mask)
        poly_area = result_image[1:5, 1:7]
        LOGGER.debug('poly_area:\n%s', poly_area)
        assert result_image.shape == source_image.shape
        assert np.all(poly_area == (255, 0, 0))
        LOGGER.debug('mask:\n%s', mask)
        LOGGER.debug('masked_result_image:\n%s', masked_result_image)
        assert np.all(masked_result_image == (0, 0, 0))
