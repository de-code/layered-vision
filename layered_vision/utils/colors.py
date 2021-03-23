from typing import Optional

import numpy as np


NAMED_COLORS = {
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255)
}


def get_color_numpy_array(color) -> Optional[np.ndarray]:
    if color:
        if color in NAMED_COLORS:
            return np.asarray(NAMED_COLORS[color])
        if isinstance(color, (tuple, list,)):
            return np.asarray(color)
        raise ValueError('unsupported color value type: %r (%r)' % (type(color), color))
    return None
