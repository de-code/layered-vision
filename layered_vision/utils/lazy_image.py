from abc import ABC, abstractmethod
from typing import Callable, List, Optional

from .image import ImageArray


T_ImageFactory = Callable[[], ImageArray]


class LazyImageProxy(ABC):
    @abstractmethod
    def get_image(self):
        pass

    @property
    def image(self):
        return self.get_image()

    @property
    def dtype(self):
        return self.image.dtype

    @property
    def shape(self):
        return self.image.shape

    def __len__(self):
        return len(self.image)

    def __getitem__(self, *args):
        return self.image.__getitem__(*args)


class LazyImage(LazyImageProxy):
    def __init__(self, factory: T_ImageFactory):
        self.factory = factory
        self._image: Optional[ImageArray] = None

    def get_image(self):
        if self._image is None:
            self._image = self.factory()
        return self._image


class LazyImageList(LazyImageProxy):
    def __init__(self, factories: List[T_ImageFactory]):
        self.images: List[LazyImage] = [
            LazyImage(factory)
            for factory in factories
        ]

    def get_image(self):
        return self.images[-1].image


def resolve_lazy_image(image) -> ImageArray:
    try:
        return image.image
    except AttributeError:
        return image
