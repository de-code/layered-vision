import numpy as np

from layered_vision.utils.image import (
    ImageArray,
    has_alpha,
    has_transparent_alpha,
    get_image_with_alpha,
    combine_images
)


IMAGE_SHAPE = (2, 2, 3)

IMAGE_DATA_1 = np.full(IMAGE_SHAPE, 10)
IMAGE_DATA_2 = np.full(IMAGE_SHAPE, 20)
IMAGE_DATA_3 = np.full(IMAGE_SHAPE, 30)


def add_alpha_channel(image: ImageArray, alpha: float) -> ImageArray:
    return get_image_with_alpha(
        image,
        (np.ones(image.shape[:-1]) * 255 * alpha).astype(np.uint8)
    )


class TestHasAlpha:
    def test_should_return_false_for_image_without_alpha_channel(self):
        assert has_alpha(IMAGE_DATA_1) is False

    def test_should_return_true_for_image_with_alpha_channel(self):
        assert has_alpha(add_alpha_channel(IMAGE_DATA_1, 1.0)) is True


class TestHasTransparentAlpha:
    def test_should_return_false_for_image_without_alpha_channel(self):
        assert not has_transparent_alpha(IMAGE_DATA_1)

    def test_should_return_false_for_image_with_alpha_equal_to_one(self):
        assert not has_transparent_alpha(add_alpha_channel(IMAGE_DATA_1, 1.0))

    def test_should_return_true_for_image_with_alpha_less_than_one(self):
        assert has_transparent_alpha(add_alpha_channel(IMAGE_DATA_1, 0.5))


class TestCombineImages:
    def test_should_return_none_if_received_empty_list(self):
        assert combine_images([]) is None

    def test_should_return_passed_in_image_if_received_single_image(self):
        np.testing.assert_allclose(combine_images([IMAGE_DATA_1]), IMAGE_DATA_1)

    def test_should_return_passed_in_image_with_alpha_if_received_single_image(self):
        image = add_alpha_channel(IMAGE_DATA_1, 0.5)
        np.testing.assert_allclose(combine_images([
            image
        ]), image)

    def test_should_return_last_image_if_without_alpha(self):
        np.testing.assert_allclose(combine_images([
            IMAGE_DATA_1,
            IMAGE_DATA_2,
            IMAGE_DATA_3
        ]), IMAGE_DATA_3)

    def test_should_overlay_image_with_alpha_on_image_without_alpha(self):
        np.testing.assert_allclose(combine_images([
            IMAGE_DATA_1,
            add_alpha_channel(IMAGE_DATA_2, 0.5)
        ]), IMAGE_DATA_1 * 0.5 + IMAGE_DATA_2 * 0.5, rtol=0.1)

    def test_should_overlay_multiple_images_with_alpha_on_image_without_alpha(self):
        np.testing.assert_allclose(combine_images([
            IMAGE_DATA_1,
            add_alpha_channel(IMAGE_DATA_2, 0.5),
            add_alpha_channel(IMAGE_DATA_3, 0.5)
        ], fixed_alpha_enabled=True), (
            (IMAGE_DATA_1 * 0.5 + IMAGE_DATA_2 * 0.5) * 0.5
            + IMAGE_DATA_3 * 0.5
        ), rtol=0.1)

    def test_should_overlay_multiple_images_with_alpha_on_image_without_alpha_using_fixed_alpha(
        self
    ):
        np.testing.assert_allclose(combine_images([
            IMAGE_DATA_1,
            add_alpha_channel(IMAGE_DATA_2, 0.5),
            add_alpha_channel(IMAGE_DATA_3, 0.5)
        ], fixed_alpha_enabled=True), (
            (IMAGE_DATA_1 * 0.5 + IMAGE_DATA_2 * 0.5) * 0.5
            + IMAGE_DATA_3 * 0.5
        ), rtol=0.1)
