import pytest

import numpy as np

from layered_vision.utils.image import (
    ImageArray,
    has_alpha,
    has_transparent_alpha,
    get_image_with_alpha,
    safe_multiply,
    combine_images,
    combine_images_or_none
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


class TestSafeMultiply:
    def test_should_multiply_float32_with_float32_array(self):
        image1 = np.array([1, 2, 3], dtype=np.float32)
        image2 = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        assert safe_multiply(
            image1, image2, out=image1
        ).tolist() == [0.5, 1.0, 1.5]

    def test_should_multiply_int_with_float32_array(self):
        image1 = np.array([1, 2, 3], dtype=np.int)
        image2 = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        assert safe_multiply(
            image1, image2, out=image1
        ).tolist() == [0.5, 1.0, 1.5]


class TestCombineImages:
    def test_should_return_raise_exception_for_empty_list(self):
        with pytest.raises(AssertionError):
            combine_images([])

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

    def test_should_not_fail_using_fixed_alpha_enabled_and_float32_dtype(self):
        image_data_3_with_multiple_alpha = add_alpha_channel(IMAGE_DATA_3, 0.5)
        image_data_3_with_multiple_alpha[0, 0, 3] = 123
        combine_images([
            IMAGE_DATA_1.astype(np.uint8),
            add_alpha_channel(IMAGE_DATA_2, 0.5).astype(np.uint8),
            image_data_3_with_multiple_alpha.astype(np.float32)
        ], fixed_alpha_enabled=True)


class TestCombineImagesOrNone:
    def test_should_return_none_if_received_empty_list(self):
        assert combine_images_or_none([]) is None

    def test_should_return_passed_in_image_if_received_single_image(self):
        np.testing.assert_allclose(combine_images_or_none([IMAGE_DATA_1]), IMAGE_DATA_1)
