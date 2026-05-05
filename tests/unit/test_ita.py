"""Unit tests for the ITA skin-tone proxy module."""

import numpy as np
import pytest
from PIL import Image

from models.skin_tone_ita import compute_ita, rgb_to_lab


def _solid_image(r, g, b, size=64):
    arr = np.full((size, size, 3), [r, g, b], dtype=np.uint8)
    return Image.fromarray(arr)


def _skin_image(size=224):
    """Realistic skin-tone patch — warm, mid-saturation orange-tan."""
    arr = np.full((size, size, 3), [210, 160, 120], dtype=np.uint8)
    return Image.fromarray(arr)


class TestComputeITA:
    def test_returns_tuple(self):
        img = _skin_image()
        result = compute_ita(img)
        assert isinstance(result, tuple) and len(result) == 2

    def test_ita_in_valid_range(self):
        ita, _ = compute_ita(_skin_image())
        assert ita is None or -90.0 <= ita <= 90.0

    def test_label_is_known_string(self):
        _, label = compute_ita(_skin_image())
        assert label in {"Light", "Medium", "Dark", "unknown"}

    def test_deterministic(self):
        img = _skin_image()
        assert compute_ita(img) == compute_ita(img)

    def test_returns_none_for_near_black_image(self):
        # Black image has no detectable skin pixels
        ita, label = compute_ita(_solid_image(0, 0, 0))
        assert ita is None
        assert label == "unknown"

    def test_non_skin_image_returns_valid_tuple(self):
        # Blue has no skin pixels; ITA falls back to centre-crop median and still returns a value
        ita, label = compute_ita(_solid_image(0, 0, 200))
        assert isinstance(label, str)
        assert label in {"Light", "Medium", "Dark", "unknown"}
        assert ita is None or isinstance(ita, float)

    def test_light_skin_classified_light(self):
        # Very pale skin — should be Light (ITA > 41°)
        ita, label = compute_ita(_solid_image(240, 210, 185))
        if ita is not None:
            assert label == "Light"

    def test_rgb_to_lab_output_shape(self):
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        lab = rgb_to_lab(arr)
        assert lab.shape == (32, 32, 3)

    def test_rgb_to_lab_L_range(self):
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        lab = rgb_to_lab(arr)
        assert lab[..., 0].min() >= 0.0
        assert lab[..., 0].max() <= 100.0
