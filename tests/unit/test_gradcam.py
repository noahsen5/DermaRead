"""Unit tests for the Grad-CAM module."""

import numpy as np
import pytest
import torch
from PIL import Image

from models.gradcam import compute_gradcam, overlay_heatmap, save_gradcam
from models.baseline import build_model


@pytest.fixture(scope="module")
def v1_model():
    """Untrained ResNet18 — sufficient for testing Grad-CAM mechanics."""
    return build_model(weights=None)


def _random_image(size=224):
    arr = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestComputeGradCAM:
    def test_returns_pil_image(self, v1_model):
        heatmap = compute_gradcam(v1_model, _random_image())
        assert isinstance(heatmap, Image.Image)

    def test_heatmap_matches_input_size(self, v1_model):
        img = _random_image(224)
        heatmap = compute_gradcam(v1_model, img)
        assert heatmap.size == img.size

    def test_heatmap_non_trivial(self, v1_model):
        # Heatmap should not be entirely zero or flat
        heatmap = compute_gradcam(v1_model, _random_image())
        arr = np.array(heatmap)
        assert arr.std() > 0

    def test_explicit_target_class(self, v1_model):
        heatmap = compute_gradcam(v1_model, _random_image(), target_class=0)
        assert isinstance(heatmap, Image.Image)

    def test_different_classes_give_different_maps(self, v1_model):
        img = _random_image()
        h0 = np.array(compute_gradcam(v1_model, img, target_class=0))
        h1 = np.array(compute_gradcam(v1_model, img, target_class=1))
        # Not identical — different class gradients should produce different maps
        assert not np.array_equal(h0, h1)


class TestOverlayHeatmap:
    def test_output_is_rgb(self, v1_model):
        img = _random_image()
        heatmap = compute_gradcam(v1_model, img)
        overlay = overlay_heatmap(img, heatmap)
        assert overlay.mode == "RGB"

    def test_output_size_matches_input(self, v1_model):
        img = _random_image(256)
        heatmap = compute_gradcam(v1_model, img)
        overlay = overlay_heatmap(img, heatmap)
        assert overlay.size == img.size


class TestSaveGradCAM:
    def test_file_written(self, v1_model, tmp_path):
        out = tmp_path / "test_gradcam.png"
        save_gradcam(v1_model, _random_image(), out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_creates_parent_dirs(self, v1_model, tmp_path):
        out = tmp_path / "nested" / "deep" / "gradcam.png"
        save_gradcam(v1_model, _random_image(), out)
        assert out.exists()
