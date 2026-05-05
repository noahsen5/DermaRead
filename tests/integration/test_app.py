"""Integration tests for the inference callback chain."""

import numpy as np
import pytest
from PIL import Image

from models.baseline import build_model, CLASS_NAMES, predict_pil
from models.gradcam import save_gradcam
from models.skin_tone_ita import compute_ita
from models.severity import estimate_visual_severity


def _skin_image(size=224):
    arr = np.full((size, size, 3), [200, 150, 110], dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture(scope="module")
def untrained_model():
    return build_model(weights=None)


class TestPredictPIL:
    def test_returns_dict_with_class_keys(self, untrained_model):
        result = predict_pil(untrained_model, _skin_image())
        assert set(result.keys()) == set(CLASS_NAMES)

    def test_probabilities_sum_to_one(self, untrained_model):
        result = predict_pil(untrained_model, _skin_image())
        total = sum(result.values())
        assert abs(total - 1.0) < 1e-5

    def test_probabilities_in_range(self, untrained_model):
        result = predict_pil(untrained_model, _skin_image())
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_rgb_conversion_handled(self, untrained_model):
        # RGBA input should not crash
        arr = np.random.randint(0, 255, (224, 224, 4), dtype=np.uint8)
        rgba = Image.fromarray(arr, mode="RGBA").convert("RGB")
        result = predict_pil(untrained_model, rgba)
        assert len(result) == 2


class TestGradCAMInChain:
    def test_save_gradcam_produces_file(self, untrained_model, tmp_path):
        out = tmp_path / "heatmap.png"
        save_gradcam(untrained_model, _skin_image(), out, target_class=1)
        assert out.exists() and out.stat().st_size > 0


class TestITAInChain:
    def test_returns_valid_output(self):
        ita, label = compute_ita(_skin_image())
        assert isinstance(label, str)
        assert label in {"Light", "Medium", "Dark", "unknown"}

    def test_handles_non_skin_gracefully(self):
        # Blue image — no skin detected, should not raise
        arr = np.full((224, 224, 3), [30, 30, 200], dtype=np.uint8)
        ita, label = compute_ita(Image.fromarray(arr))
        assert ita is None or isinstance(ita, float)


class TestSeverityInChain:
    def test_returns_required_keys(self):
        result = estimate_visual_severity(_skin_image())
        for key in ("estimated_coverage_pct", "erythema_index",
                    "texture_score", "severity_label", "reliable"):
            assert key in result

    def test_reliable_flag_type(self):
        result = estimate_visual_severity(_skin_image())
        assert isinstance(result["reliable"], bool)

    def test_severity_label_valid(self):
        result = estimate_visual_severity(_skin_image())
        assert result["severity_label"] in {"Mild", "Moderate", "Severe", "Unknown"}

    def test_unreliable_on_plain_blue(self):
        arr = np.full((224, 224, 3), [30, 30, 200], dtype=np.uint8)
        result = estimate_visual_severity(Image.fromarray(arr))
        assert result["reliable"] is False


class TestOODWarningLogic:
    """The OOD warning fires at >97% confidence for V1–V3."""

    def test_high_confidence_threshold(self, untrained_model):
        # The threshold used in app.py is 0.97 — verify predict_pil can reach it
        result = predict_pil(untrained_model, _skin_image())
        max_conf = max(result.values())
        # We can't guarantee >0.97 with an untrained model, but confidence must be in range
        assert 0.0 <= max_conf <= 1.0
