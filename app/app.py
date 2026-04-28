"""
DermaRead — Explainable psoriasis detection with skin-tone-aware fairness evaluation.

DISCLAIMER: Research prototype only. Not a medical device.
            Predictions must not be used for diagnosis.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr
from PIL import Image

from models.baseline import CLASS_NAMES, predict_pil
from models.gradcam import save_gradcam
from models.skin_tone_ita import compute_ita

# ── Model registry ────────────────────────────────────────────────────────────

_MODEL_REGISTRY: dict[str, object] = {}


def _try_register(label: str, loader_fn) -> None:
    try:
        _MODEL_REGISTRY[label] = loader_fn()
        print(f"[DermaRead] Loaded: {label}")
    except Exception as exc:
        print(f"[DermaRead] {label} not available: {exc}")


_try_register(
    "V1 — ResNet18 Baseline",
    lambda: __import__("models.baseline", fromlist=["load_trained_model"]).load_trained_model(device="cpu"),
)

try:
    from models.resnet50_model import load_v2_model, load_v3_model
    _try_register("V2 — ResNet50 Transfer", lambda: load_v2_model(device="cpu"))
    _try_register("V3 — ResNet50 Balanced", lambda: load_v3_model(device="cpu"))
except ImportError:
    pass

if not _MODEL_REGISTRY:
    raise RuntimeError(
        "No trained models found. Train at least V1 first:\n"
        "  python models/train_resnet18_baseline.py"
    )

_DEFAULT_MODEL = list(_MODEL_REGISTRY.keys())[0]

# ── Paths ─────────────────────────────────────────────────────────────────────

_GRADCAM_DIR = ROOT / "outputs/gradcam_examples"
_GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

# ── Inference ─────────────────────────────────────────────────────────────────

def _infer(img: Image.Image | None, model_choice: str):
    """Run prediction, Grad-CAM, and skin-tone estimation; return (pred, probs, heatmap, skin)."""
    if img is None:
        return "Upload an image to begin.", "", None, "_No image provided._"

    model = _MODEL_REGISTRY.get(model_choice)
    if model is None:
        return "Selected model is unavailable.", "", None, ""

    # ── Validate image ────────────────────────────────────────────────────────
    try:
        img = img.convert("RGB")
    except Exception as exc:
        return f"Invalid image format: {exc}", "", None, ""

    # ── Prediction ────────────────────────────────────────────────────────────
    try:
        probs = predict_pil(model, img)
    except Exception as exc:
        return f"Prediction failed: {exc}", "", None, ""

    label = max(probs, key=probs.get)
    confidence = probs[label]

    # Warn if image may be out-of-distribution (high confidence but unusual input).
    # The training set consists of a single curated source; external photos often trigger this.
    ood_warning = ""
    if confidence > 0.97:
        ood_warning = (
            "\n\n> **Out-of-distribution warning:** Confidence is very high "
            "({:.0%}). If this image is from a smartphone, clinic, or internet source "
            "it may differ substantially from the training data, making the prediction "
            "unreliable. See the Model Card for details.".format(confidence)
        )

    pred_md = f"### {label}\n**Confidence:** {confidence:.1%}{ood_warning}"
    prob_md = "\n".join(f"- **{cls}:** {probs[cls]:.1%}" for cls in CLASS_NAMES)

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    heatmap_path = None
    try:
        label_idx = CLASS_NAMES.index(label)
        out = _GRADCAM_DIR / f"gradcam_{uuid4().hex}.png"
        save_gradcam(model, img, out, target_class=label_idx)
        heatmap_path = str(out)
    except Exception as exc:
        print(f"[Grad-CAM] Failed: {exc}")

    # ── Skin-tone proxy ───────────────────────────────────────────────────────
    try:
        ita_val, skin_label = compute_ita(img)
        if ita_val is not None:
            skin_md = (
                f"**{skin_label}** (ITA ≈ {ita_val:.1f}°)\n\n"
                "_Estimated skin-tone proxy — not Fitzpatrick classification._  \n"
                "_ITA is affected by lesion colour, lighting, and image quality._"
            )
        else:
            skin_md = "_Could not estimate: insufficient skin-like pixels detected._"
    except Exception as exc:
        skin_md = f"_Skin-tone proxy estimation failed: {exc}_"

    return pred_md, prob_md, heatmap_path, skin_md


# ── Static text ───────────────────────────────────────────────────────────────

_DISCLAIMER = (
    "**⚠️ Research prototype only. Not a medical device. "
    "Predictions must not be used for diagnosis.**"
)

_HEATMAP_NOTE = (
    "**What does the heatmap show?**  \n"
    "Grad-CAM highlights the image regions that most influenced the prediction. "
    "Warm (red/yellow) areas had the strongest effect; cool (blue) areas had the least. "
    "This indicates model focus, not clinical significance."
)

_SUBTYPE_NOTE = (
    "**Subtype model not trained** because subtype labels "
    "(plaque, guttate, inverse, pustular) were not available in the dataset.  \n\n"
    "The code structure for subtype classification is present in the repository "
    "and can be enabled once labelled subtype data is available."
)

_MODEL_CARD = """
### Model Card

| Field | Details |
|---|---|
| **Architectures** | ResNet18 / ResNet50 (PyTorch / torchvision) |
| **Task** | Binary classification — psoriasis vs. non-psoriasis |
| **Input** | 224 × 224 RGB, ImageNet normalisation |
| **Training data** | Local augmented image dataset; split 70 / 15 / 15 (train / val / test) |
| **Known limitations** | Trained on one curated dataset; **fails on out-of-distribution external images**; limited demographic diversity; no clinical validation |
| **Fairness** | Evaluated by estimated skin-tone proxy (ITA) — not a validated Fitzpatrick-scale analysis |
| **Explainability** | Grad-CAM (Gradient-weighted Class Activation Mapping) |
| **Subtype** | Not trained — subtype labels unavailable in dataset |
| **AI/ML** | Standard supervised image classification; no generative components |
"""


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="DermaRead", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# DermaRead")
    gr.Markdown("### Explainable psoriasis detection with skin-tone-aware fairness evaluation")
    gr.Markdown(_DISCLAIMER)

    with gr.Row():
        # Left column: inputs
        with gr.Column(scale=1, min_width=280):
            img_input = gr.Image(type="pil", label="Upload skin image")
            gr.Markdown(
                "> **External image note:** This model was trained on a single "
                "curated dataset. Images from smartphones, clinics, or the internet "
                "may be **out-of-distribution** and will likely produce unreliable "
                "predictions, even with high confidence scores."
            )
            model_selector = gr.Dropdown(
                choices=list(_MODEL_REGISTRY.keys()),
                value=_DEFAULT_MODEL,
                label="Model version",
            )
            submit_btn = gr.Button("Analyse", variant="primary")

        # Right column: tabbed results
        with gr.Column(scale=2):
            with gr.Tab("Prediction"):
                pred_output = gr.Markdown(value="_Upload an image to begin._")
                prob_output = gr.Markdown()

            with gr.Tab("Grad-CAM"):
                heatmap_output = gr.Image(type="filepath", label="Grad-CAM overlay")
                gr.Markdown(_HEATMAP_NOTE)

            with gr.Tab("Skin-Tone Proxy"):
                skin_tone_output = gr.Markdown(
                    value="_Upload an image to estimate skin-tone proxy._"
                )

            with gr.Tab("Subtype"):
                gr.Markdown(_SUBTYPE_NOTE)

    with gr.Accordion("Model Card", open=False):
        gr.Markdown(_MODEL_CARD)

    _outputs = [pred_output, prob_output, heatmap_output, skin_tone_output]
    submit_btn.click(fn=_infer, inputs=[img_input, model_selector], outputs=_outputs)
    img_input.change(fn=_infer, inputs=[img_input, model_selector], outputs=_outputs)


# ── Launch ────────────────────────────────────────────────────────────────────

_PORT_ENV = os.environ.get("GRADIO_SERVER_PORT")
_SERVER = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
_SHARE = os.environ.get("GRADIO_SHARE", "false").lower() in {"1", "true", "yes"}

if __name__ == "__main__":
    port = int(_PORT_ENV) if _PORT_ENV else None
    demo.launch(server_name=_SERVER, server_port=port, share=_SHARE)
