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
    from models.resnet50_model import load_v2_model, load_v3_model, load_v4_model
    _try_register("V2 — ResNet50 Transfer", lambda: load_v2_model(device="cpu"))
    _try_register("V3 — ResNet50 Balanced", lambda: load_v3_model(device="cpu"))
    _try_register("V4 — ResNet50 Real-World (recommended)", lambda: load_v4_model(device="cpu"))
except ImportError:
    pass

if not _MODEL_REGISTRY:
    raise RuntimeError(
        "No trained models found. Train at least V1 first:\n"
        "  python models/train_resnet18_baseline.py"
    )

# Default to V4 if available — it handles real-world images
_DEFAULT_MODEL = next(
    (k for k in _MODEL_REGISTRY if "V4" in k),
    list(_MODEL_REGISTRY.keys())[0],
)

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

    # High-confidence predictions on external images are often wrong due to distribution shift.
    # V1-V3 were trained on a single Kaggle-sourced dataset and will confidently misclassify
    # real-world images. V4 is better but still has known limitations with non-Kaggle sources.
    ood_warning = ""
    if confidence > 0.97:
        ood_warning = (
            "\n\n> **Distribution shift warning:** This model was trained on a specific curated dataset. "
            "High confidence on images from smartphones, clinics, or the internet does **not** "
            "mean the prediction is correct — V1–V3 in particular will often predict with 100% confidence "
            "on images that look nothing like their training data. "
            "Use **V4 (Real-World)** for external images and treat all predictions as indicative only. "
            "See the Model Card for details."
        )

    pred_md = f"### {label}\n**Confidence:** {confidence:.1%}{ood_warning}"
    prob_md = "\n".join(f"- **{cls}:** {probs[cls]:.1%}" for cls in CLASS_NAMES)

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    # Always target the psoriasis class (index 1) so the heatmap always shows
    # "where the model looked for psoriasis features", regardless of the prediction.
    heatmap_path = None
    try:
        psoriasis_idx = CLASS_NAMES.index("psoriasis")
        out = _GRADCAM_DIR / f"gradcam_{uuid4().hex}.png"
        save_gradcam(model, img, out, target_class=psoriasis_idx)
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
    "This map always shows which regions the model examined when looking for **psoriasis features**, "
    "regardless of whether the final prediction is psoriasis or not.  \n"
    "Warm (red/yellow) areas = strongest psoriasis-related activation. Cool (blue) areas = least relevant.  \n"
    "If the model predicted *non-psoriasis*, it means it examined these regions and found "
    "insufficient evidence to classify as psoriasis.  \n\n"
    "_Grad-CAM indicates model attention, not clinical diagnosis._"
)

_SUBTYPE_NOTE = (
    "**Subtype classification is not available in this version.**  \n\n"
    "The external validation dataset includes subtype annotations for 53 psoriasis images "
    "across five subtypes: plaque (18), guttate (11), pustular (10), erythrodermic (7), inverse (7).  \n\n"
    "A reliable deep learning classifier requires a minimum of ~100–200 labelled examples per class. "
    "With approximately 10 images per subtype, training a subtype model would result in high variance "
    "and unreliable predictions — so it has been intentionally excluded from this version.  \n\n"
    "Expanding the annotated dataset to ≥100 images per subtype would enable this feature in a future version."
)

_MODEL_CARD = """
### Model Card

| Field | Details |
|---|---|
| **Architectures** | ResNet18 / ResNet50 (PyTorch / torchvision) |
| **Task** | Binary classification — psoriasis vs. non-psoriasis |
| **Input** | 224 × 224 RGB, ImageNet normalisation |
| **V1 — ResNet18 Baseline** | Trained on ~700 images (Kaggle-sourced) · 100% internal test · 54% external · fails on non-Kaggle images |
| **V2 — ResNet50 Transfer** | Two-phase ImageNet fine-tuning · 100% internal · 53% external · fails on non-Kaggle images |
| **V3 — ResNet50 Balanced** | Class-weighted loss · 100% internal · 53% external · no improvement over V2 externally |
| **V4 — ResNet50 Real-World** | Fine-tuned on 609 diverse images · 79% on 762-image external set · **recommended** |
| **Why V1–V3 fail externally** | Non-psoriasis training class = clear healthy skin only. Models learned: any visible skin condition → psoriasis. External non-psoriasis images (acne, eczema, etc.) are misclassified at ~99% rate. |
| **Why any model may fail on your image** | All models were trained/validated on a specific Kaggle-sourced dataset. Images from other clinical sources, smartphones, or the internet may look different enough to cause incorrect predictions — even with very high confidence. This is a known distribution shift problem in medical AI. |
| **Fairness** | ITA-based skin-tone proxy — groups: Light / Medium / Dark. Not equivalent to Fitzpatrick scale. |
| **Explainability** | Grad-CAM always targets the psoriasis class — shows where the model looked for psoriasis features |
| **Subtype** | Not trained — 53 labelled examples across 5 subtypes is insufficient (~10 per class) |
| **Known limitations** | V4 held-out test accuracy: 68% (153 images); no clinical validation; IPC-source clinical photos achieve only 20% with V4 |
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
            gr.Markdown(
                "_**V4 (Real-World)** is recommended for all use. "
                "V1–V3 are research baselines: they score 100% on the internal test set "
                "but classify nearly all real-world skin conditions as psoriasis due to "
                "training data limitations — they are included to demonstrate this finding, "
                "not for practical use._"
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

            with gr.Tab("Fairness Context"):
                gr.Markdown(
                    "**Why is skin-tone shown here?**  \n"
                    "Medical AI systems can perform differently across skin tones. "
                    "This project evaluated model fairness by estimating skin tone from each image "
                    "using the Individual Typology Angle (ITA) — a photometric proxy computed from "
                    "CIELAB colour values. Performance was compared across Light, Medium, and Dark "
                    "estimated-tone groups to identify any disparity.  \n\n"
                    "_This estimate is NOT used to make predictions and does NOT affect the output above. "
                    "It is provided as context for the fairness evaluation research._"
                )
                skin_tone_output = gr.Markdown(
                    value="_Upload an image to see the estimated skin-tone proxy._"
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
