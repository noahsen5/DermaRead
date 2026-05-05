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
from models.severity import BODY_PARTS, estimate_visual_severity, get_clinical_note
from models.skin_tone_ita import compute_ita

_MODEL_REGISTRY: dict[str, object] = {}


def _try_register(label: str, loader_fn) -> None:
    try:
        _MODEL_REGISTRY[label] = loader_fn()
        print(f"[DermaRead] Loaded: {label}")
    except Exception as exc:
        print(f"[DermaRead] {label} unavailable: {exc}")


_try_register(
    "V1 — ResNet18 Baseline",
    lambda: __import__("models.baseline", fromlist=["load_trained_model"]).load_trained_model("cpu"),
)

try:
    from models.resnet50_model import load_v2_model, load_v3_model, load_v4_model
    _try_register("V2 — ResNet50 Transfer", lambda: load_v2_model("cpu"))
    _try_register("V3 — ResNet50 Balanced", lambda: load_v3_model("cpu"))
    _try_register("V4 — ResNet50 Real-World (recommended)", lambda: load_v4_model("cpu"))
except ImportError:
    pass

if not _MODEL_REGISTRY:
    raise RuntimeError(
        "No trained models found. Run: python models/train_resnet18_baseline.py"
    )

_DEFAULT_MODEL = next(
    (k for k in _MODEL_REGISTRY if "V4" in k),
    list(_MODEL_REGISTRY.keys())[0],
)

_GRADCAM_DIR = ROOT / "outputs/gradcam_examples"
_GRADCAM_DIR.mkdir(parents=True, exist_ok=True)


def _infer(img: Image.Image | None, model_choice: str, body_part: str):
    if img is None:
        return (
            "_Upload an image to begin._", "", None,
            "_No image provided._", "_No image provided._", "",
        )

    model = _MODEL_REGISTRY.get(model_choice)
    if model is None:
        return "Selected model is unavailable.", "", None, "", "", ""

    try:
        img = img.convert("RGB")
    except Exception as exc:
        return f"Invalid image format: {exc}", "", None, "", "", ""

    try:
        probs = predict_pil(model, img)
    except Exception as exc:
        return f"Prediction failed: {exc}", "", None, "", "", ""

    label      = max(probs, key=probs.get)
    confidence = probs[label]

    ood_note = ""
    if confidence > 0.97 and "V4" not in model_choice:
        ood_note = (
            "\n\n> **Note:** Very high confidence from a model not fine-tuned on "
            "real-world images (V1–V3). If this is an external photo, consider "
            "switching to **V4** for more reliable results."
        )

    pred_md = f"### {label}\n**Confidence:** {confidence:.1%}{ood_note}"
    prob_md  = "\n".join(f"- **{cls}:** {probs[cls]:.1%}" for cls in CLASS_NAMES)

    heatmap_path = None
    try:
        idx = CLASS_NAMES.index(label)
        out = _GRADCAM_DIR / f"gradcam_{uuid4().hex}.png"
        save_gradcam(model, img, out, target_class=idx)
        heatmap_path = str(out)
    except Exception as exc:
        print(f"[Grad-CAM] {exc}")

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
        skin_md = f"_Estimation failed: {exc}_"

    try:
        sv = estimate_visual_severity(img)
        if sv["reliable"]:
            cov = sv["estimated_coverage_pct"]
            ery = sv["erythema_index"]
            tex = sv["texture_score"]
            lbl = sv["severity_label"]

            def _bar(v: float, n: int = 10) -> str:
                filled = round(v * n)
                return "█" * filled + "░" * (n - filled)

            severity_md = (
                f"### Visual Severity Indicators\n"
                f"*Estimated from image colour and texture — not a clinical PASI score*\n\n"
                f"| Indicator | Estimate | Scale |\n"
                f"|---|---|---|\n"
                f"| Estimated coverage | {cov:.1f}% of visible skin | — |\n"
                f"| Erythema (redness) | {ery:.2f} | {_bar(ery)} |\n"
                f"| Texture roughness | {tex:.2f} | {_bar(tex)} |\n"
                f"\n**Composite visual label: {lbl}**\n\n"
                f"> This is a visual approximation based on colour and texture analysis "
                f"only. Clinical PASI scoring requires physical examination across four "
                f"body regions by a dermatologist."
            )
        else:
            severity_md = "_Could not estimate severity: insufficient skin pixels detected._"
    except Exception as exc:
        severity_md = f"_Severity estimation failed: {exc}_"

    clinical_note = get_clinical_note(body_part)
    if clinical_note and body_part != "Not specified":
        context_md = f"**{body_part}**\n\n{clinical_note}"
    else:
        context_md = "_Select a body part above to see clinical context._"

    return pred_md, prob_md, heatmap_path, skin_md, severity_md, context_md


_DISCLAIMER = (
    "**⚠️ Research prototype only. Not a medical device. "
    "Predictions must not be used for diagnosis.**"
)

_OOD_NOTE = (
    "> **About uploaded images:** This system was trained on a curated image dataset. "
    "Photos from smartphones, clinics, or the internet differ from training data and "
    "may produce unreliable predictions even with high confidence. "
    "**V4 is recommended for real-world images** — it was fine-tuned on diverse clinical conditions."
)

_HEATMAP_NOTE = (
    "**What does the heatmap show?**  \n"
    "Grad-CAM highlights which image regions most influenced the prediction. "
    "Warm (red/orange/yellow) areas had the strongest effect. "
    "This shows model attention, not clinically validated diagnostic regions."
)

_SUBTYPE_NOTE = (
    "**Subtype classification not trained.**  \n"
    "The dataset contains 54 psoriasis images with subtype labels "
    "(plaque, guttate, pustular, erythrodermic, inverse), insufficient for reliable training. "
    "Subtype classification is implemented in the codebase and can be enabled "
    "once sufficient labelled data per subtype is available.  \n\n"
    "Known psoriasis subtypes and their clinical significance:\n"
    "- **Plaque** : most common (~80–90%), well-demarcated erythematous plaques with silvery scale\n"
    "- **Guttate** : small teardrop lesions, often triggered by streptococcal infection\n"
    "- **Pustular** : sterile pustules; palmoplantar variant is particularly disabling\n"
    "- **Erythrodermic** : widespread erythema affecting >90% BSA; medical emergency\n"
    "- **Inverse** : smooth, shiny lesions in skin folds; no scale due to moisture"
)

_MODEL_CARD = """
### Model Card

| Field | Details |
|---|---|
| **Architecture** | ResNet18 / ResNet50 (PyTorch / torchvision) |
| **Task** | Binary classification — psoriasis vs. non-psoriasis |
| **Input** | 224 × 224 RGB, ImageNet normalisation |
| **Training set** | 7,005 images (4,136 psoriasis + 2,869 non-psoriasis) — full dataset, 70/15/15 split |
| **External validation** | 762 images (405 psoriasis, 357 diverse non-psoriasis conditions) + 16,550 Fitzpatrick17k images |
| **V1 ResNet18 Baseline** | 100% internal accuracy · 54% external · not suitable for real-world use |
| **V2 ResNet50 Transfer** | 99.93% internal · 53% external · two-phase ImageNet fine-tuning |
| **V3 ResNet50 Balanced** | 100% internal · 53% external · inverse-frequency class weighting |
| **V4 ResNet50 Real-World** | 81% internal · 82% full-external · **68% on held-out external test** · recommended |
| **Key finding** | V1–V3 remain near-100% even when trained on 7,005 images, confirming the problem is dataset homogeneity (not training set size). All models classify real-world conditions as psoriasis (~0% non-psoriasis recall). V4 fixes this. |
| **Fitzpatrick17k validation** | V3: AUC 0.630 · sensitivity 0.711 across 16,550 images (114 conditions). Sensitivity gap across skin types I–VI = 0.291. |
| **Fairness** | ITA proxy (Light/Medium/Dark) on internal data; real Fitzpatrick scale (Types I–VI) on Fitzpatrick17k external set |
| **Explainability** | Grad-CAM heatmaps on final convolutional layer |
| **Severity** | Visual approximation only — not clinical PASI |
| **Subtype** | Not trained — insufficient per-subtype samples |
| **Limitations** | No clinical validation; single training source; ITA proxy imperfect; V4 held-out accuracy 68% |
"""


with gr.Blocks(title="DermaRead") as demo:
    gr.Markdown("# DermaRead")
    gr.Markdown("### Explainable psoriasis detection with skin-tone-aware fairness evaluation")
    gr.Markdown(_DISCLAIMER)

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            img_input = gr.Image(type="pil", label="Upload skin image")
            gr.Markdown(_OOD_NOTE)

            body_part = gr.Dropdown(
                choices=BODY_PARTS,
                value="Not specified",
                label="Body location (optional — adds clinical context)",
            )
            model_selector = gr.Dropdown(
                choices=list(_MODEL_REGISTRY.keys()),
                value=_DEFAULT_MODEL,
                label="Model version",
            )
            submit_btn = gr.Button("Analyse", variant="primary", size="lg")

        with gr.Column(scale=2):
            with gr.Tab("Prediction"):
                pred_output = gr.Markdown(value="_Upload an image to begin._")
                prob_output = gr.Markdown()

            with gr.Tab("Grad-CAM"):
                heatmap_output = gr.Image(type="filepath", label="Grad-CAM overlay")
                gr.Markdown(_HEATMAP_NOTE)

            with gr.Tab("Skin-Tone Proxy"):
                skin_tone_output = gr.Markdown(
                    value="_Upload an image to estimate._"
                )

            with gr.Tab("Visual Severity"):
                severity_output = gr.Markdown(
                    value="_Upload an image to estimate._"
                )

            with gr.Tab("Body Part Context"):
                context_output = gr.Markdown(
                    value="_Select a body part on the left to see clinical notes._"
                )

            with gr.Tab("Subtype"):
                gr.Markdown(_SUBTYPE_NOTE)

    with gr.Accordion("Model Card", open=False):
        gr.Markdown(_MODEL_CARD)

    _outputs = [
        pred_output, prob_output, heatmap_output,
        skin_tone_output, severity_output, context_output,
    ]
    submit_btn.click(fn=_infer, inputs=[img_input, model_selector, body_part], outputs=_outputs)
    img_input.change(fn=_infer, inputs=[img_input, model_selector, body_part], outputs=_outputs)
    body_part.change(fn=_infer, inputs=[img_input, model_selector, body_part], outputs=_outputs)


_PORT_ENV = os.environ.get("GRADIO_SERVER_PORT")
_SERVER   = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
_SHARE    = os.environ.get("GRADIO_SHARE", "false").lower() in {"1", "true", "yes"}

if __name__ == "__main__":
    port = int(_PORT_ENV) if _PORT_ENV else None
    demo.launch(server_name=_SERVER, server_port=port, share=_SHARE,
                theme=gr.themes.Soft())
