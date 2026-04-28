import os
from pathlib import Path
from uuid import uuid4
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr
from PIL import Image

from models.baseline import CLASS_NAMES, load_trained_model, predict_pil
from models.gradcam import save_gradcam

model = load_trained_model(device="cpu")
GRADCAM_DIR = ROOT / "docs/screenshots"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
SERVER_PORT_ENV = os.environ.get("GRADIO_SERVER_PORT")
SERVER_NAME = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
SHARE = os.environ.get("GRADIO_SHARE", "false").lower() in {"1", "true", "yes"}

DISCLAIMER = (
    "**Research prototype — not a medical device. "
    "Predictions must not be used for diagnosis.**"
)


def infer(img: Image.Image):
    if img is None:
        return "Upload an image first.", None, None

    probs = predict_pil(model, img)
    label = max(probs, key=probs.get)
    confidence = probs[label]

    result_text = f"### Prediction: {label}\n**Confidence:** {confidence:.1%}"

    heatmap_path = None
    label_idx = CLASS_NAMES.index(label)
    try:
        filename = GRADCAM_DIR / f"gradcam_{uuid4().hex}.png"
        save_gradcam(model, img, filename, target_class=label_idx)
        heatmap_path = str(filename)
    except Exception as exc:
        print(f"Grad-CAM failed: {exc}")

    # Format probabilities as a readable string
    prob_text = "\n".join(
        f"- **{cls}:** {probs[cls]:.1%}" for cls in CLASS_NAMES
    )

    return result_text, prob_text, heatmap_path


with gr.Blocks(title="DermaRead") as demo:
    gr.Markdown("# DermaRead — Psoriasis Classifier")
    gr.Markdown(DISCLAIMER)

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="Upload skin image")
            submit_btn = gr.Button("Analyse", variant="primary")
        with gr.Column():
            pred_output = gr.Markdown(label="Prediction")
            prob_output = gr.Markdown(label="Probabilities")
            heatmap_output = gr.Image(type="filepath", label="Grad-CAM heatmap")

    submit_btn.click(
        fn=infer,
        inputs=img_input,
        outputs=[pred_output, prob_output, heatmap_output],
    )
    img_input.change(
        fn=infer,
        inputs=img_input,
        outputs=[pred_output, prob_output, heatmap_output],
    )


if __name__ == "__main__":
    port = int(SERVER_PORT_ENV) if SERVER_PORT_ENV else None
    demo.launch(server_name=SERVER_NAME, server_port=port, share=SHARE)
