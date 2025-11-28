import os
from pathlib import Path
from uuid import uuid4
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr
import pandas as pd
from PIL import Image

from models.baseline import CLASS_NAMES, load_trained_model, predict_pil
from models.gradcam import save_gradcam

model = load_trained_model(device="cpu")
GRADCAM_DIR = Path("docs/screenshots")
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
SERVER_PORT = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
SERVER_NAME = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")

DISCLAIMER = (
    "Research prototype — not a medical device. "
    "Predictions must not be used for diagnosis."
)


def infer(img: Image.Image):
    if img is None:
        empty = pd.DataFrame(columns=["label", "value"])
        return "Upload an image", empty, None

    probs = predict_pil(model, img)
    label = max(probs, key=probs.get)
    chart = [{"label": k, "value": v} for k, v in probs.items()]
    chart_df = pd.DataFrame(chart)

    heatmap_path = None
    label_idx = CLASS_NAMES.index(label)
    try:
        filename = GRADCAM_DIR / f"gradcam_{uuid4().hex}.png"
        save_gradcam(model, img, filename, target_class=label_idx)
        heatmap_path = str(filename)
    except Exception as exc:  # pragma: no cover
        print(f"Grad-CAM generation failed: {exc}")

    return f"Prediction: **{label}**", chart_df, heatmap_path


demo = gr.Interface(
    fn=infer,
    inputs=gr.Image(type="pil", label="Upload skin image"),
    outputs=[
        gr.Markdown(),
        gr.BarPlot(x="label", y="value", title="Class probabilities"),
        gr.Image(type="filepath", label="Grad-CAM heatmap"),
    ],
    title="DermaRead — Baseline",
    description=DISCLAIMER,
)


if __name__ == "__main__":
    demo.launch(server_name=SERVER_NAME, server_port=SERVER_PORT)
