# DermaRead: Psoriasis Detection (Darker Skin Focus)

## Vision
Reduce misdiagnosis on darker skin tones by providing a transparent image-screening tool (with heatmaps) for psoriasis vs. non-psoriasis.

## Aims (Sprint 1–2)
- Build a dataset pipeline with explicit consent/licensing & metadata (skin tone tags).
- Train a baseline classifier (ResNet18) and log metrics (accuracy, F1).
- Ship a web demo (Gradio) with upload + prediction + disclaimer.
- Add XAI (Grad-CAM) for clinician-style explanations.

## How to run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python DermaRead/app.py

