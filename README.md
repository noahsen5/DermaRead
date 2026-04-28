# DermaRead

Reduce misdiagnosis on darker skin tones by providing a transparent image-screening tool (with heatmaps) for psoriasis vs. non-psoriasis.

**Disclaimer:** Research prototype. Not a medical device. Predictions must not be used for diagnosis.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Data preparation

```bash
# Create the train / val / test split manifest
python data/pipeline_from_manifest.py

# Optional: limit images per class (e.g. 500 each) for faster iteration
python data/pipeline_from_manifest.py --limit-per-label 500
```

---

## Training

### V1 — ResNet18 Baseline

```bash
python models/train_resnet18_baseline.py
# Options:
python models/train_resnet18_baseline.py --epochs 20 --lr 5e-5 --batch-size 32
```

Saves checkpoint to `models/checkpoints/resnet18_baseline.pt`.

### V2 — ResNet50 Transfer Learning

```bash
python models/train_resnet50.py
python models/train_resnet50.py --epochs 20 --lr 1e-4
```

Saves checkpoint to `models/checkpoints/resnet50_v2.pt`.

### V3 — ResNet50 Balanced (Weighted Loss)

```bash
python models/train_resnet50_balanced.py
python models/train_resnet50_balanced.py --epochs 20 --lr 1e-4
```

Saves checkpoint to `models/checkpoints/resnet50_v3_balanced.pt`.

---

## Evaluation

Each training script automatically evaluates the best checkpoint on the test split.
To re-run evaluation standalone:

```bash
python models/evaluate_model.py --model v1
python models/evaluate_model.py --model v2
python models/evaluate_model.py --model v3
```

Results are written to:
- `outputs/results_summary.csv` — comparison table across all versions
- `outputs/v1_resnet18_baseline/confusion_matrix.png`
- `outputs/v1_resnet18_baseline/classification_report.csv`
- `outputs/v1_resnet18_baseline/training_log.csv`
- *(same structure for v2 and v3)*

---

## Skin-tone proxy annotation

```bash
python models/skin_tone_ita.py
```

Reads `data/manifest.split.csv`, computes ITA for every image,
and writes `data/processed_manifest.csv` with columns `ita_value` and `estimated_skin_tone_proxy`.

---

## Fairness evaluation

```bash
python models/fairness_eval.py
```

Requires `data/processed_manifest.csv` (run skin_tone_ita.py first).
Writes:
- `outputs/fairness_analysis/fairness_by_skin_tone.csv`
- `outputs/fairness_analysis/fairness_summary.md`

---

## Gradio UI

```bash
python app/app.py
```

Environment variables (all optional):

| Variable | Default | Description |
|---|---|---|
| `GRADIO_SERVER_NAME` | `127.0.0.1` | Bind address |
| `GRADIO_SERVER_PORT` | auto | Port number |
| `GRADIO_SHARE` | `false` | Create public share link |

---

## Output structure

```
outputs/
  results_summary.csv
  v1_resnet18_baseline/
    training_log.csv
    confusion_matrix.png
    classification_report.csv
  v2_resnet50/
    training_log.csv
    confusion_matrix.png
    classification_report.csv
  v3_resnet50_balanced/
    training_log.csv
    confusion_matrix.png
    classification_report.csv
  gradcam_examples/
  fairness_analysis/
    fairness_by_skin_tone.csv
    fairness_summary.md
  known_limitations.md
  dissertation_ready_summary.md

models/
  checkpoints/
    resnet18_baseline.pt
    resnet50_v2.pt
    resnet50_v3_balanced.pt
  baseline.py               # ResNet18 model definition (V1)
  resnet50_model.py         # ResNet50 model definition (V2, V3)
  train_baseline.py         # Original training script (preserved)
  train_resnet18_baseline.py
  train_resnet50.py
  train_resnet50_balanced.py
  evaluate_model.py
  skin_tone_ita.py
  fairness_eval.py
  gradcam.py
  dataset.py

data/
  manifest.csv
  manifest.split.csv
  processed_manifest.csv    # Created by skin_tone_ita.py
```

---

## Reproducibility

Random seed is fixed at `42` in all training scripts.
Results may vary slightly across platforms due to floating-point differences.
