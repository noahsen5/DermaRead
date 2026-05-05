# DermaRead

Binary psoriasis classifier with Grad-CAM explainability and skin-tone-aware fairness evaluation. Built as a final-year dissertation project at the University of Plymouth.

> **Disclaimer:** Research prototype only. Not a medical device. Predictions must not be used for diagnosis.

---

## What it does

Upload a skin image and get:
- A psoriasis / non-psoriasis prediction with confidence score
- A Grad-CAM heatmap showing which regions drove the prediction
- An ITA-based estimated skin-tone proxy (Light / Medium / Dark)
- A visual severity approximation (not clinical PASI)
- Body-part clinical context for the selected location

Four model versions are available. V4 is recommended for real-world images.

---

## Models

| Version | Architecture | Training | Internal accuracy | External accuracy |
|---|---|---|---|---|
| V1 | ResNet18 | ImageNet transfer | 100% | 54% |
| V2 | ResNet50 | Two-phase fine-tune | 99.93% | 53% |
| V3 | ResNet50 | V2 + weighted loss | 100% | 53% |
| V4 | ResNet50 | V3 fine-tuned on diverse external lesions | 81% | **82%** |

V1–V3 reach near-100% on the internal test split but classify almost every real-world lesion as psoriasis because the original non-psoriasis class was healthy skin, not realistic alternatives. V4 was fine-tuned on 609 diverse external lesion images (acne, eczema, melanoma, scabies, etc.) to fix this.

**Fitzpatrick17k external validation (16,550 images, 114 conditions, real skin-type labels):**
V3 AUC = 0.630, sensitivity = 0.711. Sensitivity gap across Fitzpatrick Types I–VI = 0.291 — unchanged after retraining on the full dataset, confirming demographic bias cannot be fixed by adding more data from the same homogeneous source.

---

## Setup

```
pip install -r requirements.txt
```

Requires Python 3.10+. CUDA or Apple MPS recommended for training; CPU works for inference.

---

## Run the app

```
python app/app.py
```

Opens at `http://127.0.0.1:7860`. V4 loads by default.

| Environment variable | Default | Description |
|---|---|---|
| `GRADIO_SERVER_PORT` | auto | Port |
| `GRADIO_SERVER_NAME` | `127.0.0.1` | Bind address |
| `GRADIO_SHARE` | `false` | Public Gradio share link |

---

## Reproduce from scratch

Run these in order:

```bash
# 1. Build train/val/test split from manifest
python data/pipeline_from_manifest.py

# 2. Train models
python models/train_resnet18_baseline.py
python models/train_resnet50.py
python models/train_resnet50_balanced.py

# 3. Fine-tune V4 on external data
python models/finetune_external.py

# 4. Skin-tone proxy annotation
python models/skin_tone_ita.py

# 5. Internal fairness evaluation
python models/fairness_eval.py

# 6. External validation (762-image set)
python models/evaluate_external.py

# 7. Fitzpatrick17k external validation
python models/evaluate_fitzpatrick.py
```

Each training script saves its checkpoint, training log, confusion matrix and classification report automatically.

---

## Generate dissertation figures

All figures are saved to `outputs/figures/`.

```bash
python models/plot_training_curves.py      # Fig 5.2 — training/val loss curves V1–V4
python models/plot_confusion_matrices.py   # Fig 7.1 + 7.2 — internal and external confusion matrices
python models/plot_roc_fitzpatrick.py      # Fig 7.3 — ROC curves V1–V3 on Fitzpatrick17k
python models/plot_fairness_bar.py         # Fig 7.4 — sensitivity by Fitzpatrick type
python models/plot_threshold_sweep.py      # Fig 7.5 — threshold sweep with bootstrap CIs
python models/plot_calibration.py          # Fig 7.6 — V4 calibration plot
```

---

## Datasets

| Dataset | Images | Purpose |
|---|---|---|
| Primary training pool (`data/raw/`) | 10,007 | Training — 7,005 train / 1,501 val / 1,501 test |
| External validation (`data/external_validation/`) | 762 | Independent test — 405 psoriasis, 357 diverse conditions |
| Fitzpatrick17k (`data/fitzpatrick17k/`) | 16,577 | External fairness audit — real Fitzpatrick scale I–VI labels |

The pipeline uses a single CSV manifest (`data/manifest.csv`) as the source of truth. The stratified split groups by `patient_id` to prevent data leakage across splits.

---

## Output structure

```
outputs/
  results_summary.csv                    cross-model metrics table
  dissertation_ready_summary.md          all results + dissertation framing
  known_limitations.md                   limitations log
  v1_resnet18_baseline/
  v2_resnet50/
  v3_resnet50_balanced/
  v4_external_finetuned/
    training_log.csv
    confusion_matrix.png
    classification_report.csv
  external_validation/
    external_results.csv
    external_summary.md
  fairness_analysis/
    fairness_by_skin_tone.csv
    fairness_summary.md
  fitzpatrick_validation/
    fitzpatrick_predictions.csv
    fitzpatrick_metrics.csv
    fitzpatrick_fairness_by_scale.csv
    fitzpatrick_summary.md
  figures/
    fig_training_curves.png
    fig_confusion_internal.png
    fig_confusion_external.png
    fig_roc_fitzpatrick.png
    fig_fairness_bar.png
    fig_threshold_sweep.png
    fig_calibration_v4.png
  gradcam_examples/

models/checkpoints/
  resnet18_baseline.pt
  resnet50_v2.pt
  resnet50_v3_balanced.pt
  resnet50_v4_external.pt
```

---

## Reproducibility

Random seed is fixed at 42 in all training and split scripts. Results are within ±0.5 percentage points across CUDA and MPS backends due to floating-point differences.
