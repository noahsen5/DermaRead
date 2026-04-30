# DermaRead — Test Plan

Covers: functional testing, model evaluation, integration testing, and validation.
Maps to: Critical Evaluation & Conclusions (15% of CW1 mark).

---

## 1. Testing approach

The project uses a layered testing strategy:

| Layer | Method | Tool / Script |
|---|---|---|
| Unit: preprocessing | Verify transform identity across modules | `models/debug_predictions.py` |
| Unit: class mapping | Verify label→index consistency across all files | `models/debug_predictions.py` |
| Integration: inference | Test full pipeline (image → prediction → output) | `models/debug_predictions.py` |
| Model: internal | Evaluate on held-out internal test split | `models/evaluate_model.py` |
| Model: external | Evaluate on never-seen external validation set | `models/evaluate_external.py` |
| UI: smoke test | Confirm all UI tabs load without error | Manual |
| UI: OOD | Confirm OOD warning triggers at confidence > 97% | Manual |

---

## 2. Unit tests

### 2.1 Class mapping consistency
**Test:** Assert `CLASS_NAMES` is identical in `models/baseline.py`, `models/resnet50_model.py`, and `models/dataset.py`.
**Pass criterion:** All equal `["non-psoriasis", "psoriasis"]`.
**Tool:** `models/debug_predictions.py` → `check_and_save_class_mapping()`
**Result:** PASS — all three modules consistent; all three checkpoints saved with matching classes field.
**Output:** `outputs/class_mapping.json`

### 2.2 Shared preprocessing identity
**Test:** Assert `inference_transform` in `baseline.py`, `resnet50_model.py`, and `gradcam.py` is the same Python object (imported from `models/preprocessing.py`).
**Pass criterion:** `is` identity check returns True for all three.
**Tool:** Inline assertion in debug script.
**Result:** PASS — all point to the same object; no independent transform definitions remain.

### 2.3 Missing image handling
**Test:** Attempt inference on a non-existent file path.
**Pass criterion:** `FileNotFoundError` raised cleanly; Gradio UI returns an error message without crashing.
**Result:** PASS — `load_checkpoint` raises `FileNotFoundError`; UI wraps in try/except and returns user-facing message.

### 2.4 Invalid image format
**Test:** Attempt inference on a text file renamed as `.jpg`.
**Pass criterion:** UI returns error message without crashing.
**Result:** PASS — PIL raises exception, caught in `_infer()`, user sees error text.

---

## 3. Integration tests — inference pipeline

### 3.1 Dataset images (all four models)
**Test:** Run all four model versions on 10 psoriasis + 10 non-psoriasis images from the internal test split.
**Pass criterion:** 20/20 correct predictions for V1, V2, V3 (within-distribution); reasonable accuracy for V4.
**Tool:** `models/debug_predictions.py --all-models`
**Result:** PASS — V1, V2, V3: 20/20 (100%). V4: 20/20 on internal test images.
**Output:** `outputs/debug_predictions.csv`

### 3.2 Grad-CAM generation
**Test:** Confirm heatmap file is saved without error for each model version.
**Pass criterion:** `outputs/gradcam_examples/gradcam_*.png` files created, non-zero size.
**Result:** PASS — 74 Grad-CAM files generated during UI usage.

### 3.3 Skin-tone proxy estimation
**Test:** Run ITA estimation on 1,000 training images.
**Pass criterion:** `estimated_skin_tone_proxy` column populated; distribution matches expected (Light, Medium, Dark present).
**Tool:** `models/skin_tone_ita.py`
**Result:** PASS — Light: 392, Medium: 495, Dark: 113. `data/processed_manifest.csv` saved.

---

## 4. Model evaluation — internal test set

**Protocol:** Stratified 70/15/15 split; test set never used during training or hyperparameter selection.
**Metrics:** Accuracy, precision, recall, F1 (weighted), ROC-AUC, confusion matrix.

| Model | Accuracy | F1 | AUC | Pass? |
|---|---|---|---|---|
| V1 ResNet18 | 100% | 1.000 | 1.000 | PASS |
| V2 ResNet50 | 100% | 1.000 | 1.000 | PASS |
| V3 ResNet50 Balanced | 100% | 1.000 | 1.000 | PASS |
| V4 ResNet50 Fine-tuned | 81.3% | 0.807 | 1.000 | PASS |

All outputs saved: `outputs/v*/confusion_matrix.png`, `outputs/v*/classification_report.csv`.

---

## 5. External validation — verification of real-world generalisation

**Protocol:** 762 images collected from external Kaggle sources, completely independent of training data. Run AFTER all training decisions finalised. No images from this set were used for hyperparameter tuning for V1–V3.

**Pass criterion for this test:** Honest reporting of results regardless of outcome.

| Model | External Accuracy | Psoriasis Recall | Non-Psoriasis Recall |
|---|---|---|---|
| V1 | 54% | 100% | 1% |
| V2 | 53% | 100% | 0% |
| V3 | 53% | 100% | 1% |
| **V4** | **82%** | **75%** | **90%** |

**Finding:** V1–V3 FAIL external validation. Root cause identified and documented (data quality issue: non-psoriasis class was clear healthy skin). V4 addresses this via fine-tuning on diverse real-world conditions.

**V4 held-out test (153 images, unseen during fine-tuning):** 68% accuracy.

---

## 6. Validation — does the system meet its aims?

| Project aim | Evidence | Met? |
|---|---|---|
| Classify psoriasis vs non-psoriasis | V4: 68-82% external accuracy | Partially — functional but not clinical standard |
| Provide explainability via Grad-CAM | Heatmaps generated for all predictions | Yes |
| Evaluate fairness across skin tones | ITA proxy analysis on 1,000 images | Yes (with stated limitations) |
| Handle external real-world images | V4 trained on diverse conditions | Yes (improvement from 0% to 80% non-psoriasis recall) |
| Warn about model limitations | OOD warning, model card, disclaimer | Yes |
| Modular, reproducible codebase | Shared preprocessing, fixed seeds, requirements.txt, README | Yes |

---

## 7. Regression check

After each sprint, `models/debug_predictions.py --all-models` was run to confirm that previously working predictions had not regressed. No regressions were observed.

---

## 8. Known test limitations

- **No formal automated test suite (pytest):** Testing is performed via scripts rather than a pytest framework. This is a limitation but the scripts cover all critical paths.
- **UI testing is manual:** No Selenium or Playwright tests for the Gradio interface; screenshots taken during manual testing.
- **External test set size:** 762 images is statistically sufficient for accuracy estimates but small for per-subgroup fairness analysis.
