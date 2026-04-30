# DermaRead — Agile Sprint Plan

Project: COMP3000 Final Year Project
Student: Noah Sengupta
Method: Scrum-inspired sprints (2-week cadence)
Start: November 2025

---

## Product Backlog (initial)

| ID | User Story | Priority | Story Points |
|---|---|---|---|
| US-01 | As a user I want to upload a skin image and receive a psoriasis prediction | Must | 8 |
| US-02 | As a user I want to see which part of the image the model focused on | Must | 5 |
| US-03 | As a researcher I want models evaluated on multiple metrics (F1, AUC, confusion matrix) | Must | 5 |
| US-04 | As a researcher I want to compare multiple model architectures | Must | 8 |
| US-05 | As a reviewer I want fairness evaluated across estimated skin-tone groups | Should | 5 |
| US-06 | As a user I want contextual clinical information based on body location | Should | 3 |
| US-07 | As a researcher I want external validation on real-world diverse images | Must | 8 |
| US-08 | As a user I want a visual indication of lesion severity | Could | 5 |
| US-09 | As a researcher I want the pipeline reproducible with fixed seeds and documented requirements | Must | 3 |
| US-10 | As a user I want the system to warn me if my image may be out-of-distribution | Should | 2 |

---

## Sprint 0 — Project setup and research
**Dates:** November 2025 (2 weeks)
**Goal:** Understand the domain, establish the tech stack, scaffold the repository.

### Tasks completed
- Researched psoriasis: clinical presentation, subtypes, PASI scoring, diagnostic criteria
- Researched related work: CNN-based skin lesion classification, Grad-CAM explainability, fairness in medical AI, ITA skin-tone proxy
- Set up PyTorch, torchvision, Gradio environment
- Scaffolded repository structure: `/models`, `/data`, `/app`, `/outputs`
- Created initial `data/manifest.csv` schema with id, label, relative_path, skin_tone, subtype columns
- Defined backlog and initial sprint plan

### Review
Goal met. Domain research informed all subsequent design decisions — in particular, understanding PASI revealed early that full automated scoring was out of scope, leading to the "visual severity indicators" design instead.

---

## Sprint 1 — Baseline pipeline
**Dates:** Early December 2025 (2 weeks)
**Goal:** Working ResNet18 baseline with training, inference, and basic UI.
**Backlog items:** US-01 (partial), US-09

### Tasks completed
- Implemented `data/pipeline_from_manifest.py` — stratified 70/15/15 train/val/test split
- Implemented `models/baseline.py` — ResNet18 with ImageNet pretrained weights
- Implemented `models/dataset.py` — `ManifestDataset` class
- Implemented `models/train_baseline.py` — training loop with best-checkpoint saving
- Implemented `models/gradcam.py` — Grad-CAM hook on `layer4[-1].conv2`
- Built initial Gradio UI with image upload, prediction, Grad-CAM tab
- Added `requirements.txt`

### Review
Baseline trained. Early UI showed Grad-CAM working correctly on dataset images. First MVP delivered — system shows "life" as a functional prototype. Identified that training on full 10,000-image dataset caused 100% internal accuracy, suggesting overfitting; noted for investigation in later sprint.

---

## Sprint 2 — ResNet50 and evaluation pipeline
**Dates:** Early January 2026 (2 weeks)
**Goal:** Add ResNet50 V2 and V3, create full evaluation artefacts.
**Backlog items:** US-03, US-04

### Tasks completed
- Implemented `models/resnet50_model.py`
- Implemented `models/train_resnet50.py` — two-phase transfer learning (head warmup → full fine-tune)
- Implemented `models/train_resnet50_balanced.py` — class-weighted CrossEntropyLoss (V3)
- Implemented `models/evaluate_model.py` — confusion matrix PNG, classification report CSV, results_summary.csv
- Created output directory structure: `outputs/v1_resnet18_baseline/`, `outputs/v2_resnet50/`, `outputs/v3_resnet50_balanced/`
- Trained all three models; all achieved 100% internal test accuracy

### Review
Internal accuracy of 100% flagged as suspicious. Investigated dataset composition — confirmed no structural data leakage but identified that training images were from a single curated source, likely causing distribution-specific memorisation rather than generalisation. Decision: build external validation set to test real-world performance. This is an example of dynamic response to a setback.

---

## Sprint 3 — ITA proxy, fairness evaluation, UI redesign
**Dates:** Late January 2026 (2 weeks)
**Goal:** Fairness analysis infrastructure, UI redesign, shared preprocessing module.
**Backlog items:** US-05, US-09

### Tasks completed
- Implemented `models/preprocessing.py` — single shared source for all transforms
- Implemented `models/skin_tone_ita.py` — RGB→CIELAB, skin-pixel detection, ITA→Light/Medium/Dark
- Implemented `models/fairness_eval.py` — per-skin-tone-proxy evaluation, fairness_by_skin_tone.csv
- Updated `models/gradcam.py` — correct layer selection for Bottleneck (ResNet50) vs BasicBlock (ResNet18)
- Redesigned `app/app.py` — tabbed UI, model selector, OOD warning, model card
- Implemented `models/debug_predictions.py` — class mapping check, 20-image test
- Saved `outputs/class_mapping.json`

### Review
Fairness analysis on internal test showed 100% across all skin-tone groups — expected given 100% overall accuracy. This confirmed that internal metrics were uninformative about fairness. Decision confirmed: external validation required.

---

## Sprint 4 — External dataset and real-world validation
**Dates:** February–March 2026 (2 weeks)
**Goal:** Build, validate, and evaluate against a real-world diverse external dataset.
**Backlog items:** US-07, US-10

### Tasks completed
- Collected 762 images from Kaggle: 405 psoriasis (with subtype labels), 357 non-psoriasis (acne, eczema, melanoma, scabies, cellulitis, nail fungus, warts, cellulitis)
- Created `data/external_validation/external_manifest.csv` with source, image_type, skin_tone_notes, subtype columns
- Implemented `models/evaluate_external.py` — evaluation against external set
- **Key finding:** V1–V3 achieve ~0% non-psoriasis recall on external set despite 100% internal accuracy. Root cause: original non-psoriasis class was clear healthy skin, not real conditions. This is a distribution shift problem.
- Implemented `models/finetune_external.py` — V4 fine-tuning from V2 checkpoint
- V4 achieves 68% held-out external accuracy (72% psoriasis recall, 64% non-psoriasis recall)

### Review
This sprint validated the entire project's critical finding. The gap between internal and external performance demonstrates the importance of independent external validation in medical AI. V4 is the recommended model for real-world use.

---

## Sprint 5 — Clinical depth, severity, body part context
**Dates:** April 2026 (2 weeks)
**Goal:** Add clinical awareness features, prepare for viva and video.
**Backlog items:** US-02 (extended), US-06, US-08

### Tasks completed
- Implemented `models/severity.py` — visual severity estimation (erythema index, estimated coverage, texture roughness)
- Updated `app/app.py` — body part selector with clinical notes per location, severity tab, full 6-tab UI
- Added body-location-specific clinical context using dermatological knowledge
- Updated `outputs/dissertation_ready_summary.md` with all actual experimental results
- Completed `outputs/known_limitations.md`
- Updated `README.md` with full training and run instructions

### Review
Final sprint. System is a complete research prototype with: 4 model versions, external validation, Grad-CAM explainability, skin-tone proxy analysis, visual severity estimation, body-part clinical context, and comprehensive evaluation outputs. All dissertation artefacts generated.

---

## Sprint Velocity Summary

| Sprint | Goal | Status | Key Outcome |
|---|---|---|---|
| 0 | Setup + research | Complete | Domain knowledge, repository scaffold |
| 1 | Baseline MVP | Complete | Working ResNet18 + Grad-CAM + Gradio UI |
| 2 | Multi-model evaluation | Complete | V1/V2/V3, confusion matrices, metric exports |
| 3 | Fairness + shared preprocessing | Complete | ITA proxy, fairness CSV, UI redesign |
| 4 | External validation + V4 | Complete | Critical distribution-shift finding, V4 model |
| 5 | Clinical depth | Complete | Severity, body part context, final artefacts |

---

## Definition of Done (applied to all sprints)
- Code committed to git
- Outputs generated and saved to `/outputs`
- Tested on at least 10 images from the test split
- No crashes in the Gradio UI for the affected features
