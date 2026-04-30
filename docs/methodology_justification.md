# Methodology Justification — DermaRead

## Approach selection: Supervised CNN classification

### Alternatives considered

| Approach | Considered? | Rejected because |
|---|---|---|
| Traditional ML (SVM + hand-crafted features) | Yes | Inferior performance on high-dimensional image data; requires manual feature engineering; worse generalisation |
| CNN trained from scratch | Yes | Requires large labelled dataset (100k+ images); not feasible with available data; transfer learning achieves better results with limited labels |
| **Transfer learning from ImageNet (chosen)** | — | Strong prior knowledge from ImageNet (1.2M images); well-established for medical imaging; achieves good performance with limited data |
| Unsupervised anomaly detection (autoencoder) | Yes | Defining "normal" skin is not well-specified; would require separate healthy-skin-only dataset; classification task benefits from labels |
| Self-supervised pretraining (SimCLR, DINO) | Yes | Requires significant compute and data for pretraining phase; overkill for a binary classification task with available labels |
| Vision Transformer (ViT) | Yes | Requires more data than CNNs for equivalent performance; ResNet50 is better suited to limited-data regimes |

### Why transfer learning is justified

Transfer learning from ImageNet-pretrained ResNets is the dominant approach in medical imaging literature (Esteva et al., 2017; Litjens et al., 2017). ImageNet features (edges, textures, colour gradients) transfer well to skin lesion analysis because:
1. Skin conditions manifest as visible texture and colour changes
2. The low-level features (Gabor-like, colour channels) learned on natural images apply to dermatological images
3. Limited labelled medical data makes from-scratch training impractical

---

## Architecture selection: ResNet18 and ResNet50

**ResNet18** was chosen as the baseline because:
- 11.2M parameters — appropriately sized for limited data (avoids extreme overfitting)
- Well-understood architecture; widely used as a reference in the literature
- Fast training; suitable for iterative experimentation

**ResNet50** was chosen for V2/V3/V4 because:
- 23.5M parameters — greater representational capacity
- Bottleneck architecture allows deeper feature extraction
- Compared directly against ResNet18 to quantify the impact of model capacity

**Why not ResNet101 or ResNet152?**
- Higher parameter count with limited data would increase overfitting risk
- Training time would increase substantially without proportional accuracy gain
- The research question (psoriasis detection + fairness) does not require maximum capacity

---

## Class imbalance mitigation (V3)

The training set contains a 59:41 (psoriasis:non-psoriasis) ratio. Three strategies exist:

| Strategy | Chosen? | Notes |
|---|---|---|
| Oversample minority class | No | Amplifies identical images; can introduce bias |
| Undersample majority class | No | Discards real data unnecessarily |
| **Inverse-frequency class weights in loss (chosen)** | Yes | Penalises misclassification of minority class more heavily; no data modification required |
| Data augmentation (targeted) | Partially | Used in training transform (random flip, colour jitter) |

V3 (class-weighted loss) was compared directly against V2 (unweighted) to isolate the effect. Result: no measurable improvement — confirming the root cause was data quality, not class imbalance.

---

## Fine-tuning strategy (V4)

V4 uses a low learning rate (5×10⁻⁶) fine-tuning from V2's checkpoint on the external dataset. Alternatives:

| Strategy | Considered | Rationale |
|---|---|---|
| Train from ImageNet scratch on external data | Yes | Cleaner start but 609 images may be insufficient for scratch training |
| Fine-tune only classifier head | Yes | Insufficient — the non-psoriasis features learned by the backbone are also wrong |
| **Fine-tune full network at low LR (chosen)** | — | Preserves psoriasis knowledge while adapting non-psoriasis boundary; class weights address slight imbalance |

---

## Explainability: Grad-CAM

**Why Grad-CAM over alternatives?**

| Method | Notes |
|---|---|
| LIME | Model-agnostic, computationally expensive, result is approximation |
| SHAP | Powerful but complex; SHAP DeepExplainer has limitations with batch normalisation |
| Attention maps | Only applicable to attention-based architectures (ViT) |
| **Grad-CAM (chosen)** | Computationally efficient; works natively with ResNets; spatially resolved; widely cited in medical AI |

Grad-CAM (Selvaraju et al., 2017) uses gradients flowing into the final convolutional layer to produce a class-discriminative localisation map. This provides spatial explanations suitable for the clinically motivated aim of the project.

Layer selection: `layer4[-1].conv3` for ResNet50 (Bottleneck); `layer4[-1].conv2` for ResNet18 (BasicBlock). The final spatial feature map before global average pooling gives the richest spatial information.

---

## Fairness proxy: ITA

**Why ITA over alternatives?**

| Method | Notes |
|---|---|
| Fitzpatrick scale | Requires clinical annotation by a dermatologist; not available in dataset |
| Individual Typology Angle (ITA) | Computable from image pixels; used in academic literature [Chardon et al., 1991]; standard proxy in fairness research |
| CIELAB L* value alone | Less discriminative than ITA; ignores b* component |
| **ITA (chosen)** | Best available proxy given unlabelled data; explicitly framed as estimate |

Limitations explicitly acknowledged: ITA is distorted by lesion colour, lighting, and compression. Results are presented as exploratory, not definitive.

---

## Software engineering principles applied

### DRY 
- `models/preprocessing.py` — single transform definition imported everywhere; previously each file had its own inline `T.Compose()` block
- `models/dataset.py` `ManifestDataset` — single class used across all training scripts

### YAGNI 
- Subtype classifier not built — data showed <15 images per subtype, insufficient for training; code structure prepared but disabled
- No database server deployed — CSV manifests are sufficient for the dataset scale; SQLite would add complexity with no benefit

### SOLID
- **Single Responsibility:** each module has one job (`skin_tone_ita.py` does ITA, `fairness_eval.py` does evaluation, `severity.py` does severity)
- **Open/Closed:** new model versions (V4) added by extending `resnet50_model.py` with a new loader function, not modifying existing V2/V3 code
- **Liskov Substitution:** all models are interchangeable in the UI — any model passed to `predict_pil()` works identically
- **Dependency Inversion:** `app.py` depends on abstract `predict_pil()` interface, not on specific model internals
