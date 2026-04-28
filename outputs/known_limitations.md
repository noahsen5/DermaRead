# Known Limitations

This file logs limitations discovered during development.
Update with actual results after training and evaluation.

---

## Training Observations (April 2026)

- **100% validation accuracy and loss → 0.0000 observed for V2 and V3.** Investigation confirmed no structural data leakage: same-numbered images across classes (e.g. `aug_0_1010.jpg` in both psoriasis and non-psoriasis folders) are pixel-different files with distinct content. The high accuracy reflects the dataset's relative simplicity for a fine-tuned ResNet50 pretrained on ImageNet, combined with a curated/non-clinical image distribution.
- **Overfitting despite high val accuracy:** Training loss collapsing to 0.0000 indicates the model has memorised the training set. L2 regularisation (weight_decay=1e-4) added to all optimisers to mitigate this. This does not guarantee real-world generalisation.
- **Dataset is curated, not clinical:** All images are sourced from a user-provided augmented set. Performance on diverse, real-world clinical images is unknown and likely lower.

## Data

- **No subtype labels:** The dataset contains no valid psoriasis subtype labels (plaque, guttate, inverse, pustular). The subtype classifier has not been trained. The UI displays an explanatory message.
- **Skin-tone metadata absent:** All entries in manifest.csv have `skin_tone = unknown`. Fairness analysis relies entirely on the ITA-based estimated skin-tone proxy, which is imperfect.
- **Dataset balance unknown until training:** Class distribution should be checked after running `data/pipeline_from_manifest.py` to assess whether balancing is necessary.
- **Image provenance:** All images are listed as `source = user_provided` with no licence URL or consent field populated. Ethical review and licensing should be confirmed before any public release.

## ITA Skin-Tone Proxy

- ITA computed from lesion images is biased by lesion colour (e.g. erythema, silvery scale).
- Background pixels, clothing, and shadows may be included in the skin-pixel estimate.
- The three-group scheme (Light / Medium / Dark) is a coarse approximation of skin-tone diversity.
- ITA does not map to the Fitzpatrick phototype scale; terminology must be "estimated skin-tone proxy" throughout.

## Model

- Models are trained on a single institution's augmented dataset — generalisation to clinical images from other sources is unknown.
- No external validation set has been used.
- Class-weighted loss (V3) mitigates imbalance during training but does not guarantee fairness across demographic groups.
- Grad-CAM heatmaps show model attention, not clinically validated diagnostic regions.

## Fairness Evaluation

- Fairness analysis is conducted on the test split only, which may contain very few images per skin-tone proxy group.
- Small group sizes make accuracy and F1 estimates high-variance and potentially misleading.
- This analysis cannot substitute for a study with properly annotated demographic metadata.

## Reproducibility

- Results may differ slightly across operating systems and hardware due to floating-point non-determinism in PyTorch (particularly on GPU).
- Random seed is fixed at 42 in all training scripts, but DataLoader worker randomness on multi-core systems is not fully seeded.

---

*Log entries should be updated with specific values (e.g. class counts, group sizes) after training scripts have been run.*
