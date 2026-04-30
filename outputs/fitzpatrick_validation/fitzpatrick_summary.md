# External Validation — Fitzpatrick17k

Independent evaluation on the Fitzpatrick17k dataset (Groh et al., 2021).
These images were **not** used in training or internal test splits.

**Binary task:** psoriasis / pustular psoriasis → positive; all other conditions → negative.

Total images evaluated: 16,550

## Overall Metrics

| Model | N | AUC | AP | Accuracy | Sensitivity | Specificity | PPV | F1 |
|---|---|---|---|---|---|---|---|---|
| v1 (ResNet18 Baseline) | 16,550 | 0.5531 | 0.0529 | 0.2947 | 0.7745 | 0.2733 | 0.0453 | 0.0855 |
| v2 (ResNet50 Transfer) | 16,550 | 0.6130 | 0.0678 | 0.4194 | 0.7404 | 0.4051 | 0.0525 | 0.0980 |
| v3 (ResNet50 Balanced) | 16,550 | 0.6271 | 0.0767 | 0.3731 | 0.7830 | 0.3548 | 0.0512 | 0.0962 |

## Fairness by Fitzpatrick Scale (Real Labels)

> Unlike the ITA proxy used in training-set fairness analysis, these groups
> use the **original Fitzpatrick scale ratings** from the dataset.
> Sensitivity (psoriasis detection rate) is the primary fairness metric.

### v1 (ResNet18 Baseline)

| Fitzpatrick Type | N | Psoriasis N | AUC | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|---|---|
| Unknown | 565 | 32 | 0.6124 | 0.8438 | 0.2608 | 0.2938 |
| Type I (very fair) | 2,944 | 129 | 0.5201 | 0.6667 | 0.3275 | 0.3424 |
| Type II (fair) | 4,802 | 250 | 0.5352 | 0.7720 | 0.2797 | 0.3053 |
| Type III (medium) | 3,304 | 105 | 0.5926 | 0.8476 | 0.2579 | 0.2766 |
| Type IV (olive) | 2,775 | 98 | 0.6160 | 0.8571 | 0.2417 | 0.2634 |
| Type V (brown) | 1,529 | 67 | 0.5082 | 0.7164 | 0.2373 | 0.2583 |
| Type VI (dark brown/black) | 631 | 24 | 0.5977 | 0.7917 | 0.2932 | 0.3122 |

- **Best sensitivity:** Type IV (olive) (0.8571)
- **Worst sensitivity:** Type I (very fair) (0.6667)
- **Sensitivity gap:** 0.1904

### v2 (ResNet50 Transfer)

| Fitzpatrick Type | N | Psoriasis N | AUC | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|---|---|
| Unknown | 565 | 32 | 0.6400 | 0.8750 | 0.3133 | 0.3451 |
| Type I (very fair) | 2,944 | 129 | 0.5784 | 0.6434 | 0.4707 | 0.4783 |
| Type II (fair) | 4,802 | 250 | 0.5970 | 0.6960 | 0.4332 | 0.4469 |
| Type III (medium) | 3,304 | 105 | 0.6112 | 0.7333 | 0.4151 | 0.4252 |
| Type IV (olive) | 2,775 | 98 | 0.6411 | 0.8061 | 0.3717 | 0.3870 |
| Type V (brown) | 1,529 | 67 | 0.6935 | 0.8806 | 0.3064 | 0.3316 |
| Type VI (dark brown/black) | 631 | 24 | 0.6721 | 0.9167 | 0.3031 | 0.3265 |

- **Best sensitivity:** Type VI (dark brown/black) (0.9167)
- **Worst sensitivity:** Type I (very fair) (0.6434)
- **Sensitivity gap:** 0.2733

### v3 (ResNet50 Balanced)

| Fitzpatrick Type | N | Psoriasis N | AUC | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|---|---|
| Unknown | 565 | 32 | 0.6375 | 0.8125 | 0.2833 | 0.3133 |
| Type I (very fair) | 2,944 | 129 | 0.6150 | 0.7364 | 0.4149 | 0.4290 |
| Type II (fair) | 4,802 | 250 | 0.6167 | 0.7480 | 0.3759 | 0.3953 |
| Type III (medium) | 3,304 | 105 | 0.6191 | 0.7905 | 0.3632 | 0.3768 |
| Type IV (olive) | 2,775 | 98 | 0.6293 | 0.8061 | 0.3224 | 0.3395 |
| Type V (brown) | 1,529 | 67 | 0.7168 | 0.9104 | 0.2756 | 0.3035 |
| Type VI (dark brown/black) | 631 | 24 | 0.6680 | 0.8750 | 0.2702 | 0.2932 |

- **Best sensitivity:** Type V (brown) (0.9104)
- **Worst sensitivity:** Type I (very fair) (0.7364)
- **Sensitivity gap:** 0.1740

## Dataset Notes

- Source: Fitzpatrick17k (Groh et al., 2021) — 16,577 dermatology images, 114 conditions.
- QC exclusions: 'Wrongly labelled' and 'Other' rows removed.
- Fitzpatrick scale -1 indicates images without a skin-tone rating.
- Psoriasis prevalence in this dataset (~4.3%) reflects real-world rarity,
  which inflates apparent accuracy but suppresses sensitivity.