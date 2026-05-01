# External Validation — Fitzpatrick17k

Independent evaluation on the Fitzpatrick17k dataset (Groh et al., 2021).
These images were **not** used in training or internal test splits.

**Binary task:** psoriasis / pustular psoriasis → positive; all other conditions → negative.

Total images evaluated: 16,550

## Overall Metrics

| Model | N | AUC | AP | Accuracy | Sensitivity | Specificity | PPV | F1 |
|---|---|---|---|---|---|---|---|---|
| v1 (ResNet18 Baseline) | 16,550 | 0.6131 | 0.0637 | 0.5657 | 0.6071 | 0.5639 | 0.0583 | 0.1064 |
| v2 (ResNet50 Transfer) | 16,550 | 0.6204 | 0.0709 | 0.4576 | 0.7064 | 0.4465 | 0.0537 | 0.0999 |
| v3 (ResNet50 Balanced) | 16,550 | 0.6300 | 0.0760 | 0.4763 | 0.7106 | 0.4658 | 0.0559 | 0.1036 |

## Fairness by Fitzpatrick Scale (Real Labels)

> Unlike the ITA proxy used in training-set fairness analysis, these groups
> use the **original Fitzpatrick scale ratings** from the dataset.
> Sensitivity (psoriasis detection rate) is the primary fairness metric.

### v1 (ResNet18 Baseline)

| Fitzpatrick Type | N | Psoriasis N | AUC | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|---|---|
| Unknown | 565 | 32 | 0.6282 | 0.7500 | 0.4615 | 0.4779 |
| Type I (very fair) | 2,944 | 129 | 0.6130 | 0.4961 | 0.6252 | 0.6196 |
| Type II (fair) | 4,802 | 250 | 0.6046 | 0.5640 | 0.6217 | 0.6187 |
| Type III (medium) | 3,304 | 105 | 0.5936 | 0.5524 | 0.5927 | 0.5914 |
| Type IV (olive) | 2,775 | 98 | 0.6696 | 0.7245 | 0.5196 | 0.5268 |
| Type V (brown) | 1,529 | 67 | 0.6425 | 0.8209 | 0.4056 | 0.4238 |
| Type VI (dark brown/black) | 631 | 24 | 0.5646 | 0.6250 | 0.3608 | 0.3708 |

- **Best sensitivity:** Type V (brown) (0.8209)
- **Worst sensitivity:** Type I (very fair) (0.4961)
- **Sensitivity gap:** 0.3248

### v2 (ResNet50 Transfer)

| Fitzpatrick Type | N | Psoriasis N | AUC | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|---|---|
| Unknown | 565 | 32 | 0.5939 | 0.7188 | 0.3565 | 0.3770 |
| Type I (very fair) | 2,944 | 129 | 0.5662 | 0.5736 | 0.5094 | 0.5122 |
| Type II (fair) | 4,802 | 250 | 0.6194 | 0.6880 | 0.4785 | 0.4894 |
| Type III (medium) | 3,304 | 105 | 0.6265 | 0.7143 | 0.4676 | 0.4755 |
| Type IV (olive) | 2,775 | 98 | 0.6673 | 0.8061 | 0.4064 | 0.4205 |
| Type V (brown) | 1,529 | 67 | 0.6853 | 0.8358 | 0.3413 | 0.3630 |
| Type VI (dark brown/black) | 631 | 24 | 0.6236 | 0.7917 | 0.3130 | 0.3312 |

- **Best sensitivity:** Type V (brown) (0.8358)
- **Worst sensitivity:** Type I (very fair) (0.5736)
- **Sensitivity gap:** 0.2622

### v3 (ResNet50 Balanced)

| Fitzpatrick Type | N | Psoriasis N | AUC | Sensitivity | Specificity | Accuracy |
|---|---|---|---|---|---|---|
| Unknown | 565 | 32 | 0.6085 | 0.7188 | 0.3771 | 0.3965 |
| Type I (very fair) | 2,944 | 129 | 0.5814 | 0.5891 | 0.5222 | 0.5251 |
| Type II (fair) | 4,802 | 250 | 0.6221 | 0.6760 | 0.4980 | 0.5073 |
| Type III (medium) | 3,304 | 105 | 0.6324 | 0.7143 | 0.4817 | 0.4891 |
| Type IV (olive) | 2,775 | 98 | 0.6781 | 0.8265 | 0.4300 | 0.4440 |
| Type V (brown) | 1,529 | 67 | 0.7123 | 0.8806 | 0.3680 | 0.3905 |
| Type VI (dark brown/black) | 631 | 24 | 0.6275 | 0.7500 | 0.3509 | 0.3661 |

- **Best sensitivity:** Type V (brown) (0.8806)
- **Worst sensitivity:** Type I (very fair) (0.5891)
- **Sensitivity gap:** 0.2915

## Dataset Notes

- Source: Fitzpatrick17k (Groh et al., 2021) — 16,577 dermatology images, 114 conditions.
- QC exclusions: 'Wrongly labelled' and 'Other' rows removed.
- Fitzpatrick scale -1 indicates images without a skin-tone rating.
- Psoriasis prevalence in this dataset (~4.3%) reflects real-world rarity,
  which inflates apparent accuracy but suppresses sensitivity.