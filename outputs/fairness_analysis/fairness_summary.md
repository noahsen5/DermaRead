# Fairness Summary — Estimated Skin-Tone Proxy

> **Disclaimer:** Groups are estimated from ITA (Individual Typology Angle).
> ITA is affected by lesion colour, lighting, white-balance, and compression.
> These groups do NOT map to Fitzpatrick phototype scale.
> Sample sizes per group may be small, making estimates unreliable.

## v1 — ResNet18 Baseline

| Group | N | Accuracy | F1 | FPR | FNR |
|---|---|---|---|---|---|
| Dark | 189 | 100.00% | 1.0000 | 0.0000 | 0.0000 |
| Light | 514 | 100.00% | 1.0000 | 0.0000 | 0.0000 |
| Medium | 798 | 100.00% | 1.0000 | 0.0000 | 0.0000 |

- **Best group:** Dark (100.00%)
- **Worst group:** Dark (100.00%)
- **Accuracy gap:** 0.00%

## v2 — ResNet50 Transfer

| Group | N | Accuracy | F1 | FPR | FNR |
|---|---|---|---|---|---|
| Dark | 189 | 99.47% | 0.9948 | 0.0000 | 0.0057 |
| Light | 514 | 100.00% | 1.0000 | 0.0000 | 0.0000 |
| Medium | 798 | 100.00% | 1.0000 | 0.0000 | 0.0000 |

- **Best group:** Light (100.00%)
- **Worst group:** Dark (99.47%)
- **Accuracy gap:** 0.53%

## v3 — ResNet50 Balanced

| Group | N | Accuracy | F1 | FPR | FNR |
|---|---|---|---|---|---|
| Dark | 189 | 100.00% | 1.0000 | 0.0000 | 0.0000 |
| Light | 514 | 100.00% | 1.0000 | 0.0000 | 0.0000 |
| Medium | 798 | 100.00% | 1.0000 | 0.0000 | 0.0000 |

- **Best group:** Dark (100.00%)
- **Worst group:** Dark (100.00%)
- **Accuracy gap:** 0.00%

## Effect of Class Balancing (V2 vs V3)

- V2 accuracy gap: 0.53%
- V3 accuracy gap: 0.00%
- The balanced model **reduced** the cross-group performance gap.

## Limitations

1. ITA computed from lesion images is biased by the lesion colour itself.
2. Skin-pixel detection uses loose HSV thresholds; background and clothing pixels may be included.
3. The three-group scheme is coarse and may not capture skin-tone diversity.
4. Group sample sizes are likely unbalanced; accuracy estimates can be high-variance.
5. No formal demographic metadata exists in this dataset; proxy methods are inherently imprecise.