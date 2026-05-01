"""
Figure 7.3 — ROC curves for V1, V2, V3 on Fitzpatrick17k (single axes).

Reads:
    outputs/fitzpatrick_validation/fitzpatrick_predictions.csv

Saves:
    outputs/figures/fig_roc_fitzpatrick.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "outputs/figures"
OUT.mkdir(parents=True, exist_ok=True)

PREDS_CSV = ROOT / "outputs/fitzpatrick_validation/fitzpatrick_predictions.csv"

MODEL_STYLE = {
    "v1": ("V1 — ResNet18 Baseline",  "#1F77B4", "-"),
    "v2": ("V2 — ResNet50 Transfer",  "#FF7F0E", "--"),
    "v3": ("V3 — ResNet50 Balanced",  "#2CA02C", "-."),
}

df = pd.read_csv(PREDS_CSV)
df = df.dropna(subset=["psoriasis_prob"])
true_binary = (df["binary_label"] == "psoriasis").astype(int)

fig, ax = plt.subplots(figsize=(7, 6))

for model_key, (label, colour, linestyle) in MODEL_STYLE.items():
    subset = df[df["model"] == model_key]
    y_true = (subset["binary_label"] == "psoriasis").astype(int).values
    y_prob = subset["psoriasis_prob"].values
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=colour, linestyle=linestyle, linewidth=2,
            label=f"{label}  (AUC = {roc_auc:.4f})")

ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random classifier")
ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=11)
ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
ax.set_title("ROC Curves — Fitzpatrick17k External Benchmark\n(psoriasis vs. 113 other conditions, n = 16,550)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

out_path = OUT / "fig_roc_fitzpatrick.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
