"""
Figure 7.6 — Calibration plot (reliability diagram) for V4 on held-out external set.

Reads:
    outputs/v4_external_finetuned/external_test_results.csv

Saves:
    outputs/figures/fig_calibration_v4.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "outputs/figures"
OUT.mkdir(parents=True, exist_ok=True)
CSV  = ROOT / "outputs/v4_external_finetuned/external_test_results.csv"

df = pd.read_csv(CSV)
y_true = (df["true_label"] == "psoriasis").astype(int).values
y_prob = df["psoriasis_probability"].values
n = len(y_true)

# Use fewer bins when n is small to avoid empty bins
n_bins = min(10, max(5, n // 10))
frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
brier = brier_score_loss(y_true, y_prob)

# Expected Calibration Error
bin_edges = np.linspace(0, 1, n_bins + 1)
ece_sum = 0.0
for lo, hi, fp, mp in zip(bin_edges[:-1], bin_edges[1:], frac_pos, mean_pred):
    mask = (y_prob >= lo) & (y_prob < hi)
    bin_n = mask.sum()
    if bin_n > 0:
        ece_sum += (bin_n / n) * abs(fp - mp)
ece = ece_sum

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# ── Left: reliability diagram ─────
ax = axes[0]
ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")
ax.plot(mean_pred, frac_pos, "o-", color="#D62728", linewidth=2, markersize=6,
        label=f"V4  (Brier={brier:.4f}, ECE={ece:.4f})")
ax.fill_between(mean_pred, frac_pos, mean_pred,
                alpha=0.1, color="#D62728", label="Calibration gap")
ax.set_xlabel("Mean predicted probability", fontsize=11)
ax.set_ylabel("Fraction of positives", fontsize=11)
ax.set_title("Reliability Diagram — V4 Held-Out External Set", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

# ── Right: predicted probability histogram ───
ax2 = axes[1]
pos_probs = y_prob[y_true == 1]
neg_probs = y_prob[y_true == 0]
bins = np.linspace(0, 1, 21)
ax2.hist(neg_probs, bins=bins, alpha=0.6, color="#1F77B4", label="Non-psoriasis (true negative class)")
ax2.hist(pos_probs, bins=bins, alpha=0.6, color="#D62728", label="Psoriasis (true positive class)")
ax2.axvline(x=0.5, color="grey", linestyle="--", linewidth=1.2, label="Decision threshold (0.5)")
ax2.set_xlabel("Predicted psoriasis probability", fontsize=11)
ax2.set_ylabel("Image count", fontsize=11)
ax2.set_title("Predicted Probability Distribution — V4", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

fig.suptitle(f"V4 Calibration Analysis  |  n={n} held-out images  |  "
             f"Brier score={brier:.4f}  |  ECE={ece:.4f}",
             fontsize=10, y=1.01)
fig.tight_layout()

out_path = OUT / "fig_calibration_v4.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
print(f"Brier score: {brier:.4f}  (0 = perfect, 0.25 = uninformative)")
print(f"ECE:         {ece:.4f}")
