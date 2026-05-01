"""
Figure 7.5 — Threshold sensitivity sweep with 95% bootstrap CIs for V4 held-out set.

Reads:
    outputs/v4_external_finetuned/external_test_results.csv

Saves:
    outputs/figures/fig_threshold_sweep.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = Path(__file__).resolve().parents[1]
OUT     = ROOT / "outputs/figures"
OUT.mkdir(parents=True, exist_ok=True)
CSV     = ROOT / "outputs/v4_external_finetuned/external_test_results.csv"
N_BOOT  = 1000
SEED    = 42
rng     = np.random.default_rng(SEED)

df = pd.read_csv(CSV)
y_true = (df["true_label"] == "psoriasis").astype(int).values
y_prob = df["psoriasis_probability"].values

thresholds = np.arange(0.30, 0.71, 0.01)

def _metrics(y_true, y_prob, thresh):
    pred = (y_prob >= thresh).astype(int)
    acc  = (pred == y_true).mean()
    tp = ((pred == 1) & (y_true == 1)).sum()
    fn = ((pred == 0) & (y_true == 1)).sum()
    tn = ((pred == 0) & (y_true == 0)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return acc, sens, spec

point = np.array([_metrics(y_true, y_prob, t) for t in thresholds])
acc_pt, sens_pt, spec_pt = point[:, 0], point[:, 1], point[:, 2]

# Bootstrap CIs
boot_acc  = np.zeros((N_BOOT, len(thresholds)))
boot_sens = np.zeros((N_BOOT, len(thresholds)))
boot_spec = np.zeros((N_BOOT, len(thresholds)))
n = len(y_true)
for b in range(N_BOOT):
    idx = rng.integers(0, n, n)
    for j, t in enumerate(thresholds):
        a, s, sp = _metrics(y_true[idx], y_prob[idx], t)
        boot_acc[b, j]  = a
        boot_sens[b, j] = s
        boot_spec[b, j] = sp

lo_acc,  hi_acc  = np.percentile(boot_acc,  [2.5, 97.5], axis=0)
lo_sens, hi_sens = np.percentile(boot_sens, [2.5, 97.5], axis=0)
lo_spec, hi_spec = np.percentile(boot_spec, [2.5, 97.5], axis=0)

fig, ax = plt.subplots(figsize=(9, 5))

for vals, lo, hi, colour, label in [
    (acc_pt,  lo_acc,  hi_acc,  "#1F77B4", "Accuracy"),
    (sens_pt, lo_sens, hi_sens, "#D62728", "Sensitivity"),
    (spec_pt, lo_spec, hi_spec, "#2CA02C", "Specificity"),
]:
    ax.plot(thresholds, vals, color=colour, linewidth=2, label=label)
    ax.fill_between(thresholds, lo, hi, color=colour, alpha=0.15)

ax.axvline(x=0.5, color="grey", linestyle="--", linewidth=1, alpha=0.7, label="Default threshold (0.5)")
ax.set_xlabel("Decision Threshold", fontsize=11)
ax.set_ylabel("Metric value", fontsize=11)
ax.set_title(f"V4 Threshold Sensitivity Sweep (n = {n} held-out images)\n"
             "Shaded band = 95% bootstrap CI (1,000 resamples)", fontsize=11, fontweight="bold")
ax.set_xlim(0.30, 0.70)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

out_path = OUT / "fig_threshold_sweep.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")

# Print the 0.5 threshold row for reference
i50 = np.argmin(np.abs(thresholds - 0.50))
print(f"\nAt threshold=0.50:")
print(f"  Accuracy:    {acc_pt[i50]:.4f}  [{lo_acc[i50]:.4f}, {hi_acc[i50]:.4f}]")
print(f"  Sensitivity: {sens_pt[i50]:.4f}  [{lo_sens[i50]:.4f}, {hi_sens[i50]:.4f}]")
print(f"  Specificity: {spec_pt[i50]:.4f}  [{lo_spec[i50]:.4f}, {hi_spec[i50]:.4f}]")
