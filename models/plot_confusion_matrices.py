"""
Figures 7.1 and 7.2 — Confusion matrix collages.

Fig 7.1: Internal test set  — collages the four existing confusion_matrix.png files
Fig 7.2: External validation — generates matrices from external_results.csv

Saves:
    outputs/figures/fig_confusion_internal.png
    outputs/figures/fig_confusion_external.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "outputs/figures"
OUT.mkdir(parents=True, exist_ok=True)

VERSIONS = [
    ("V1", "ResNet18 Baseline",   ROOT / "outputs/v1_resnet18_baseline/confusion_matrix.png"),
    ("V2", "ResNet50 Transfer",   ROOT / "outputs/v2_resnet50/confusion_matrix.png"),
    ("V3", "ResNet50 Balanced",   ROOT / "outputs/v3_resnet50_balanced/confusion_matrix.png"),
    ("V4", "ResNet50 Real-World", ROOT / "outputs/v4_external_finetuned/confusion_matrix.png"),
]

CLASS_NAMES = ["non-psoriasis", "psoriasis"]


# ── Figure 7.1 — internal ──────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for ax, (ver, name, png) in zip(axes, VERSIONS):
    if png.exists():
        img = mpimg.imread(str(png))
        ax.imshow(img)
        ax.set_title(f"{ver} — {name}", fontsize=11, fontweight="bold")
        ax.axis("off")
    else:
        ax.text(0.5, 0.5, f"Missing:\n{png.name}", ha="center", va="center",
                transform=ax.transAxes, color="grey")
        ax.set_title(f"{ver} — {name}", fontsize=11, fontweight="bold")
        ax.axis("off")

fig.suptitle("Confusion Matrices — Internal Test Set (n = 1,501 per model)",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
out7_1 = OUT / "fig_confusion_internal.png"
fig.savefig(out7_1, dpi=150, bbox_inches="tight")
print(f"Saved: {out7_1}")


# ── Figure 7.2 — external ──────────────────────────────────────────────────

ext_csv = ROOT / "outputs/external_validation/external_results.csv"
if not ext_csv.exists():
    print(f"External results not found: {ext_csv}")
else:
    df = pd.read_csv(ext_csv)
    df["true_idx"] = (df["true_label"] == "psoriasis").astype(int)
    df["pred_idx"] = (df["predicted_label"] == "psoriasis").astype(int)

    fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))
    axes2 = axes2.flatten()

    for ax, (ver, name, _) in zip(axes2, VERSIONS):
        sub = df[df["model_version"] == ver.lower()]
        if sub.empty:
            ax.text(0.5, 0.5, f"No data for {ver}", ha="center", va="center",
                    transform=ax.transAxes, color="grey")
            ax.set_title(f"{ver} — {name}", fontsize=11, fontweight="bold")
            ax.axis("off")
            continue

        cm = confusion_matrix(sub["true_idx"], sub["pred_idx"], labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{ver} — {name}\n(n={len(sub)})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
        ax.tick_params(axis="x", labelsize=8, rotation=15)
        ax.tick_params(axis="y", labelsize=8)

        acc = (sub["true_idx"] == sub["pred_idx"]).mean()
        ax.text(0.98, 0.02, f"Acc: {acc:.1%}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    fig2.suptitle("Confusion Matrices — External Validation Set (762 images)",
                  fontsize=13, fontweight="bold", y=1.01)
    fig2.tight_layout()
    out7_2 = OUT / "fig_confusion_external.png"
    fig2.savefig(out7_2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out7_2}")
