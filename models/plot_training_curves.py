"""
Figure 5.2 — Training and validation curves for V1–V4 (2×2 grid).

Reads:
    outputs/v1_resnet18_baseline/training_log.csv
    outputs/v2_resnet50/training_log.csv
    outputs/v3_resnet50_balanced/training_log.csv
    outputs/v4_external_finetuned/training_log.csv

Saves:
    outputs/figures/fig_training_curves.png
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "outputs/figures"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("V1 — ResNet18 Baseline",        ROOT / "outputs/v1_resnet18_baseline/training_log.csv",  "#1F77B4"),
    ("V2 — ResNet50 Transfer",         ROOT / "outputs/v2_resnet50/training_log.csv",            "#FF7F0E"),
    ("V3 — ResNet50 Balanced",         ROOT / "outputs/v3_resnet50_balanced/training_log.csv",   "#2CA02C"),
    ("V4 — ResNet50 Real-World",       ROOT / "outputs/v4_external_finetuned/training_log.csv",  "#D62728"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for ax, (title, csv_path, colour) in zip(axes, MODELS):
    if not csv_path.exists():
        ax.text(0.5, 0.5, f"No log: {csv_path.parent.name}", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="grey")
        ax.set_title(title, fontsize=11, fontweight="bold")
        continue

    df = pd.read_csv(csv_path)
    epochs = df["epoch"]

    ax2 = ax.twinx()

    l1, = ax.plot(epochs, df["train_loss"], color=colour,   linewidth=2,   label="Train loss")
    l2, = ax2.plot(epochs, df["val_acc"] * 100, color=colour, linewidth=2,
                   linestyle="--", alpha=0.7, label="Val accuracy (%)")

    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Training loss", fontsize=9, color=colour)
    ax2.set_ylabel("Val accuracy (%)", fontsize=9, color=colour)
    ax2.set_ylim(0, 105)
    ax.tick_params(axis="y", labelcolor=colour)
    ax2.tick_params(axis="y", labelcolor=colour)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=8, loc="upper right")

fig.suptitle("Training Loss and Validation Accuracy — V1 to V4", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
out_path = OUT / "fig_training_curves.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
