"""
Figure 7.4 — Sensitivity by Fitzpatrick type, grouped bar chart (V1, V2, V3).

Reads:
    outputs/fitzpatrick_validation/fitzpatrick_fairness_by_scale.csv

Saves:
    outputs/figures/fig_fairness_bar.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "outputs/figures"
OUT.mkdir(parents=True, exist_ok=True)

CSV = ROOT / "outputs/fitzpatrick_validation/fitzpatrick_fairness_by_scale.csv"

df = pd.read_csv(CSV)
df = df[df["fitzpatrick_scale"] != -1]  # drop unknown

TYPE_LABELS = {
    1: "Type I\n(very fair)",
    2: "Type II\n(fair)",
    3: "Type III\n(medium)",
    4: "Type IV\n(olive)",
    5: "Type V\n(brown)",
    6: "Type VI\n(dark)",
}

# Normalise model key for matching
def _model_key(name: str) -> str:
    if "v1" in name.lower() or "resnet18" in name.lower(): return "v1"
    if "v2" in name.lower() or "transfer" in name.lower(): return "v2"
    if "v3" in name.lower() or "balanced" in name.lower(): return "v3"
    return name

df["model_key"] = df["model"].apply(_model_key)

MODEL_INFO = [
    ("v1", "V1 — ResNet18", "#1F77B4"),
    ("v2", "V2 — ResNet50 Transfer", "#FF7F0E"),
    ("v3", "V3 — ResNet50 Balanced", "#2CA02C"),
]

scales = sorted(df["fitzpatrick_scale"].unique())
x = np.arange(len(scales))
width = 0.25

fig, ax = plt.subplots(figsize=(11, 6))

for i, (key, label, colour) in enumerate(MODEL_INFO):
    sub = df[df["model_key"] == key].sort_values("fitzpatrick_scale")
    vals = []
    for s in scales:
        row = sub[sub["fitzpatrick_scale"] == s]
        vals.append(float(row["sensitivity"].values[0]) if len(row) else 0.0)
    bars = ax.bar(x + i * width, vals, width, label=label, color=colour, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)

ax.set_xlabel("Fitzpatrick Skin Type", fontsize=11)
ax.set_ylabel("Sensitivity (Psoriasis Detection Rate)", fontsize=11)
ax.set_title("Sensitivity by Fitzpatrick Skin Type — V1, V2, V3\n(Fitzpatrick17k benchmark, n = 16,550)",
             fontsize=11, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels([TYPE_LABELS[s] for s in scales], fontsize=9)
ax.set_ylim(0, 1.08)
ax.axhline(y=0.5, color="grey", linestyle=":", linewidth=1, alpha=0.6)
ax.legend(fontsize=9)
ax.grid(True, axis="y", alpha=0.3)

# Annotate sample sizes
for j, s in enumerate(scales):
    row = df[(df["model_key"] == "v3") & (df["fitzpatrick_scale"] == s)]
    if len(row):
        n_ps = int(row["n_psoriasis"].values[0])
        n_tot = int(row["n"].values[0])
        ax.text(j + width, -0.06, f"n={n_tot}\n({n_ps} +ve)", ha="center", va="top",
                fontsize=7, color="grey", transform=ax.get_xaxis_transform())

fig.tight_layout()
out_path = OUT / "fig_fairness_bar.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
