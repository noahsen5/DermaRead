"""
Fairness evaluation by estimated skin-tone proxy group.

Requires data/processed_manifest.csv (run models/skin_tone_ita.py first).
Evaluates every trained model version on the test split,
broken down by estimated_skin_tone_proxy (Light / Medium / Dark / unknown).

Outputs:
    outputs/fairness_analysis/fairness_by_skin_tone.csv
    outputs/fairness_analysis/fairness_summary.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.dataset import CLASS_NAMES

PROCESSED_MANIFEST = ROOT / "data/processed_manifest.csv"
FAIRNESS_DIR = ROOT / "outputs/fairness_analysis"

_PREPROCESS = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_VERSIONS = {
    "v1": ("ResNet18 Baseline", "models.baseline", "build_model",
           ROOT / "models/checkpoints/resnet18_baseline.pt"),
    "v2": ("ResNet50 Transfer", "models.resnet50_model", "build_resnet50",
           ROOT / "models/checkpoints/resnet50_v2.pt"),
    "v3": ("ResNet50 Balanced", "models.resnet50_model", "build_resnet50",
           ROOT / "models/checkpoints/resnet50_v3_balanced.pt"),
}


def _load_model(version: str, device: str):
    label, module_name, fn_name, ckpt = _VERSIONS[version]
    import importlib
    mod = importlib.import_module(module_name)
    build_fn = getattr(mod, fn_name)
    return build_fn(weights=None, checkpoint_path=ckpt, device=device)


def _run_predictions(model, rows: pd.DataFrame, img_root: Path, device: str):
    label_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    preds, labels = [], []
    model.eval()
    with torch.no_grad():
        for _, row in rows.iterrows():
            img_path = img_root / row["relative_path"]
            if not img_path.is_file():
                continue
            try:
                pil = Image.open(img_path).convert("RGB")
                x = _PREPROCESS(pil).unsqueeze(0).to(device)
                pred = int(model(x).argmax(1).item())
                true = label_to_idx.get(row["label"], -1)
                if true == -1:
                    continue
                preds.append(pred)
                labels.append(true)
            except Exception:
                continue
    return np.array(labels), np.array(preds)


def _group_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score, confusion_matrix,
        f1_score, precision_score, recall_score,
    )
    n = len(labels)
    if n == 0:
        return dict(sample_count=0, accuracy=0, precision=0, recall=0,
                    f1=0, false_positive_rate=0, false_negative_rate=0)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="weighted", zero_division=0)
    rec = recall_score(labels, preds, average="weighted", zero_division=0)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASS_NAMES))))
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    else:
        fpr = fnr = float("nan")

    return dict(sample_count=n, accuracy=round(acc, 4), precision=round(prec, 4),
                recall=round(rec, 4), f1=round(f1, 4),
                false_positive_rate=round(fpr, 4), false_negative_rate=round(fnr, 4))


# ── Summary markdown ──────────────────────────────────────────────────────────

def _write_summary(df: pd.DataFrame) -> None:
    lines = [
        "# Fairness Summary — Estimated Skin-Tone Proxy",
        "",
        "> **Disclaimer:** Groups are estimated from ITA (Individual Typology Angle).",
        "> ITA is affected by lesion colour, lighting, white-balance, and compression.",
        "> These groups do NOT map to Fitzpatrick phototype scale.",
        "> Sample sizes per group may be small, making estimates unreliable.",
        "",
    ]

    for version in df["model_version"].unique():
        vdf = df[df["model_version"] == version].reset_index(drop=True)
        model_name = _VERSIONS.get(version, (version,))[0]
        lines += [f"## {version} — {model_name}", ""]
        lines += ["| Group | N | Accuracy | F1 | FPR | FNR |",
                  "|---|---|---|---|---|---|"]
        for _, row in vdf.iterrows():
            lines.append(
                f"| {row['skin_tone_proxy']} | {row['sample_count']} | "
                f"{row['accuracy']:.2%} | {row['f1']:.4f} | "
                f"{row['false_positive_rate']:.4f} | {row['false_negative_rate']:.4f} |"
            )
        lines.append("")
        valid = vdf[vdf["sample_count"] > 0]
        if len(valid) > 1:
            best = valid.loc[valid["accuracy"].idxmax()]
            worst = valid.loc[valid["accuracy"].idxmin()]
            gap = best["accuracy"] - worst["accuracy"]
            lines += [
                f"- **Best group:** {best['skin_tone_proxy']} ({best['accuracy']:.2%})",
                f"- **Worst group:** {worst['skin_tone_proxy']} ({worst['accuracy']:.2%})",
                f"- **Accuracy gap:** {gap:.2%}",
                "",
            ]

    # Cross-version fairness delta
    gaps = {}
    for ver in df["model_version"].unique():
        vdf = df[(df["model_version"] == ver) & (df["sample_count"] > 0)]
        if len(vdf) > 1:
            gaps[ver] = vdf["accuracy"].max() - vdf["accuracy"].min()

    if "v2" in gaps and "v3" in gaps:
        lines += ["## Effect of Class Balancing (V2 vs V3)", ""]
        lines.append(f"- V2 accuracy gap: {gaps['v2']:.2%}")
        lines.append(f"- V3 accuracy gap: {gaps['v3']:.2%}")
        if gaps["v3"] < gaps["v2"]:
            lines.append("- The balanced model **reduced** the cross-group performance gap.")
        else:
            lines.append(
                "- The balanced model did **not** substantially reduce the fairness gap; "
                "further mitigation may be required."
            )
        lines.append("")

    lines += [
        "## Limitations",
        "",
        "1. ITA computed from lesion images is biased by the lesion colour itself.",
        "2. Skin-pixel detection uses loose HSV thresholds; background and clothing pixels may be included.",
        "3. The three-group scheme is coarse and may not capture skin-tone diversity.",
        "4. Group sample sizes are likely unbalanced; accuracy estimates can be high-variance.",
        "5. No formal demographic metadata exists in this dataset; proxy methods are inherently imprecise.",
    ]

    out = FAIRNESS_DIR / "fairness_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_fairness_evaluation(versions: list[str] | None = None) -> None:
    if not PROCESSED_MANIFEST.exists():
        print("processed_manifest.csv not found. Run:  python models/skin_tone_ita.py")
        return

    FAIRNESS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED_MANIFEST)

    if "split" in df.columns:
        df = df[df["split"] == "test"].reset_index(drop=True)

    if "estimated_skin_tone_proxy" not in df.columns:
        print("Column 'estimated_skin_tone_proxy' missing. Run models/skin_tone_ita.py first.")
        return

    img_root = ROOT / "data/raw"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    versions = versions or list(_VERSIONS.keys())
    groups = sorted(df["estimated_skin_tone_proxy"].dropna().unique())

    rows = []
    for ver in versions:
        if not _VERSIONS[ver][3].exists():
            print(f"[{ver}] checkpoint not found — skipping.")
            continue
        print(f"Evaluating {ver}...")
        try:
            model = _load_model(ver, device)
        except Exception as e:
            print(f"  Could not load {ver}: {e}")
            continue

        for group in groups:
            group_df = df[df["estimated_skin_tone_proxy"] == group]
            labels, preds = _run_predictions(model, group_df, img_root, device)
            m = _group_metrics(labels, preds)
            print(f"  {group}: n={m['sample_count']}, acc={m['accuracy']:.2%}")
            rows.append({"model_version": ver, "skin_tone_proxy": group, **m})

    if not rows:
        print("No results to save.")
        return

    result_df = pd.DataFrame(rows)
    out_csv = FAIRNESS_DIR / "fairness_by_skin_tone.csv"
    result_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    _write_summary(result_df)


if __name__ == "__main__":
    run_fairness_evaluation()
