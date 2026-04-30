"""
External validation on the Fitzpatrick17k dataset.

Images live at  data/fitzpatrick17k/data/finalfitz17k/<md5hash>.jpg
CSV lives at    data/fitzpatrick17k.csv

Binary mapping:
  psoriasis / pustular psoriasis  → psoriasis
  everything else                 → non-psoriasis

Fairness analysis uses REAL Fitzpatrick scale labels (1–6), which is
stronger evidence than the ITA proxy used in the training-set fairness eval.

Usage:
    python models/evaluate_fitzpatrick.py            # all models
    python models/evaluate_fitzpatrick.py --model v3
    python models/evaluate_fitzpatrick.py --quick    # 500-image sanity check
"""

from __future__ import annotations

import argparse
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

IMG_DIR = ROOT / "data/fitzpatrick17k/data/finalfitz17k"
CSV_PATH = ROOT / "data/fitzpatrick17k.csv"
OUT_DIR  = ROOT / "outputs/fitzpatrick_validation"

PSORIASIS_LABELS = {"psoriasis", "pustular psoriasis"}

# QC values that indicate unreliable labels — exclude these
BAD_QC = {"3 Wrongly labelled", "4 Other"}

_VERSIONS = {
    "v1": ("ResNet18 Baseline",  "models.baseline",      "build_model",
           ROOT / "models/checkpoints/resnet18_baseline.pt"),
    "v2": ("ResNet50 Transfer",  "models.resnet50_model", "build_resnet50",
           ROOT / "models/checkpoints/resnet50_v2.pt"),
    "v3": ("ResNet50 Balanced",  "models.resnet50_model", "build_resnet50",
           ROOT / "models/checkpoints/resnet50_v3_balanced.pt"),
}

_PREPROCESS = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

FITZ_LABELS = {
    -1: "Unknown",
    1:  "Type I (very fair)",
    2:  "Type II (fair)",
    3:  "Type III (medium)",
    4:  "Type IV (olive)",
    5:  "Type V (brown)",
    6:  "Type VI (dark brown/black)",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset(quick: bool = False) -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)

    # Drop QC-flagged rows
    df = df[~df["qc"].isin(BAD_QC)].copy()

    # Binary label
    df["binary_label"] = df["label"].apply(
        lambda x: "psoriasis" if x in PSORIASIS_LABELS else "non-psoriasis"
    )
    df["binary_idx"] = (df["binary_label"] == "psoriasis").astype(int)

    # Keep only images that exist on disk
    df["img_path"] = df["md5hash"].apply(lambda h: IMG_DIR / f"{h}.jpg")
    df = df[df["img_path"].apply(lambda p: p.is_file())].copy()

    if quick:
        pos = df[df["binary_idx"] == 1].sample(min(250, len(df[df["binary_idx"] == 1])),
                                                random_state=42)
        neg = df[df["binary_idx"] == 0].sample(min(250, len(df[df["binary_idx"] == 0])),
                                                random_state=42)
        df = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"[quick] sampled {len(df)} images ({len(pos)} pos, {len(neg)} neg)")

    df["fitzpatrick_scale"] = pd.to_numeric(df["fitzpatrick_scale"], errors="coerce").fillna(-1).astype(int)
    return df.reset_index(drop=True)


# ── Model loading / inference ─────────────────────────────────────────────────

def _load_model(version: str, device: str):
    _, module_name, fn_name, ckpt = _VERSIONS[version]
    import importlib
    mod = importlib.import_module(module_name)
    build_fn = getattr(mod, fn_name)
    return build_fn(weights=None, checkpoint_path=ckpt, device=device)


def run_inference(model, df: pd.DataFrame, device: str) -> np.ndarray:
    """Returns array of psoriasis probabilities, one per row."""
    from models.dataset import CLASS_NAMES
    ps_idx = CLASS_NAMES.index("psoriasis")
    probs = []
    model.eval()
    total = len(df)
    with torch.no_grad():
        for i, row in enumerate(df.itertuples(), 1):
            if i % 500 == 0 or i == total:
                print(f"  {i}/{total}", end="\r", flush=True)
            try:
                pil = Image.open(row.img_path).convert("RGB")
                x = _PREPROCESS(pil).unsqueeze(0).to(device)
                p = torch.softmax(model(x), dim=1).cpu()[0][ps_idx].item()
            except Exception:
                p = float("nan")
            probs.append(p)
    print()
    return np.array(probs)


# ── Metrics ───────────────────────────────────────────────────────────────────

def _binary_metrics(true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, confusion_matrix,
        f1_score, precision_score, recall_score, average_precision_score,
    )
    mask = ~np.isnan(prob)
    true, prob = true[mask], prob[mask]
    if len(true) == 0 or len(np.unique(true)) < 2:
        return dict(n=len(true), auc=float("nan"), ap=float("nan"),
                    accuracy=float("nan"), sensitivity=float("nan"),
                    specificity=float("nan"), ppv=float("nan"),
                    f1=float("nan"))

    pred = (prob >= threshold).astype(int)
    auc  = roc_auc_score(true, prob)
    ap   = average_precision_score(true, prob)
    acc  = accuracy_score(true, pred)
    cm   = confusion_matrix(true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall for psoriasis
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # recall for non-psoriasis
    ppv  = precision_score(true, pred, zero_division=0)
    f1   = f1_score(true, pred, zero_division=0)
    return dict(n=int(mask.sum()), auc=round(auc, 4), ap=round(ap, 4),
                accuracy=round(acc, 4), sensitivity=round(sens, 4),
                specificity=round(spec, 4), ppv=round(ppv, 4),
                f1=round(f1, 4))


# ── Summary output ────────────────────────────────────────────────────────────

def _write_summary(metrics_df: pd.DataFrame, per_scale_df: pd.DataFrame) -> None:
    lines = [
        "# External Validation — Fitzpatrick17k",
        "",
        "Independent evaluation on the Fitzpatrick17k dataset (Groh et al., 2021).",
        "These images were **not** used in training or internal test splits.",
        "",
        "**Binary task:** psoriasis / pustular psoriasis → positive; all other conditions → negative.",
        "",
        f"Total images evaluated: {metrics_df['n'].iloc[0]:,}",
        "",
    ]

    lines += ["## Overall Metrics", ""]
    lines += ["| Model | N | AUC | AP | Accuracy | Sensitivity | Specificity | PPV | F1 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for _, r in metrics_df.iterrows():
        lines.append(
            f"| {r['model']} | {r['n']:,} | {r['auc']:.4f} | {r['ap']:.4f} | "
            f"{r['accuracy']:.4f} | {r['sensitivity']:.4f} | {r['specificity']:.4f} | "
            f"{r['ppv']:.4f} | {r['f1']:.4f} |"
        )
    lines.append("")

    lines += [
        "## Fairness by Fitzpatrick Scale (Real Labels)",
        "",
        "> Unlike the ITA proxy used in training-set fairness analysis, these groups",
        "> use the **original Fitzpatrick scale ratings** from the dataset.",
        "> Sensitivity (psoriasis detection rate) is the primary fairness metric.",
        "",
    ]

    for model in per_scale_df["model"].unique():
        mdf = per_scale_df[per_scale_df["model"] == model]
        lines += [f"### {model}", ""]
        lines += ["| Fitzpatrick Type | N | Psoriasis N | AUC | Sensitivity | Specificity | Accuracy |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in mdf.iterrows():
            name = FITZ_LABELS.get(int(r["fitzpatrick_scale"]), str(r["fitzpatrick_scale"]))

            def fmt(v):
                return f"{v:.4f}" if not (isinstance(v, float) and np.isnan(v)) else "n/a"

            lines.append(
                f"| {name} | {r['n']:,} | {int(r['n_psoriasis'])} | "
                f"{fmt(r['auc'])} | {fmt(r['sensitivity'])} | "
                f"{fmt(r['specificity'])} | {fmt(r['accuracy'])} |"
            )
        # Compute fairness gap
        valid = mdf[(mdf["fitzpatrick_scale"] != -1) & (mdf["n_psoriasis"] >= 5) & ~mdf["sensitivity"].isna()]
        if len(valid) > 1:
            best = valid.loc[valid["sensitivity"].idxmax()]
            worst = valid.loc[valid["sensitivity"].idxmin()]
            gap = best["sensitivity"] - worst["sensitivity"]
            best_name = FITZ_LABELS.get(int(best["fitzpatrick_scale"]), "?")
            worst_name = FITZ_LABELS.get(int(worst["fitzpatrick_scale"]), "?")
            lines += [
                "",
                f"- **Best sensitivity:** {best_name} ({best['sensitivity']:.4f})",
                f"- **Worst sensitivity:** {worst_name} ({worst['sensitivity']:.4f})",
                f"- **Sensitivity gap:** {gap:.4f}",
            ]
        lines.append("")

    lines += [
        "## Dataset Notes",
        "",
        "- Source: Fitzpatrick17k (Groh et al., 2021) — 16,577 dermatology images, 114 conditions.",
        "- QC exclusions: 'Wrongly labelled' and 'Other' rows removed.",
        "- Fitzpatrick scale -1 indicates images without a skin-tone rating.",
        "- Psoriasis prevalence in this dataset (~4.3%) reflects real-world rarity,",
        "  which inflates apparent accuracy but suppresses sensitivity.",
    ]

    out = OUT_DIR / "fitzpatrick_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(versions: list[str], quick: bool = False) -> None:
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}")
        return
    if not IMG_DIR.exists():
        print(f"Image directory not found: {IMG_DIR}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset(quick=quick)

    pos = (df["binary_idx"] == 1).sum()
    neg = (df["binary_idx"] == 0).sum()
    print(f"Dataset: {len(df):,} images  ({pos} psoriasis, {neg} non-psoriasis)")
    print(f"Fitzpatrick scale distribution:")
    for scale, grp in df.groupby("fitzpatrick_scale"):
        ps = (grp["binary_idx"] == 1).sum()
        print(f"  Scale {scale:2d} ({FITZ_LABELS.get(scale,'?')[:20]:20s}): {len(grp):5d} imgs, {ps} psoriasis")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}\n")

    all_preds_rows = []
    overall_rows   = []
    per_scale_rows = []

    for ver in versions:
        name, _, _, ckpt = _VERSIONS[ver]
        if not ckpt.exists():
            print(f"[{ver}] checkpoint not found — skipping.")
            continue

        print(f"=== {ver}: {name} ===")
        try:
            model = _load_model(ver, device)
        except Exception as e:
            print(f"  Could not load: {e}")
            continue

        probs = run_inference(model, df, device)

        # Per-image prediction CSV
        for i, (idx, row) in enumerate(df.iterrows()):
            all_preds_rows.append({
                "model":             ver,
                "md5hash":           row["md5hash"],
                "original_label":    row["label"],
                "binary_label":      row["binary_label"],
                "fitzpatrick_scale": row["fitzpatrick_scale"],
                "psoriasis_prob":    round(float(probs[i]), 4) if not np.isnan(probs[i]) else None,
                "predicted":         "psoriasis" if probs[i] >= 0.5 else "non-psoriasis",
                "correct":           ((probs[i] >= 0.5) == row["binary_idx"]) if not np.isnan(probs[i]) else None,
            })

        # Overall metrics
        m = _binary_metrics(df["binary_idx"].values, probs)
        print(f"  Overall  — AUC={m['auc']:.4f}  sens={m['sensitivity']:.4f}  spec={m['specificity']:.4f}  acc={m['accuracy']:.4f}")
        overall_rows.append({"model": f"{ver} ({name})", **m})

        # Per Fitzpatrick scale (df is reset_index'd so index == iloc position)
        for scale, grp in df.groupby("fitzpatrick_scale"):
            sub_probs = probs[grp.index.values]
            sub_true  = grp["binary_idx"].values
            m_scale = _binary_metrics(sub_true, sub_probs)
            ps_n = int(sub_true.sum())
            fitz_name = FITZ_LABELS.get(int(scale), str(scale))
            print(f"  Scale {scale:2d} ({fitz_name[:18]:18s}) n={m_scale['n']:5d}  ps={ps_n:4d}  "
                  f"sens={m_scale['sensitivity'] if not np.isnan(m_scale['sensitivity']) else 'n/a'}")
            per_scale_rows.append({
                "model":             f"{ver} ({name})",
                "fitzpatrick_scale": int(scale),
                "scale_name":        fitz_name,
                "n":                 m_scale["n"],
                "n_psoriasis":       ps_n,
                **{k: v for k, v in m_scale.items() if k != "n"},
            })
        print()

    if not overall_rows:
        print("No results to save.")
        return

    metrics_df   = pd.DataFrame(overall_rows)
    per_scale_df = pd.DataFrame(per_scale_rows)
    preds_df     = pd.DataFrame(all_preds_rows)

    metrics_df.to_csv(OUT_DIR / "fitzpatrick_metrics.csv", index=False)
    per_scale_df.to_csv(OUT_DIR / "fitzpatrick_fairness_by_scale.csv", index=False)
    preds_df.to_csv(OUT_DIR / "fitzpatrick_predictions.csv", index=False)
    print(f"Saved CSVs to {OUT_DIR}/")

    _write_summary(metrics_df, per_scale_df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate models on Fitzpatrick17k.")
    parser.add_argument("--model", choices=list(_VERSIONS.keys()),
                        help="Single model version (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="Run on 500 images only (sanity check)")
    args = parser.parse_args()
    versions = [args.model] if args.model else list(_VERSIONS.keys())
    run(versions, quick=args.quick)


if __name__ == "__main__":
    main()
