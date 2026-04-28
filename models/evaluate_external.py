"""
Evaluate all trained models against the external validation set.

The external set lives in data/external_validation/ and uses images
from different sources to test real-world generalisation.

Run AFTER adding images and filling in external_manifest.csv.

Usage:
    python models/evaluate_external.py
    python models/evaluate_external.py --model v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXT_DIR      = ROOT / "data/external_validation"
EXT_MANIFEST = EXT_DIR / "external_manifest.csv"
OUT_DIR      = ROOT / "outputs/external_validation"

from models.preprocessing import inference_transform
from models.dataset import CLASS_NAMES


def _load_model(version: str, device: str):
    if version == "v1":
        from models.baseline import build_model, DEFAULT_CHECKPOINT
        return build_model(weights=None, checkpoint_path=DEFAULT_CHECKPOINT, device=device)
    if version == "v2":
        from models.resnet50_model import build_resnet50, DEFAULT_CHECKPOINT_V2
        return build_resnet50(weights=None, checkpoint_path=DEFAULT_CHECKPOINT_V2, device=device)
    if version == "v3":
        from models.resnet50_model import build_resnet50, DEFAULT_CHECKPOINT_V3
        return build_resnet50(weights=None, checkpoint_path=DEFAULT_CHECKPOINT_V3, device=device)
    raise ValueError(f"Unknown version: {version}")


def _predict(model, img_path: Path, device: str) -> dict:
    from PIL import Image
    pil = Image.open(img_path).convert("RGB")
    x = inference_transform(pil).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu()[0].tolist()
    pred_idx  = int(probs.index(max(probs)))
    pred_label = CLASS_NAMES[pred_idx]
    return {
        "predicted_label":           pred_label,
        "confidence":                round(probs[pred_idx], 4),
        "psoriasis_probability":     round(probs[CLASS_NAMES.index("psoriasis")], 4),
        "non_psoriasis_probability": round(probs[CLASS_NAMES.index("non-psoriasis")], 4),
    }


def run(versions: list[str]) -> None:
    if not EXT_MANIFEST.exists() or EXT_MANIFEST.stat().st_size < 10:
        print("external_manifest.csv is empty. Add images and fill in the manifest first.")
        print("See data/external_validation/ for the required format.")
        return

    df = pd.read_csv(EXT_MANIFEST)
    if df.empty:
        print("No rows in external_manifest.csv yet.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for ver in versions:
        try:
            model = _load_model(ver, device)
        except FileNotFoundError as e:
            print(f"[{ver}] Checkpoint not found — skipping: {e}")
            continue

        print(f"\n=== External validation [{ver}] ({len(df)} images) ===")
        correct = 0
        rows = []

        for _, row in df.iterrows():
            img_path = EXT_DIR / row["relative_path"]
            if not img_path.is_file():
                print(f"  MISSING: {img_path}")
                continue

            result = _predict(model, img_path, device)
            ok = result["predicted_label"] == row["label"]
            if ok:
                correct += 1

            status = "OK   " if ok else "WRONG"
            print(
                f"  [{status}] true={row['label']:20s} "
                f"pred={result['predicted_label']:20s} "
                f"conf={result['confidence']:.3f}  "
                f"src={row.get('source_dataset','?')}"
            )
            rows.append({
                "model_version":             ver,
                "id":                        row["id"],
                "image_path":                str(img_path),
                "true_label":                row["label"],
                "source_dataset":            row.get("source_dataset", ""),
                "image_type":                row.get("image_type", ""),
                "skin_tone_notes":           row.get("skin_tone_notes", ""),
                "predicted_label":           result["predicted_label"],
                "confidence":                result["confidence"],
                "psoriasis_probability":     result["psoriasis_probability"],
                "non_psoriasis_probability": result["non_psoriasis_probability"],
                "correct":                   ok,
            })

        n = len(rows)
        if n:
            acc = correct / n
            print(f"\n  Accuracy: {correct}/{n} = {acc:.0%}")
        all_results.extend(rows)

    if all_results:
        out_df = pd.DataFrame(all_results)
        out_csv = OUT_DIR / "external_results.csv"
        out_df.to_csv(out_csv, index=False)
        print(f"\nSaved: {out_csv}")
        _write_summary(out_df)


def _write_summary(df: pd.DataFrame) -> None:
    lines = [
        "# External Validation Results",
        "",
        "Models evaluated against images from external sources (not in training data).",
        "",
    ]

    for ver in df["model_version"].unique():
        vdf = df[df["model_version"] == ver]
        acc = vdf["correct"].mean()
        lines += [
            f"## {ver}",
            f"- Total images: {len(vdf)}",
            f"- Accuracy: {acc:.0%}",
            f"- Correct: {int(vdf['correct'].sum())}/{len(vdf)}",
            "",
        ]

        # Per source breakdown
        if "source_dataset" in vdf.columns:
            lines.append("### By source dataset")
            lines.append("| Source | N | Accuracy |")
            lines.append("|---|---|---|")
            for src, grp in vdf.groupby("source_dataset"):
                lines.append(f"| {src} | {len(grp)} | {grp['correct'].mean():.0%} |")
            lines.append("")

        # Per class breakdown
        lines.append("### By true label")
        lines.append("| Label | N | Accuracy |")
        lines.append("|---|---|---|")
        for lbl, grp in vdf.groupby("true_label"):
            lines.append(f"| {lbl} | {len(grp)} | {grp['correct'].mean():.0%} |")
        lines.append("")

    out = OUT_DIR / "external_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["v1", "v2", "v3"], default=None,
                        help="Single model version. Omit to run all.")
    args = parser.parse_args()
    versions = [args.model] if args.model else ["v1", "v2", "v3"]
    run(versions)


if __name__ == "__main__":
    main()
