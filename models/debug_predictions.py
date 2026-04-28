"""
Debug prediction pipeline for DermaRead.

Checks:
  1. CLASS_NAMES consistency across all modules
  2. Checkpoint 'classes' field vs. runtime mapping
  3. Predictions on 10 known psoriasis + 10 known non-psoriasis images (test split)

Saves:
  outputs/class_mapping.json
  outputs/debug_predictions.csv
  outputs/external_image_tests.csv  (template for manual external testing)

Usage:
  python models/debug_predictions.py
  python models/debug_predictions.py --model v2
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

from models.dataset import CLASS_NAMES as DS_CLASS_NAMES
from models.baseline import CLASS_NAMES as BL_CLASS_NAMES
from models.resnet50_model import CLASS_NAMES as R50_CLASS_NAMES
from models.preprocessing import inference_transform, RESIZE_SIZE, INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD

OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)
MANIFEST = ROOT / "data/manifest.split.csv"


# ── Class mapping ──────────────────────────────────────────────────────────────

def check_and_save_class_mapping() -> dict:
    """Verify CLASS_NAMES is consistent everywhere and save to JSON."""
    sources = {
        "models/baseline.py":      BL_CLASS_NAMES,
        "models/resnet50_model.py": R50_CLASS_NAMES,
        "models/dataset.py":        DS_CLASS_NAMES,
    }

    label_to_idx = {c: i for i, c in enumerate(DS_CLASS_NAMES)}
    idx_to_label = {i: c for i, c in enumerate(DS_CLASS_NAMES)}

    mapping = {
        "class_names":   DS_CLASS_NAMES,
        "label_to_index": label_to_idx,
        "index_to_label": {str(k): v for k, v in idx_to_label.items()},
        "preprocessing": {
            "resize":    RESIZE_SIZE,
            "crop":      INPUT_SIZE,
            "mean":      IMAGENET_MEAN,
            "std":       IMAGENET_STD,
        },
        "consistency_check": {},
    }

    all_ok = True
    for src, names in sources.items():
        match = names == DS_CLASS_NAMES
        mapping["consistency_check"][src] = "OK" if match else f"MISMATCH: {names}"
        if not match:
            all_ok = False

    # Check checkpoint 'classes' fields
    ckpt_results = {}
    for ver, fname in [("v1", "resnet18_baseline.pt"),
                       ("v2", "resnet50_v2.pt"),
                       ("v3", "resnet50_v3_balanced.pt")]:
        p = ROOT / "models/checkpoints" / fname
        if p.exists():
            d = torch.load(p, map_location="cpu", weights_only=False)
            saved_classes = d.get("classes", "NOT_SAVED")
            match = saved_classes == DS_CLASS_NAMES
            ckpt_results[ver] = "OK" if match else f"MISMATCH: {saved_classes}"
            if not match:
                all_ok = False
        else:
            ckpt_results[ver] = "CHECKPOINT_NOT_FOUND"
    mapping["checkpoint_classes"] = ckpt_results
    mapping["all_consistent"] = all_ok

    out = OUTPUTS / "class_mapping.json"
    out.write_text(json.dumps(mapping, indent=2))
    print(f"Saved: {out}")

    if all_ok:
        print("  Class mapping: ALL CONSISTENT")
    else:
        print("  WARNING: Class mapping inconsistencies detected — see class_mapping.json")

    return mapping


# ── Model loader ───────────────────────────────────────────────────────────────

def _load_model(version: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if version == "v1":
        from models.baseline import build_model, DEFAULT_CHECKPOINT
        return build_model(weights=None, checkpoint_path=DEFAULT_CHECKPOINT, device=device), device
    if version == "v2":
        from models.resnet50_model import build_resnet50, DEFAULT_CHECKPOINT_V2
        return build_resnet50(weights=None, checkpoint_path=DEFAULT_CHECKPOINT_V2, device=device), device
    if version == "v3":
        from models.resnet50_model import build_resnet50, DEFAULT_CHECKPOINT_V3
        return build_resnet50(weights=None, checkpoint_path=DEFAULT_CHECKPOINT_V3, device=device), device
    raise ValueError(f"Unknown version: {version}")


# ── Prediction ─────────────────────────────────────────────────────────────────

def _predict(model, img_path: Path, device: str) -> dict:
    from PIL import Image
    pil = Image.open(img_path).convert("RGB")
    x = inference_transform(pil).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu()[0].tolist()
    pred_idx = int(probs.index(max(probs)))
    pred_label = DS_CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx]
    return {
        "predicted_label":          pred_label,
        "confidence":               round(confidence, 6),
        "psoriasis_probability":    round(probs[DS_CLASS_NAMES.index("psoriasis")], 6),
        "non_psoriasis_probability": round(probs[DS_CLASS_NAMES.index("non-psoriasis")], 6),
    }


# ── Dataset debug run ──────────────────────────────────────────────────────────

def run_dataset_debug(version: str = "v1") -> pd.DataFrame:
    """Test 10 psoriasis + 10 non-psoriasis images from the test split."""
    if not MANIFEST.exists():
        raise FileNotFoundError("Run data/pipeline_from_manifest.py first.")

    model, device = _load_model(version)
    df = pd.read_csv(MANIFEST)
    test = df[df["split"] == "test"]

    samples = pd.concat([
        test[test["label"] == "psoriasis"].head(10),
        test[test["label"] == "non-psoriasis"].head(10),
    ]).reset_index(drop=True)

    rows = []
    correct = 0
    print(f"\n=== Dataset debug [{version}] — 20 test-split images ===")
    for _, row in samples.iterrows():
        img_path = ROOT / "data/raw" / row["relative_path"]
        if not img_path.is_file():
            print(f"  MISSING: {img_path}")
            continue
        result = _predict(model, img_path, device)
        ok = result["predicted_label"] == row["label"]
        if ok:
            correct += 1
        status = "OK" if ok else "WRONG"
        print(
            f"  [{status}] true={row['label']:20s}  "
            f"pred={result['predicted_label']:20s}  "
            f"conf={result['confidence']:.4f}  "
            f"p(pso)={result['psoriasis_probability']:.4f}"
        )
        rows.append({
            "model_version":            version,
            "image_path":               str(img_path),
            "true_label":               row["label"],
            "predicted_label":          result["predicted_label"],
            "confidence":               result["confidence"],
            "psoriasis_probability":    result["psoriasis_probability"],
            "non_psoriasis_probability": result["non_psoriasis_probability"],
            "correct":                  ok,
            "split":                    row["split"],
        })

    print(f"\n  Accuracy on 20 samples: {correct}/20 = {correct/20:.0%}")
    return pd.DataFrame(rows)


# ── External image tests template ──────────────────────────────────────────────

def create_external_tests_template() -> None:
    """
    Create a CSV template where you can log predictions on external (out-of-distribution) images.
    Fill in image_path and true_label, then re-run with --external to populate predictions.
    """
    out = OUTPUTS / "external_image_tests.csv"
    if out.exists():
        print(f"  external_image_tests.csv already exists — not overwriting.")
        return

    template = pd.DataFrame([
        {
            "image_path": "C:/path/to/your/psoriasis_photo.jpg",
            "true_label": "psoriasis",
            "source": "smartphone_photo",
            "notes": "Clinical photo taken under natural light",
            "model_version": "",
            "predicted_label": "",
            "confidence": "",
            "psoriasis_probability": "",
            "non_psoriasis_probability": "",
            "ood_suspected": "",
        },
        {
            "image_path": "C:/path/to/your/non_psoriasis_photo.jpg",
            "true_label": "non-psoriasis",
            "source": "smartphone_photo",
            "notes": "Normal skin under indoor lighting",
            "model_version": "",
            "predicted_label": "",
            "confidence": "",
            "psoriasis_probability": "",
            "non_psoriasis_probability": "",
            "ood_suspected": "",
        },
    ])
    template.to_csv(out, index=False)
    print(f"Saved template: {out}")


def run_external_tests(version: str = "v1") -> None:
    """Run predictions on any rows in external_image_tests.csv that have a path but no prediction."""
    csv_path = OUTPUTS / "external_image_tests.csv"
    if not csv_path.exists():
        print("external_image_tests.csv not found. Run without --external first to create it.")
        return

    model, device = _load_model(version)
    df = pd.read_csv(csv_path)

    changed = False
    for idx, row in df.iterrows():
        p = Path(str(row["image_path"]))
        if not p.is_file():
            print(f"  MISSING: {p}")
            continue
        if pd.notna(row.get("predicted_label")) and str(row.get("predicted_label")).strip():
            continue  # already predicted
        result = _predict(model, p, device)
        df.at[idx, "model_version"]            = version
        df.at[idx, "predicted_label"]          = result["predicted_label"]
        df.at[idx, "confidence"]               = result["confidence"]
        df.at[idx, "psoriasis_probability"]    = result["psoriasis_probability"]
        df.at[idx, "non_psoriasis_probability"] = result["non_psoriasis_probability"]
        true = str(row.get("true_label", ""))
        df.at[idx, "ood_suspected"] = (
            result["confidence"] > 0.95 and result["predicted_label"] != true
        ) if true else ""
        print(
            f"  {p.name}: pred={result['predicted_label']}  "
            f"conf={result['confidence']:.4f}  "
            f"p(pso)={result['psoriasis_probability']:.4f}"
        )
        changed = True

    if changed:
        df.to_csv(csv_path, index=False)
        print(f"Updated: {csv_path}")
    else:
        print("No new rows to predict.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Debug DermaRead prediction pipeline.")
    parser.add_argument("--model", choices=["v1", "v2", "v3"], default="v1",
                        help="Model version to debug (default: v1).")
    parser.add_argument("--all-models", action="store_true",
                        help="Run debug on all available model versions.")
    parser.add_argument("--external", action="store_true",
                        help="Run predictions on external_image_tests.csv.")
    args = parser.parse_args()

    print("=== DermaRead Debug ===\n")

    # 1. Class mapping
    check_and_save_class_mapping()

    # 2. External image test template
    create_external_tests_template()

    if args.external:
        run_external_tests(args.model)
        return

    # 3. Dataset predictions
    versions = ["v1", "v2", "v3"] if args.all_models else [args.model]
    all_rows = []
    for ver in versions:
        try:
            rows = run_dataset_debug(ver)
            all_rows.append(rows)
        except FileNotFoundError as e:
            print(f"  [{ver}] Skipping: {e}")

    if all_rows:
        out_df = pd.concat(all_rows, ignore_index=True)
        out_csv = OUTPUTS / "debug_predictions.csv"
        out_df.to_csv(out_csv, index=False)
        print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
