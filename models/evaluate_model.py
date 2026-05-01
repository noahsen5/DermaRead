"""
Evaluate a trained model on the test split and export dissertation-ready outputs.

Outputs per model version:
    <output_dir>/confusion_matrix.png
    <output_dir>/classification_report.csv
    outputs/results_summary.csv  (appended)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.dataset import ManifestDataset, CLASS_NAMES

MANIFEST = ROOT / "data/manifest.split.csv"
RESULTS_SUMMARY = ROOT / "outputs/results_summary.csv"
_SUMMARY_FIELDS = [
    "model_version", "model_name", "dataset_size",
    "accuracy", "precision", "recall", "f1", "auc", "notes",
]


# ── Inference ────────────────
def collect_predictions(model, split: str, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model on a split; return (true_labels, predicted_labels, pos_probs)."""
    ds = ManifestDataset(MANIFEST, split=split, img_root=ROOT / "data/raw")
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    all_labels, all_preds, all_probs = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_labels.extend(y.numpy())
            all_preds.extend(np.argmax(probs, axis=1))
            all_probs.extend(probs[:, 1])  # probability of positive class (psoriasis)

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        f1_score, precision_score, recall_score, roc_auc_score,
    )

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="weighted", zero_division=0)
    rec = recall_score(labels, preds, average="weighted", zero_division=0)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=CLASS_NAMES, output_dict=True)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "auc": auc, "confusion_matrix": cm, "classification_report": report}


# ── Exports ───────────────────────────────────────────────────────────────────

def _save_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ticks = range(len(CLASS_NAMES))
    ax.set_xticks(list(ticks))
    ax.set_yticks(list(ticks))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _save_classification_report(report: dict, output_path: Path) -> None:
    rows = []
    for cls in CLASS_NAMES:
        r = report.get(cls, {})
        rows.append({"class": cls, "precision": r.get("precision", 0),
                     "recall": r.get("recall", 0), "f1_score": r.get("f1-score", 0),
                     "support": r.get("support", 0)})
    wa = report.get("weighted avg", {})
    rows.append({"class": "weighted_avg", "precision": wa.get("precision", 0),
                 "recall": wa.get("recall", 0), "f1_score": wa.get("f1-score", 0),
                 "support": wa.get("support", 0)})
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _append_results_summary(row: dict) -> None:
    RESULTS_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_SUMMARY.exists()

    # Remove duplicate row for same model_version before appending
    if RESULTS_SUMMARY.exists():
        existing = pd.read_csv(RESULTS_SUMMARY)
        existing = existing[existing["model_version"] != row["model_version"]]
        existing.to_csv(RESULTS_SUMMARY, index=False)
        write_header = False

    with open(RESULTS_SUMMARY, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ── Public API ────────────────────────────────────────────────────────────────

def run_full_evaluation(
    model,
    model_version: str,
    model_name: str,
    output_dir: Path,
    device,
    dataset_size: int | None = None,
    notes: str = "",
) -> dict:
    """
    Evaluate model on the test split, save all artefacts, and update results_summary.csv.
    Returns the summary dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels, preds, probs = collect_predictions(model, "test", device)
    metrics = compute_metrics(labels, preds, probs)

    _save_confusion_matrix(metrics["confusion_matrix"], output_dir / "confusion_matrix.png")
    _save_classification_report(metrics["classification_report"], output_dir / "classification_report.csv")

    auc_str = f"{metrics['auc']:.4f}" if not np.isnan(metrics["auc"]) else "N/A"
    summary = {
        "model_version": model_version,
        "model_name": model_name,
        "dataset_size": dataset_size if dataset_size is not None else int(len(labels)),
        "accuracy": round(metrics["accuracy"], 4),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "f1": round(metrics["f1"], 4),
        "auc": auc_str,
        "notes": notes,
    }
    _append_results_summary(summary)

    print(
        f"[{model_version}] acc={metrics['accuracy']:.2%}  "
        f"f1={metrics['f1']:.4f}  auc={auc_str}"
    )
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a trained model checkpoint.")
    parser.add_argument("--model", choices=["v1", "v2", "v3"], required=True,
                        help="Model version to evaluate.")
    parser.add_argument("--checkpoint", default=None,
                        help="Override checkpoint path.")
    args = parser.parse_args()

    device = _get_device()

    if args.model == "v1":
        from models.baseline import DEFAULT_CHECKPOINT, build_model
        ckpt = args.checkpoint or DEFAULT_CHECKPOINT
        model = build_model(weights=None, checkpoint_path=ckpt, device=str(device))
        run_full_evaluation(model, "v1", "ResNet18 Baseline",
                            ROOT / "outputs/v1_resnet18_baseline", device)

    elif args.model == "v2":
        from models.resnet50_model import DEFAULT_CHECKPOINT_V2, build_resnet50
        ckpt = args.checkpoint or DEFAULT_CHECKPOINT_V2
        model = build_resnet50(weights=None, checkpoint_path=ckpt, device=str(device))
        run_full_evaluation(model, "v2", "ResNet50 Transfer",
                            ROOT / "outputs/v2_resnet50", device)

    elif args.model == "v3":
        from models.resnet50_model import DEFAULT_CHECKPOINT_V3, build_resnet50
        ckpt = args.checkpoint or DEFAULT_CHECKPOINT_V3
        model = build_resnet50(weights=None, checkpoint_path=ckpt, device=str(device))
        run_full_evaluation(model, "v3", "ResNet50 Balanced (Weighted Loss)",
                            ROOT / "outputs/v3_resnet50_balanced", device,
                            notes="Class-weighted CrossEntropyLoss")


if __name__ == "__main__":
    main()
