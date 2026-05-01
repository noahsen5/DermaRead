"""
Fine-tune ResNet50 (V2 checkpoint) on the external validation dataset — produces V4.

The external set contains real-world diverse non-psoriasis conditions
(acne, eczema, melanoma, scabies, etc.) that the original training set lacked.
Starting from V2 preserves the psoriasis features already learned.

Saves:
    models/checkpoints/resnet50_v4_external.pt
    outputs/v4_external_finetuned/training_log.csv
    outputs/v4_external_finetuned/confusion_matrix.png
    outputs/v4_external_finetuned/classification_report.csv
    outputs/results_summary.csv  (appended)

Usage:
    python models/finetune_external.py
"""

import csv
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch.manual_seed(42)
random.seed(42)

from models.dataset import CLASS_NAMES
from models.evaluate_model import run_full_evaluation, MANIFEST as INTERNAL_MANIFEST
from models.preprocessing import inference_transform, train_transform
from models.resnet50_model import DEFAULT_CHECKPOINT_V2, build_resnet50

EXT_DIR      = ROOT / "data/external_validation"
EXT_MANIFEST = EXT_DIR / "external_manifest.csv"
CHECKPOINT   = ROOT / "models/checkpoints/resnet50_v4_external.pt"
OUTPUT_DIR   = ROOT / "outputs/v4_external_finetuned"
LOG_PATH     = OUTPUT_DIR / "training_log.csv"

# Held-out test results path, separate from the internal test set
EXT_TEST_CSV = ROOT / "outputs/v4_external_finetuned/external_test_results.csv"


# ── Dataset ────────
class ExternalDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, img_root: Path, split: str = "train"):
        self.rows = rows.reset_index(drop=True)
        self.img_root = img_root
        self.transform = train_transform if split == "train" else inference_transform
        self.to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        img = Image.open(self.img_root / r["relative_path"]).convert("RGB")
        x = self.transform(img)
        y = self.to_idx[r["label"]]
        return x, y


def _split_manifest(df: pd.DataFrame, train_ratio: float = 0.8):
    train_rows, test_rows = [], []
    for label, grp in df.groupby("label"):
        idx = grp.sample(frac=1, random_state=42).index.tolist()
        n_train = int(len(idx) * train_ratio)
        train_rows.append(grp.loc[idx[:n_train]])
        test_rows.append(grp.loc[idx[n_train:]])
    return pd.concat(train_rows), pd.concat(test_rows)


def _class_weights(df: pd.DataFrame) -> torch.Tensor:
    counts = np.array([len(df[df["label"] == c]) for c in CLASS_NAMES], dtype=np.float64)
    total = counts.sum()
    w = total / (len(CLASS_NAMES) * np.maximum(counts, 1))
    for c, n, wt in zip(CLASS_NAMES, counts.astype(int), w):
        print(f"  {c}: n={n}, weight={wt:.4f}")
    return torch.tensor(w, dtype=torch.float32)


# ── Training ──────
def _get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")


def _train_epoch(model, loader, device, optimizer, loss_fn) -> float:
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))


def _eval_accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total if total else 0.0


def _eval_per_class(model, loader, device) -> dict:
    """Return per-class accuracy for the two classes."""
    model.eval()
    counts = {c: {"correct": 0, "total": 0} for c in CLASS_NAMES}
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).argmax(1).cpu()
            for idx, (pred, true) in enumerate(zip(preds, y)):
                cls = CLASS_NAMES[int(true)]
                counts[cls]["total"] += 1
                if pred == true:
                    counts[cls]["correct"] += 1
    return {c: v["correct"] / max(1, v["total"]) for c, v in counts.items()}


# ── External test evaluation ───────
def _eval_external_test(model, test_df: pd.DataFrame, device: str) -> None:
    """Predict on external held-out test rows and save to CSV."""
    from models.resnet50_model import predict_pil
    rows = []
    correct = 0
    for _, row in test_df.iterrows():
        img_path = EXT_DIR / row["relative_path"]
        if not img_path.is_file():
            continue
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        probs = predict_pil(model, pil)
        pred = max(probs, key=probs.get)
        ok = pred == row["label"]
        if ok:
            correct += 1
        rows.append({
            "id": row["id"],
            "true_label": row["label"],
            "predicted_label": pred,
            "confidence": round(probs[pred], 4),
            "psoriasis_probability": round(probs["psoriasis"], 4),
            "non_psoriasis_probability": round(probs["non-psoriasis"], 4),
            "correct": ok,
            "source_dataset": row.get("source_dataset", ""),
            "image_type": row.get("image_type", ""),
            "skin_tone_notes": row.get("skin_tone_notes", ""),
        })
    n = len(rows)
    if n:
        pd.DataFrame(rows).to_csv(EXT_TEST_CSV, index=False)
        print(f"\nExternal held-out test: {correct}/{n} = {correct/n:.0%}")
        # Per-class breakdown
        rdf = pd.DataFrame(rows)
        for lbl, grp in rdf.groupby("true_label"):
            acc = grp["correct"].mean()
            print(f"  {lbl:20s}: {acc:.0%} ({int(grp['correct'].sum())}/{len(grp)})")
        print(f"Saved: {EXT_TEST_CSV}")


# ── Main ──────────
def main(epochs: int = 15, lr: float = 5e-6, batch_size: int = 16) -> None:
    if not EXT_MANIFEST.exists():
        raise FileNotFoundError("external_manifest.csv not found.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = _get_device()
    print(f"Device: {device}")

    df = pd.read_csv(EXT_MANIFEST)
    df = df[df["label"].isin(CLASS_NAMES)].copy()

    # 80/20 split — stratified by label, held-out test never seen during training
    train_df, test_df = _split_manifest(df, train_ratio=0.80)
    print(f"\nExternal split: {len(train_df)} train, {len(test_df)} test")
    print("Class weights (inverse-frequency):")
    weights = _class_weights(train_df).to(device)

    train_ds  = ExternalDataset(train_df, EXT_DIR, split="train")
    train_ldr = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)

    val_ds    = ExternalDataset(test_df, EXT_DIR, split="val")
    val_ldr   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # Start from V2 checkpoint (already knows psoriasis features)
    if DEFAULT_CHECKPOINT_V2.exists():
        print(f"\nLoading V2 checkpoint: {DEFAULT_CHECKPOINT_V2}")
        model = build_resnet50(weights=None, checkpoint_path=DEFAULT_CHECKPOINT_V2,
                               device=str(device))
    else:
        print("\nV2 checkpoint not found — starting from ImageNet weights")
        model = build_resnet50(weights="IMAGENET1K_V2")
    model = model.to(device)

    loss_fn   = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"\nFine-tuning for {epochs} epochs (lr={lr})...")
    log_rows = []
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(model, train_ldr, device, optimizer, loss_fn)
        val_acc    = _eval_accuracy(model, val_ldr, device)
        per_cls    = _eval_per_class(model, val_ldr, device)

        log_rows.append({"epoch": epoch, "train_loss": round(train_loss, 5),
                         "val_acc": round(val_acc, 5),
                         "val_psoriasis": round(per_cls["psoriasis"], 4),
                         "val_non_psoriasis": round(per_cls["non-psoriasis"], 4)})

        print(f"Epoch {epoch:>2}/{epochs}  loss={train_loss:.4f}  "
              f"acc={val_acc:.2%}  "
              f"p(pso)={per_cls['psoriasis']:.0%}  "
              f"p(nps)={per_cls['non-psoriasis']:.0%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "classes": CLASS_NAMES}, CHECKPOINT)
            print(f"  -> Saved best ({best_val_acc:.2%})")

    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\nTraining log: {LOG_PATH}")

    # Load best and run external test
    model = build_resnet50(weights=None, checkpoint_path=CHECKPOINT, device=str(device))
    _eval_external_test(model, test_df, str(device))

    # Also run on internal test split if manifest exists
    if INTERNAL_MANIFEST.exists():
        print("\nRunning on internal test split for comparison...")
        run_full_evaluation(
            model, "v4", "ResNet50 Fine-tuned (External Data)",
            OUTPUT_DIR, device,
            notes="Fine-tuned from V2 on external diverse-condition dataset",
        )

    print(f"\nDone. Checkpoint: {CHECKPOINT}")


if __name__ == "__main__":
    main()
