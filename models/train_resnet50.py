"""
Train ResNet50 transfer learning — V2.

Uses the same train/val/test split as V1.
Two-phase training:
  Phase 1 (5 epochs): classifier head only, high LR — fast convergence.
  Phase 2 (epochs):   full fine-tuning, low LR — quality refinement.

Saves:
    models/checkpoints/resnet50_v2.pt
    outputs/v2_resnet50/training_log.csv
    outputs/v2_resnet50/confusion_matrix.png
    outputs/v2_resnet50/classification_report.csv
    outputs/results_summary.csv  (appended)

Usage:
    python models/train_resnet50.py
    python models/train_resnet50.py --epochs 20 --lr 5e-5
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch.manual_seed(42)

from models.dataset import ManifestDataset
from models.evaluate_model import run_full_evaluation
from models.resnet50_model import CLASS_NAMES, DEFAULT_CHECKPOINT_V2, build_resnet50

MANIFEST = ROOT / "data/manifest.split.csv"
OUTPUT_DIR = ROOT / "outputs/v2_resnet50"
LOG_PATH = OUTPUT_DIR / "training_log.csv"


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _make_loader(split: str, batch_size: int) -> DataLoader:
    ds = ManifestDataset(MANIFEST, split=split, img_root=ROOT / "data/raw")
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=0)


def _train_epoch(model, loader, device, optimizer, loss_fn) -> float:
    model.train()
    total = 0.0
    for x, y, _ in loader:
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
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total if total else 0.0


def main(epochs: int = 15, lr: float = 1e-4, batch_size: int = 32) -> None:
    if not MANIFEST.exists():
        raise FileNotFoundError(
            "manifest.split.csv not found. Run:  python data/pipeline_from_manifest.py"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = _get_device()
    print(f"Device: {device}")

    train_loader = _make_loader("train", batch_size)
    try:
        val_loader = _make_loader("val", batch_size)
    except ValueError:
        print("No 'val' split — using 'test' for validation.")
        val_loader = _make_loader("test", batch_size)

    model = build_resnet50(weights="IMAGENET1K_V2").to(device)
    loss_fn = nn.CrossEntropyLoss()

    # Phase 1 , warm up head only
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    optimizer = optim.Adam(model.fc.parameters(), lr=lr * 10, weight_decay=1e-4)

    print("Phase 1: Training classifier head (5 epochs)...")
    for epoch in range(1, 6):
        loss = _train_epoch(model, train_loader, device, optimizer, loss_fn)
        val_acc = _eval_accuracy(model, val_loader, device)
        print(f"  Phase1 epoch {epoch}/5  loss={loss:.4f}  val_acc={val_acc:.2%}")

    # Phase 2 , fine-tune full network with L2 regularisation to prevent loss→0
    for p in model.parameters():
        p.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"\nPhase 2: Fine-tuning full model ({epochs} epochs)...")
    log_rows = []
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(model, train_loader, device, optimizer, loss_fn)
        val_acc = _eval_accuracy(model, val_loader, device)
        log_rows.append({"epoch": epoch,
                         "train_loss": round(train_loss, 5),
                         "val_acc": round(val_acc, 5)})
        print(f"Epoch {epoch:>2}/{epochs}  loss={train_loss:.4f}  val_acc={val_acc:.2%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            DEFAULT_CHECKPOINT_V2.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "classes": CLASS_NAMES},
                       DEFAULT_CHECKPOINT_V2)
            print(f"  -> Saved best ({best_val_acc:.2%})")

    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"Training log: {LOG_PATH}")

    print("\nRunning final evaluation on test split...")
    model = build_resnet50(weights=None, checkpoint_path=DEFAULT_CHECKPOINT_V2, device=str(device))
    train_size = len(ManifestDataset(MANIFEST, split="train", img_root=ROOT / "data/raw"))
    run_full_evaluation(model, "v2", "ResNet50 Transfer", OUTPUT_DIR, device,
                        dataset_size=train_size)
    print(f"\nOutputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    main(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
