import sys
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.baseline import build_model
from models.dataset import ManifestDataset, CLASS_NAMES

MANIFEST = ROOT / "data/manifest.split.csv"
CHECKPOINT_PATH = ROOT / "models/checkpoints/resnet18_baseline.pt"


def loader(split, bs, workers=4):
    ds = ManifestDataset(MANIFEST, split=split, img_root=ROOT / "data/raw")
    return DataLoader(ds, batch_size=bs, shuffle=(split == "train"), num_workers=workers, persistent_workers=True)


def train_one_epoch(model, dataloader, device, optimizer, loss_fn):
    model.train()
    running = 0.0
    for x, y, _ in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        running += loss.item()
    return running / max(1, len(dataloader))


def evaluate(model, dataloader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y, _ in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
    return correct / total if total else 0.0


def main(epochs: int = 10, lr: float = 1e-4):
    if not MANIFEST.exists():
        raise FileNotFoundError("Run data/pipeline_from_manifest.py to create the split manifest.")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    bs = 32
    train_loader = loader("train", bs)
    try:
        val_loader = loader("val", bs)
    except ValueError:
        print("No 'val' split — falling back to 'test' for validation.")
        val_loader = loader("test", bs)

    # Pretrained weights give much better accuracy with limited data
    model = build_model(weights="IMAGENET1K_V1")
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, device, optimizer, loss_fn)
        val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch+1}/{epochs}  loss={loss:.4f}  val_acc={val_acc:.2%}", flush=True)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "classes": CLASS_NAMES}, CHECKPOINT_PATH)
            print(f"  -> saved new best ({best_val_acc:.2%})", flush=True)

    print(f"Best val_acc: {best_val_acc:.2%}. Weights at {CHECKPOINT_PATH}", flush=True)

if __name__ == "__main__":
    main()
