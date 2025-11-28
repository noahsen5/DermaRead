from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .baseline import build_model
from .dataset import ManifestDataset, CLASS_NAMES

MANIFEST = Path("data/manifest.split.csv")
CHECKPOINT_PATH = Path("models/checkpoints/resnet18_baseline.pt")


def loader(split, bs, workers=0):
    ds = ManifestDataset(MANIFEST, split=split, img_root="data/raw")
    return DataLoader(ds, batch_size=bs, shuffle=(split == "train"), num_workers=workers)


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


def main(epochs: int = 5, lr: float = 1e-3):
    if not MANIFEST.exists():
        raise FileNotFoundError("Run data/pipeline_from_manifest.py to create the split manifest.")

    train_loader = loader("train", 4)
    try:
        val_loader = loader("val", 4)
    except ValueError:
        print("No 'val' split present — falling back to 'test' for validation.")
        val_loader = loader("test", 4)

    model = build_model(weights=None)  # offline-friendly, we fine-tune ourselves
    device = torch.device("cpu")
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, device, optimizer, loss_fn)
        val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch+1}/{epochs} loss={loss:.4f} val_acc={val_acc:.2%}")

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": CLASS_NAMES}, CHECKPOINT_PATH)
    print("Saved weights to", CHECKPOINT_PATH)

if __name__ == "__main__":
    main()
