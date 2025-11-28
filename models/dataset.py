from pathlib import Path

import pandas as pd
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


CLASS_NAMES = ["non-psoriasis", "psoriasis"]


def _build_transform(split: str):
    augments = []
    if split == "train":
        augments.append(T.RandomHorizontalFlip())
    augments.extend(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return T.Compose(augments)


class ManifestDataset(Dataset):
    def __init__(self, manifest_csv, split, img_root="data/raw", transform=None):
        self.df = pd.read_csv(manifest_csv)
        if "split" not in self.df.columns:
            raise ValueError("Manifest is missing a 'split' column. Run the pipeline first.")
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows found for split '{split}'.")
        self.img_root = Path(img_root)
        self.transform = transform or _build_transform(split)
        self.to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img_path = self.img_root / r["relative_path"]
        if not img_path.is_file():
            raise FileNotFoundError(f"Missing image: {img_path}")
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        y = self.to_idx[r["label"]]
        meta = {
            "skin_tone": r.get("skin_tone", "unknown"),
            "subtype": r.get("subtype", "unknown"),
            "patient_id": r.get("patient_id", ""),
        }
        return x, y, meta
