"""Generate a tiny synthetic dataset + manifest for local experiments.

The goal is to keep the repository runnable without shipping sensitive
imagery.  We synthesize simple colored squares and populate
``data/manifest.csv`` with realistic-looking metadata so the downstream
pipeline (database mirror, splitting, training, etc.) has something to
work with.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw

RAW_ROOT = Path("data/raw")
MANIFEST = Path("data/manifest.csv")


PATIENTS = [
    {"patient_id": "np001", "label": "non-psoriasis", "skin_tone": "II", "subtype": "healthy"},
    {"patient_id": "np002", "label": "non-psoriasis", "skin_tone": "III", "subtype": "healthy"},
    {"patient_id": "ps001", "label": "psoriasis", "skin_tone": "IV", "subtype": "plaque"},
    {"patient_id": "ps002", "label": "psoriasis", "skin_tone": "V", "subtype": "guttate"},
]


def ensure_dirs():
    for lbl in {p["label"] for p in PATIENTS}:
        (RAW_ROOT / lbl.replace("-", "_")).mkdir(parents=True, exist_ok=True)


def synth_image(label: str) -> Image.Image:
    rng = random.Random(label)
    base_color = {
        "non-psoriasis": (200, 150, 120),
        "psoriasis": (180, 40, 40),
    }.get(label, (128, 128, 128))
    w = h = 256
    img = Image.new("RGB", (w, h), base_color)
    draw = ImageDraw.Draw(img)
    for _ in range(50):
        bbox = [rng.randint(0, w), rng.randint(0, h), rng.randint(0, w), rng.randint(0, h)]
        bbox = [min(bbox[0], bbox[2]), min(bbox[1], bbox[3]), max(bbox[0], bbox[2]), max(bbox[1], bbox[3])]
        color = tuple(min(255, max(0, c + rng.randint(-30, 30))) for c in base_color)
        draw.ellipse(bbox, outline=color, width=3)
    return img


def main():
    ensure_dirs()
    rows = []
    img_idx = 1
    for pt in PATIENTS:
        for i in range(2):
            label_dir = pt["label"].replace("-", "_")
            rel_path = f"{label_dir}/{pt['patient_id']}_img{i+1}.png"
            img_path = RAW_ROOT / rel_path
            img = synth_image(pt["label"]).rotate(random.randint(-5, 5))
            img.save(img_path)
            rows.append(
                {
                    "id": f"{pt['patient_id']}_{i+1:02d}",
                    "patient_id": pt["patient_id"],
                    "relative_path": rel_path,
                    "label": pt["label"],
                    "subtype": pt["subtype"],
                    "skin_tone": pt["skin_tone"],
                    "source": "synthetic",
                    "license_url": "https://creativecommons.org/licenses/by/4.0/",
                    "consent": "research_use",
                    "split": "",
                }
            )
            img_idx += 1

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} synthetic rows to {MANIFEST}")


if __name__ == "__main__":
    main()
