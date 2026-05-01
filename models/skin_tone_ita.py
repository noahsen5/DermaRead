"""
ITA-based estimated skin-tone proxy.

ITA (Individual Typology Angle) is defined as:
    ITA = arctan((L* - 50) / b*) × (180/π)

where L* and b* are CIELAB colour coordinates (D65 illuminant).

IMPORTANT LIMITATIONS — read before interpreting results:
- ITA is NOT equivalent to Fitzpatrick phototype scale.
- Lesion colour (e.g. red plaques, silvery scale in psoriasis) strongly biases ITA.
- Image lighting, camera white-balance, JPEG compression, and background pixels
  all affect the estimate.
- The three-group scheme (Light / Medium / Dark) is a coarse approximation.
- All results must be labelled "estimated skin-tone proxy" — not Fitzpatrick scale.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── sRGB → CIELAB (D65) ────────
_D65 = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)

_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lab_f(t: np.ndarray) -> np.ndarray:
    delta = 6.0 / 29.0
    return np.where(t > delta ** 3, np.cbrt(t), t / (3.0 * delta ** 2) + 4.0 / 29.0)


def rgb_to_lab(rgb_uint8: np.ndarray) -> np.ndarray:
    """Convert (H, W, 3) uint8 RGB → (H, W, 3) float64 CIELAB."""
    lin = _srgb_to_linear(rgb_uint8.astype(np.float64) / 255.0)
    xyz = lin @ _RGB_TO_XYZ.T
    f = _lab_f(xyz / _D65)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


# ── Skin-pixel detection ─────
def _skin_mask(rgb: np.ndarray) -> np.ndarray:
    """
    Loose HSV-based skin-pixel detector covering light to dark skin tones.
    Returns a boolean mask; may include lesion pixels — treat as approximate.
    """
    f = rgb.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]

    maxc = np.maximum.reduce([r, g, b])
    minc = np.minimum.reduce([r, g, b])
    diff = maxc - minc + 1e-8
    sat = np.where(maxc > 0, (maxc - minc) / maxc, 0.0)
    val = maxc

    hue = np.where(
        maxc == r, (g - b) / diff % 6,
        np.where(maxc == g, (b - r) / diff + 2.0, (r - g) / diff + 4.0),
    ) / 6.0

    # Skin hues fall in the red-orange range (H ≈ 0–0.10 or 0.93–1.0)
    hue_ok = (hue <= 0.10) | (hue >= 0.93)
    sat_ok = (sat >= 0.08) & (sat <= 0.90)
    val_ok = val >= 0.20
    return hue_ok & sat_ok & val_ok


# ── ITA computation ──────

# Three-group scheme (simplified from Chardon et al.):
#   Light  : ITA >  41°
#   Medium : -30° < ITA ≤ 41°
#   Dark   : ITA ≤ -30°
_THRESHOLDS = [(41.0, "Light"), (-30.0, "Medium")]


def _ita_to_label(ita: float) -> str:
    for threshold, label in _THRESHOLDS:
        if ita > threshold:
            return label
    return "Dark"


def compute_ita(pil_image: Image.Image) -> tuple[float | None, str]:
    """
    Estimate ITA and skin-tone proxy label for a PIL image.

    Returns
    -------
    (ita_value, label)  where label ∈ {"Light", "Medium", "Dark", "unknown"}.
    Returns (None, "unknown") when estimation is unreliable.
    """
    rgb = np.array(pil_image.convert("RGB"))
    h, w = rgb.shape[:2]
    mask = _skin_mask(rgb)
    n_skin = int(mask.sum())

    if n_skin >= max(50, h * w * 0.01):
        # Use detected skin pixels
        lab = rgb_to_lab(rgb)
        skin_lab = lab[mask]
        L_med = float(np.median(skin_lab[:, 0]))
        b_med = float(np.median(skin_lab[:, 2]))
    else:
        # Fall back to centre-crop median (less reliable)
        cy, cx = h // 2, w // 2
        ch, cw = max(1, h // 6), max(1, w // 6)
        patch = rgb[cy - ch: cy + ch, cx - cw: cx + cw]
        lab = rgb_to_lab(patch)
        L_med = float(np.median(lab[..., 0]))
        b_med = float(np.median(lab[..., 2]))

    if abs(b_med) < 1e-6:
        return None, "unknown"

    ita = float(np.degrees(np.arctan((L_med - 50.0) / b_med)))
    return round(ita, 2), _ita_to_label(ita)


# ── Manifest annotation ────────
def annotate_manifest(manifest_path: Path, img_root: Path, out_path: Path) -> None:
    """
    Read manifest CSV, compute ITA for every image, and write processed_manifest.csv
    with added columns: ita_value, estimated_skin_tone_proxy.
    """
    df = pd.read_csv(manifest_path)
    ita_vals: list[float | None] = []
    proxy_labels: list[str] = []

    total = len(df)
    for idx, (_, row) in enumerate(df.iterrows(), 1):
        img_path = img_root / row["relative_path"]
        if not img_path.is_file():
            ita_vals.append(None)
            proxy_labels.append("unknown")
            continue
        try:
            pil = Image.open(img_path).convert("RGB")
            ita, label = compute_ita(pil)
            ita_vals.append(ita)
            proxy_labels.append(label)
        except Exception:
            ita_vals.append(None)
            proxy_labels.append("unknown")

        if idx % 200 == 0 or idx == total:
            print(f"  Processed {idx}/{total}", flush=True)

    df["ita_value"] = ita_vals
    df["estimated_skin_tone_proxy"] = proxy_labels
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(df["estimated_skin_tone_proxy"].value_counts().to_string())


if __name__ == "__main__":
    manifest = ROOT / "data/manifest.split.csv"
    img_root = ROOT / "data/raw"
    out = ROOT / "data/processed_manifest.csv"
    if not manifest.exists():
        print("Run data/pipeline_from_manifest.py first to create manifest.split.csv")
    else:
        print("Annotating manifest with ITA skin-tone proxy estimates...")
        annotate_manifest(manifest, img_root, out)
