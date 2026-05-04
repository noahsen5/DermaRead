"""
Visual Severity Estimation — NOT a clinical PASI score.

Estimates three visual indicators from a single image using colour and
texture analysis. These are coarse proxies for components of the PASI
(Psoriasis Area and Severity Index) and must never be presented as
equivalent to a clinical assessment.

PASI requires clinical examination, four body regions, and scoring of
induration (tissue thickness) — none of which can be derived from a 2D
photograph. This module provides exploratory visual approximations only.

Indicators returned:
  estimated_coverage_pct  — approximate % of visible skin with abnormal colour
  erythema_index          — 0.0–1.0, proxy for redness in suspected lesion area
  texture_score           — 0.0–1.0, proxy for surface roughness / scaling
  severity_label          — "Mild" / "Moderate" / "Severe" composite label
"""

from __future__ import annotations

import numpy as np
from PIL import Image


from models.skin_tone_ita import _skin_mask  # single authoritative skin detector


# ── Lesion detection within skin pixels ──────
def _lesion_mask(rgb: np.ndarray, skin: np.ndarray) -> np.ndarray:
    """
    Within skin pixels, flag those with inflammatory (high-saturation red)
    or unusually light (silvery scale) colour as suspected lesion pixels.
    Neither heuristic is clinically validated.
    """
    f = rgb.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    maxc = np.maximum.reduce([r, g, b])
    minc = np.minimum.reduce([r, g, b])
    sat = np.where(maxc > 0, (maxc - minc) / maxc, 0.0)
    val = maxc
    # Red-inflamed: high saturation, warm hue, above-average redness vs green
    inflamed = skin & (sat > 0.25) & (r > g + 0.05) & (r > b + 0.05)
    # Silvery scale: very high value, very low saturation (whitish patches)
    scaly = skin & (val > 0.80) & (sat < 0.12)
    return inflamed | scaly


# ── Texture analysis ──────
def _texture_roughness(rgb: np.ndarray, mask: np.ndarray, window: int = 5) -> float:
    """
    Local standard deviation of lightness within masked pixels.
    Higher values indicate more surface variation (proxy for scaling).
    """
    if mask.sum() < 100:
        return 0.0
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    half = window // 2
    h, w = gray.shape
    local_stds = []
    rows, cols = np.where(mask)
    # Sample up to 500 pixels for speed
    if len(rows) > 500:
        idx = np.random.choice(len(rows), 500, replace=False)
        rows, cols = rows[idx], cols[idx]
    for r, c in zip(rows, cols):
        r0, r1 = max(0, r - half), min(h, r + half + 1)
        c0, c1 = max(0, c - half), min(w, c + half + 1)
        patch = gray[r0:r1, c0:c1]
        local_stds.append(patch.std())
    raw = float(np.mean(local_stds)) if local_stds else 0.0
    # Normalise: typical std for smooth skin ~3–8, rough surface ~15–35 (0–255 scale)
    return float(np.clip(raw / 35.0, 0.0, 1.0))


# ── Erythema index ─────────
def _erythema_index(rgb: np.ndarray, lesion: np.ndarray, skin: np.ndarray) -> float:
    """
    Ratio of redness in lesion pixels vs. surrounding skin pixels.
    Based on the principle that inflamed skin has a higher R/(R+G+B) ratio.
    """
    if lesion.sum() < 20:
        return 0.0
    f = rgb.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    total = r + g + b + 1e-6
    redness = r / total

    lesion_r = float(redness[lesion].mean())
    # Background skin (non-lesion)
    bg = skin & ~lesion
    bg_r = float(redness[bg].mean()) if bg.sum() > 20 else lesion_r * 0.7
    # Erythema = excess redness above background, normalised
    excess = max(0.0, lesion_r - bg_r)
    # ~0.05 excess = mild, ~0.15+ = severe (empirical thresholds)
    return float(np.clip(excess / 0.15, 0.0, 1.0))


# ── Composite severity label ────

def _composite_label(coverage: float, erythema: float, texture: float) -> str:
    score = (coverage / 100) * 0.5 + erythema * 0.3 + texture * 0.2
    if score < 0.25:
        return "Mild"
    if score < 0.55:
        return "Moderate"
    return "Severe"


# ── Public API ────────

def estimate_visual_severity(pil_image: Image.Image) -> dict:
    """
    Compute visual severity indicators for a skin image.

    Returns a dict with:
        estimated_coverage_pct  float  0–100
        erythema_index          float  0.0–1.0
        texture_score           float  0.0–1.0
        severity_label          str    Mild / Moderate / Severe
        reliable                bool   False if too few skin pixels detected
    """
    rgb = np.array(pil_image.convert("RGB"))
    skin = _skin_mask(rgb)
    n_skin = int(skin.sum())

    if n_skin < max(50, rgb.shape[0] * rgb.shape[1] * 0.02):
        return {
            "estimated_coverage_pct": None,
            "erythema_index": None,
            "texture_score": None,
            "severity_label": "Unknown",
            "reliable": False,
        }

    lesion = _lesion_mask(rgb, skin)
    n_lesion = int(lesion.sum())

    coverage = round(float(n_lesion / n_skin * 100), 1)
    erythema = round(_erythema_index(rgb, lesion, skin), 3)
    texture  = round(_texture_roughness(rgb, lesion), 3)
    label    = _composite_label(coverage, erythema, texture)

    return {
        "estimated_coverage_pct": coverage,
        "erythema_index": erythema,
        "texture_score": texture,
        "severity_label": label,
        "reliable": True,
    }


# ── Body part clinical context ───────
BODY_PARTS = [
    "Not specified",
    "Scalp",
    "Face",
    "Trunk (chest / abdomen / back)",
    "Upper limbs (arms / elbows)",
    "Lower limbs (legs / knees)",
    "Hands / Feet",
    "Skin folds (axillae / groin)",
    "Nails",
    "Multiple / Widespread",
]

_CLINICAL_NOTES = {
    "Not specified": "",
    "Scalp": (
        "Scalp psoriasis affects approximately 50% of patients and is frequently "
        "the first site of involvement. It can extend beyond the hairline. "
        "Associated with nail changes in a significant proportion of cases."
    ),
    "Face": (
        "Facial psoriasis is uncommon but causes significant distress. "
        "Sebopsoriasis — an overlap with seborrhoeic dermatitis — should be considered. "
        "Periocular involvement requires ophthalmological awareness."
    ),
    "Trunk (chest / abdomen / back)": (
        "Trunk involvement is common in both plaque and guttate psoriasis. "
        "Large surface area means trunk lesions contribute substantially to total "
        "disease burden. Guttate onset often presents here after streptococcal infection."
    ),
    "Upper limbs (arms / elbows)": (
        "Elbow involvement is one of the most classic and frequent presentations "
        "of plaque psoriasis. Extensor surfaces are preferentially affected. "
        "Koebner phenomenon (lesions at sites of trauma) is common on forearms."
    ),
    "Lower limbs (legs / knees)": (
        "Knee involvement mirrors elbow presentation in frequency and pattern. "
        "Lower leg involvement can occasionally resemble venous stasis changes. "
        "Assess for concomitant elbow and scalp involvement."
    ),
    "Hands / Feet": (
        "Palmoplantar psoriasis is particularly disabling as it directly impairs "
        "daily function and occupational activity. A pustular variant (palmoplantar "
        "pustulosis) exists and may represent a distinct entity. Nail involvement "
        "is frequently co-present."
    ),
    "Skin folds (axillae / groin)": (
        "Inverse (flexural) psoriasis presents without the typical silvery scale — "
        "lesions appear smooth, shiny, and erythematous due to moisture in skin folds. "
        "This variant requires different topical treatment than plaque disease. "
        "Candidal superinfection should be excluded."
    ),
    "Nails": (
        "Nail psoriasis occurs in ~50% of psoriasis patients and is an independent "
        "risk factor for psoriatic arthritis. Features include pitting, onycholysis, "
        "subungual hyperkeratosis, and oil-drop discolouration. "
        "Nail involvement alone does not always correlate with skin severity."
    ),
    "Multiple / Widespread": (
        "Widespread involvement (>10% BSA) is considered moderate-to-severe disease "
        "and typically warrants systemic or biologic therapy consideration in a "
        "clinical setting. PASI scoring across all body regions would be required "
        "for formal disease quantification."
    ),
}


def get_clinical_note(body_part: str) -> str:
    return _CLINICAL_NOTES.get(body_part, "")
