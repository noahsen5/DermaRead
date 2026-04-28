# How to build the external validation dataset

This is a step-by-step guide. Read it fully before downloading anything.

---

## What this dataset is FOR

This is an **external validation set** — not training data.
You use it to test whether the trained models work on real-world images
from different sources than the 10,000-image training set.

This is the dissertation's answer to: *"Does the model generalise?"*

---

## Step 1 — Understand the folder layout

```
data/external_validation/
  psoriasis/
    ps_001.jpg
    ps_002.jpg
    ...
  non_psoriasis/
    nps_001.jpg
    nps_002.jpg
    ...
  external_manifest.csv   ← you fill this in as you add images
  HOW_TO_BUILD_DATASET.md ← this file
```

Rules:
- All filenames must be unique across BOTH folders.
- Use simple names: ps_001.jpg, ps_002.jpg ... nps_001.jpg etc.
- Only JPEG or PNG. No HEIC, no WebP, no BMP.
- Do not put images anywhere else in the project.

---

## Step 2 — How many images to collect

| Class | Minimum | Target | Maximum |
|---|---|---|---|
| psoriasis | 20 | 40 | 60 |
| non-psoriasis | 20 | 40 | 60 |

Aim for roughly equal numbers per class.
Quality and diversity matter more than quantity here.

---

## Step 3 — Where to get images

### Kaggle datasets (recommended — free, citable)

Good datasets to look at:
- "Skin Disease Dataset" (HAM10000 subset)
- "Psoriasis Skin Disease Images"
- "DermNet Skin Disease Atlas"
- "ISIC 2019 Challenge" (dermoscopy, diverse)
- Any dataset with "skin lesion" or "dermatology" in the name

When you download from Kaggle:
1. Check the licence tab — most say CC BY-NC-SA or similar.
   As long as it is for academic/non-commercial use you are fine for a dissertation.
2. Note the exact dataset name and URL — you will need to cite this.

### What to look for in a dataset

GOOD:
- Images taken with real cameras or phones (clinical macro-photography)
- Mix of skin tones (deliberately pick some from darker skin tones)
- Different lighting conditions
- Psoriasis visible on different body parts (elbows, scalp, hands, torso)

AVOID:
- Dermoscopic images (special lens device — very different from normal photos)
- Already augmented images (blurry, heavily filtered)
- Very low resolution (below 100×100 pixels)
- Images with heavy watermarks or text overlays

### Can you use Google Images?

Yes, for a dissertation, using a small number of images for non-commercial
academic research is generally considered fair use. But:
- You CANNOT publish these images or include them in any public dataset.
- You must note in the dissertation that these were collected for evaluation only.
- Kaggle datasets are better because they are formally licensed.

---

## Step 4 — Selecting for skin tone diversity

For your fairness analysis you want a spread across skin tones.
You don't need equal numbers per tone — just don't pick ONLY one.

Suggested rough target:
- ~15 images with visibly lighter skin tones
- ~15 images with visibly medium/tan skin tones
- ~10+ images with visibly darker skin tones

When you add an image to the manifest, fill in skin_tone_notes honestly:
- "light" / "medium" / "dark" / "mixed" / "not visible"

---

## Step 5 — Filling in the manifest

The file is: data/external_validation/external_manifest.csv

Columns:

| Column | What to put |
|---|---|
| id | Unique ID. Use: ps_001, ps_002 ... nps_001, nps_002 |
| relative_path | Path from this folder. E.g. psoriasis/ps_001.jpg |
| label | psoriasis OR non-psoriasis (must match exactly) |
| source_dataset | Name of the Kaggle dataset or "google_images" |
| image_type | clinical_photo / dermoscopy / smartphone / web |
| skin_tone_notes | light / medium / dark / mixed / not_visible |
| license | CC-BY-NC or CC0 or fair_use_academic etc. |
| notes | Anything relevant — body part, image quality issues |

### Example rows (copy and edit these)

```
ps_001,psoriasis/ps_001.jpg,psoriasis,Kaggle Psoriasis Dataset,clinical_photo,medium,CC-BY-NC,Plaque psoriasis on forearm
ps_002,psoriasis/ps_002.jpg,psoriasis,Kaggle Skin Disease Dataset,clinical_photo,dark,CC-BY-NC,Scalp psoriasis
nps_001,non_psoriasis/nps_001.jpg,non-psoriasis,HAM10000,dermoscopy,light,CC-BY-NC-SA,Melanocytic nevi
nps_002,non_psoriasis/nps_002.jpg,non-psoriasis,google_images,web,medium,fair_use_academic,Eczema on arm
```

---

## Step 6 — After adding images, run the evaluation

```bash
python models/evaluate_external.py
```

This tests all three model versions (V1, V2, V3) on your images and saves:
- outputs/external_validation/external_results.csv
- outputs/external_validation/external_summary.md

You can run it repeatedly as you add more images.

---

## Step 7 — What to write in the dissertation

> "An external validation set of N images was collected from publicly available
> Kaggle datasets (cite them here) and supplementary web images used under fair use
> for non-commercial academic research. Images were selected to include a range of
> skin tones and image types not present in the original training data, with the
> specific aim of evaluating out-of-distribution generalisation.
> The external set was not used for training or hyperparameter selection at any stage."

Then present the external accuracy table from external_summary.md.

---

## Step 8 — Common mistakes to avoid

- Do NOT put the same image in both psoriasis and non_psoriasis folders.
- Do NOT use images from the existing data/raw/ folder (that is training data).
- Do NOT use dermoscopy images if your training set has none (the model will fail on both equally — that is not a useful comparison).
- Do NOT forget to fill in the source_dataset column — you will need it for citation.
