## Manifest schema

The manifest is the project’s source of truth for every example. Each row
describes one image and the metadata required by downstream steps (DB mirror,
splitting, dataset loader, fairness metrics, etc.).

| column        | required | description |
|---------------|----------|-------------|
| `id`          | ✅ | Unique row identifier. Convention: `<patient_id>_<index>` |
| `patient_id`  | ✅ | Pseudonymized patient / case ID used for patient-level splits. |
| `relative_path` | ✅ | Path to the image relative to `data/raw/`. |
| `label`       | ✅ | Ground-truth diagnosis. Use the canonical strings `non-psoriasis` or `psoriasis`. |
| `subtype`     | optional | Clinical subtype (e.g., plaque, guttate). |
| `skin_tone`   | ✅ | Fitzpatrick tone bucket (`I`–`VI`). Used for bias slices. |
| `source`      | optional | Data source for provenance tracking. |
| `license_url` | optional | License or usage terms. |
| `consent`     | optional | Consent status (research_use, etc.). |
| `split`       | derived | Added by `data/pipeline_from_manifest.py`; one of `train`/`val`/`test`. |

### Workflow

1. Populate `data/raw/` with images and run `python data/generate_sample_data.py` (or export your own CSV following the schema) to create `data/manifest.csv`.
2. Mirror the manifest into SQLite: `python data/make_db.py` → `data/dermaread.sqlite`.
3. Create patient-level splits + processed copies: `python data/pipeline_from_manifest.py` → updates `data/manifest.split.csv` and copies files under `data/processed/<split>/<label>/`.
4. Train the baseline from the split manifest: `.venv/bin/python -m models.train_baseline` → `models/checkpoints/resnet18_baseline.pt`.
5. Consume the split manifest everywhere else (notebook metrics, loader, app).
