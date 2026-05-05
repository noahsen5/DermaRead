"""Integration tests for the data pipeline and dataset loading."""

import csv
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def _make_synthetic_manifest(tmp_path: Path, n_per_class: int = 6) -> Path:
    """Create a minimal manifest CSV and matching synthetic images."""
    img_dir = tmp_path / "raw"
    for label in ("psoriasis", "non-psoriasis"):
        (img_dir / label).mkdir(parents=True)
        for i in range(n_per_class):
            arr = np.random.randint(100, 200, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(img_dir / label / f"{label[:3]}_{i}.jpg")

    manifest = tmp_path / "manifest.csv"
    rows = []
    for label in ("psoriasis", "non-psoriasis"):
        for i in range(n_per_class):
            rows.append({
                "id": f"{label[:3]}_{i}",
                "patient_id": f"{label[:3]}_{i}",
                "relative_path": f"{label}/{label[:3]}_{i}.jpg",
                "label": label,
                "subtype": "",
                "skin_tone": "unknown",
                "source": "synthetic",
                "license_url": "",
                "consent": "",
                "split": "",
            })
    with open(manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return manifest


class TestPipelineFromManifest:
    def test_split_csv_created(self, tmp_path):
        import sys
        sys.path.insert(0, str(ROOT))
        from data.pipeline_from_manifest import main as pipeline_main
        import importlib
        import data.pipeline_from_manifest as pm

        manifest = _make_synthetic_manifest(tmp_path)
        orig_in = pm.MANIFEST_IN
        orig_out = pm.MANIFEST_OUT
        orig_raw = pm.RAW
        orig_proc = pm.PROC
        try:
            pm.MANIFEST_IN  = manifest
            pm.MANIFEST_OUT = tmp_path / "manifest.split.csv"
            pm.RAW  = tmp_path / "raw"
            pm.PROC = tmp_path / "processed"
            pipeline_main()
            assert (tmp_path / "manifest.split.csv").exists()
        finally:
            pm.MANIFEST_IN  = orig_in
            pm.MANIFEST_OUT = orig_out
            pm.RAW  = orig_raw
            pm.PROC = orig_proc

    def test_split_has_required_columns(self, tmp_path):
        import data.pipeline_from_manifest as pm
        import pandas as pd

        manifest = _make_synthetic_manifest(tmp_path)
        split_out = tmp_path / "manifest.split.csv"
        orig_in, orig_out, orig_raw, orig_proc = pm.MANIFEST_IN, pm.MANIFEST_OUT, pm.RAW, pm.PROC
        try:
            pm.MANIFEST_IN, pm.MANIFEST_OUT = manifest, split_out
            pm.RAW, pm.PROC = tmp_path / "raw", tmp_path / "processed"
            pm.main()
            df = pd.read_csv(split_out)
            for col in ("id", "patient_id", "relative_path", "label", "split"):
                assert col in df.columns, f"Missing column: {col}"
        finally:
            pm.MANIFEST_IN, pm.MANIFEST_OUT, pm.RAW, pm.PROC = orig_in, orig_out, orig_raw, orig_proc

    def test_no_patient_leaks_across_splits(self, tmp_path):
        import data.pipeline_from_manifest as pm
        import pandas as pd
        from collections import defaultdict

        manifest = _make_synthetic_manifest(tmp_path, n_per_class=9)
        split_out = tmp_path / "manifest.split.csv"
        orig_in, orig_out, orig_raw, orig_proc = pm.MANIFEST_IN, pm.MANIFEST_OUT, pm.RAW, pm.PROC
        try:
            pm.MANIFEST_IN, pm.MANIFEST_OUT = manifest, split_out
            pm.RAW, pm.PROC = tmp_path / "raw", tmp_path / "processed"
            pm.main()
            df = pd.read_csv(split_out)
            patient_splits = defaultdict(set)
            for _, row in df.iterrows():
                patient_splits[row["patient_id"]].add(row["split"])
            for pid, splits in patient_splits.items():
                assert len(splits) == 1, f"Patient {pid} leaked into splits: {splits}"
        finally:
            pm.MANIFEST_IN, pm.MANIFEST_OUT, pm.RAW, pm.PROC = orig_in, orig_out, orig_raw, orig_proc


class TestManifestDatasetLoading:
    def test_images_load_without_error(self, tmp_path):
        import data.pipeline_from_manifest as pm
        from models.dataset import ManifestDataset

        manifest = _make_synthetic_manifest(tmp_path, n_per_class=6)
        split_out = tmp_path / "manifest.split.csv"
        orig_in, orig_out, orig_raw, orig_proc = pm.MANIFEST_IN, pm.MANIFEST_OUT, pm.RAW, pm.PROC
        try:
            pm.MANIFEST_IN, pm.MANIFEST_OUT = manifest, split_out
            pm.RAW, pm.PROC = tmp_path / "raw", tmp_path / "processed"
            pm.main()
            ds = ManifestDataset(split_out, split="train", img_root=tmp_path / "raw")
            assert len(ds) > 0
            x, y, meta = ds[0]
            assert x.shape == (3, 224, 224)
            assert y in (0, 1)
        finally:
            pm.MANIFEST_IN, pm.MANIFEST_OUT, pm.RAW, pm.PROC = orig_in, orig_out, orig_raw, orig_proc
