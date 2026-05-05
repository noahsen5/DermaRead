"""Unit tests for manifest loading and the ManifestDataset."""

import csv
import tempfile
from pathlib import Path

import pytest

from models.dataset import CLASS_NAMES, ManifestDataset


def _write_manifest(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


class TestClassNames:
    def test_binary(self):
        assert set(CLASS_NAMES) == {"psoriasis", "non-psoriasis"}

    def test_order_stable(self):
        # Index 0 must always be non-psoriasis so the psoriasis logit is at index 1
        assert CLASS_NAMES[0] == "non-psoriasis"
        assert CLASS_NAMES[1] == "psoriasis"


class TestManifestDataset:
    def test_raises_without_split_column(self, tmp_path):
        manifest = tmp_path / "m.csv"
        _write_manifest(manifest, [
            {"id": "a", "patient_id": "p1", "relative_path": "x.jpg", "label": "psoriasis"},
        ])
        with pytest.raises(ValueError, match="split"):
            ManifestDataset(manifest, split="train", img_root=tmp_path)

    def test_raises_on_empty_split(self, tmp_path):
        manifest = tmp_path / "m.csv"
        _write_manifest(manifest, [
            {"id": "a", "patient_id": "p1", "relative_path": "x.jpg",
             "label": "psoriasis", "split": "val"},
        ])
        with pytest.raises(ValueError):
            ManifestDataset(manifest, split="train", img_root=tmp_path)

    def test_len_matches_split(self, tmp_path):
        manifest = tmp_path / "m.csv"
        rows = [
            {"id": f"img_{i}", "patient_id": f"p{i}", "relative_path": f"img_{i}.jpg",
             "label": "psoriasis", "split": "train"}
            for i in range(5)
        ] + [
            {"id": "val_0", "patient_id": "pv", "relative_path": "val_0.jpg",
             "label": "non-psoriasis", "split": "val"}
        ]
        _write_manifest(manifest, rows)
        ds = ManifestDataset(manifest, split="train", img_root=tmp_path)
        assert len(ds) == 5

    def test_label_canonicalisation(self, tmp_path):
        manifest = tmp_path / "m.csv"
        _write_manifest(manifest, [
            {"id": "a", "patient_id": "p1", "relative_path": "a.jpg",
             "label": "psoriasis", "split": "train"},
            {"id": "b", "patient_id": "p2", "relative_path": "b.jpg",
             "label": "non-psoriasis", "split": "train"},
        ])
        ds = ManifestDataset(manifest, split="train", img_root=tmp_path)
        labels_in_manifest = set(ds.df["label"].unique())
        for lbl in labels_in_manifest:
            assert lbl in CLASS_NAMES
