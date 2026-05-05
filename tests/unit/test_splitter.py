"""Unit tests for the patient-level stratified splitter."""

import csv
import random
import tempfile
from collections import defaultdict
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.pipeline_from_manifest import stratified_patient_split, limit_rows


def _make_rows(n_patients_per_class: int = 10) -> list[dict]:
    rows = []
    for label in ["psoriasis", "non-psoriasis"]:
        for i in range(n_patients_per_class):
            pid = f"{label[:3]}_{i}"
            for img_idx in range(2):  # 2 images per patient
                rows.append({
                    "id": f"{pid}_img{img_idx}",
                    "patient_id": pid,
                    "relative_path": f"{label}/{pid}_{img_idx}.jpg",
                    "label": label,
                    "skin_tone": "unknown",
                    "split": "",
                })
    return rows


class TestStratifiedPatientSplit:
    def test_no_patient_in_multiple_splits(self):
        rows = _make_rows(10)
        split_rows = stratified_patient_split(rows)
        patient_splits = defaultdict(set)
        for r in split_rows:
            patient_splits[r["patient_id"]].add(r["split"])
        for pid, splits in patient_splits.items():
            assert len(splits) == 1, f"Patient {pid} appears in splits: {splits}"

    def test_all_splits_present(self):
        rows = _make_rows(15)
        split_rows = stratified_patient_split(rows)
        splits_found = {r["split"] for r in split_rows}
        assert "train" in splits_found
        assert "test" in splits_found

    def test_train_is_largest_split(self):
        rows = _make_rows(10)
        split_rows = stratified_patient_split(rows)
        counts = defaultdict(int)
        for r in split_rows:
            counts[r["split"]] += 1
        assert counts["train"] >= counts.get("val", 0)
        assert counts["train"] >= counts["test"]

    def test_deterministic_with_seed(self):
        rows = _make_rows(10)
        random.seed(42)
        result_a = [r["split"] for r in stratified_patient_split(list(rows))]
        random.seed(42)
        result_b = [r["split"] for r in stratified_patient_split(list(rows))]
        assert result_a == result_b

    def test_all_rows_accounted_for(self):
        rows = _make_rows(10)
        split_rows = stratified_patient_split(rows)
        assert len(split_rows) == len(rows)

    def test_class_balance_preserved_across_splits(self):
        rows = _make_rows(20)
        split_rows = stratified_patient_split(rows)
        for split in ("train", "test"):
            split_rows_for_split = [r for r in split_rows if r["split"] == split]
            if not split_rows_for_split:
                continue
            labels = [r["label"] for r in split_rows_for_split]
            psoriasis_ratio = labels.count("psoriasis") / len(labels)
            # Should be roughly 50/50 since input was balanced
            assert 0.3 <= psoriasis_ratio <= 0.7


class TestLimitRows:
    def test_limit_caps_per_label(self):
        rows = _make_rows(20)  # 40 per label
        limited = limit_rows(rows, per_label_limit=5)
        by_label = defaultdict(int)
        for r in limited:
            by_label[r["label"]] += 1
        for count in by_label.values():
            assert count <= 5

    def test_no_limit_returns_all(self):
        rows = _make_rows(10)
        assert len(limit_rows(rows, None)) == len(rows)
