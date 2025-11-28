import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

random.seed(42)

MANIFEST_IN = Path("data/manifest.csv")
MANIFEST_OUT = Path("data/manifest.split.csv")
RAW = Path("data/raw")
PROC = Path("data/processed")


def _patient_bins(n: int, ratios):
    train_ratio, val_ratio, test_ratio = ratios
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0

    train_n = max(1, round(train_ratio * n))
    val_n = max(0, round(val_ratio * n)) if n >= 3 else 0
    remaining = n - train_n - val_n
    test_n = max(1, remaining) if n >= 2 else remaining

    if remaining <= 0:
        test_n = 1
        if train_n > 1:
            train_n -= 1
        elif val_n > 0:
            val_n -= 1

    while train_n + val_n + test_n > n:
        if train_n > 1:
            train_n -= 1
        elif val_n > 0:
            val_n -= 1
        else:
            test_n -= 1

    total = train_n + val_n + test_n
    if total < n:
        train_n += n - total

    return train_n, val_n, test_n


def stratified_patient_split(rows, train=0.7, val=0.15):
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    out = []
    for label_rows in by_label.values():
        patients = {}
        for r in label_rows:
            patients.setdefault(r["patient_id"], []).append(r)
        ids = list(patients.keys())
        random.shuffle(ids)
        n = len(ids)
        t, v, te = _patient_bins(n, (train, val, 1 - train - val))
        split_ids = {
            "train": ids[:t],
            "val": ids[t : t + v],
            "test": ids[t + v : t + v + te],
        }
        for split, pid_list in split_ids.items():
            for pid in pid_list:
                for r in patients[pid]:
                    rr = dict(r)
                    rr["split"] = split
                    out.append(rr)
    return out

def copy_files(rows):
    for r in rows:
        src = RAW / r["relative_path"]
        dst = PROC / r["split"] / r["label"] / Path(r["relative_path"]).name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)

def main():
    rows = list(csv.DictReader(open(MANIFEST_IN)))
    rows = stratified_patient_split(rows)
    copy_files(rows)
    with open(MANIFEST_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("Processed to:", PROC)

if __name__ == "__main__":
    main()
