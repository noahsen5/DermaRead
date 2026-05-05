from pathlib import Path
import random, shutil, math

def create_splits(src="data/raw", dst="data/processed",
                  train=0.7, val=0.15, seed=42):
    random.seed(seed)
    src = Path(src); dst = Path(dst)
    classes = [d.name for d in src.iterdir() if d.is_dir()]

    print(f"Classes found: {classes}")
    for c in classes:
        files = [p for p in (src/c).glob("*") if p.is_file()]
        n = len(files)
        print(f"\nClass '{c}': {n} files in raw.")

        if n == 0:
            print("  (skipped: no files)")
            continue

        # Safe split for tiny datasets (ensures at least 1 file goes somewhere)
        n_train = max(1, math.floor(train*n)) if n >= 3 else max(1, n-1)
        n_val   = max(0, math.floor(val*n))
        if n_train + n_val >= n:  # leave at least 1 for test if possible
            n_val = max(0, n - n_train - 1)

        idx = list(range(n)); random.shuffle(idx)
        train_idx = set(idx[:n_train])
        val_idx   = set(idx[n_train:n_train+n_val])
        test_idx  = set(idx) - train_idx - val_idx

        splits = {
            "train": [files[i] for i in train_idx],
            "val":   [files[i] for i in val_idx],
            "test":  [files[i] for i in test_idx],
        }

        for split, items in splits.items():
            out = dst/ split / c
            out.mkdir(parents=True, exist_ok=True)
            for f in items:
                shutil.copy(f, out / f.name)
            print(f"  -> {split}: {len(items)} files -> {out}")

if __name__ == "__main__":
    create_splits()
