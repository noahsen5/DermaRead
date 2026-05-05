import csv, hashlib, sqlite3
from pathlib import Path
from PIL import Image

DB = Path("data/dermaread.sqlite")
MANIFEST = Path("data/manifest.csv")
IMGROOT = Path("data/raw")

SCHEMA = """
CREATE TABLE IF NOT EXISTS images(
  id TEXT PRIMARY KEY,
  patient_id TEXT,
  relative_path TEXT,
  label TEXT,
  subtype TEXT,
  skin_tone TEXT,
  source TEXT,
  license_url TEXT,
  consent TEXT,
  split TEXT,
  width INTEGER,
  height INTEGER,
  hash TEXT
);
CREATE INDEX IF NOT EXISTS idx_images_split ON images(split);
CREATE INDEX IF NOT EXISTS idx_images_label ON images(label);
CREATE INDEX IF NOT EXISTS idx_images_skin ON images(skin_tone);
"""

def img_hash(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""): h.update(chunk)
    return h.hexdigest()

def upsert(db, row):
    cols = ",".join(row.keys())
    qs = ",".join(["?"]*len(row))
    db.execute(f"INSERT OR REPLACE INTO images ({cols}) VALUES ({qs})", list(row.values()))

def main():
    conn = sqlite3.connect(DB); cur = conn.cursor()
    for stmt in SCHEMA.strip().split(";"):
        if stmt.strip(): cur.execute(stmt)

    with open(MANIFEST, newline="") as f:
        for r in csv.DictReader(f):
            p = IMGROOT / r["relative_path"]
            try:
                w, h = Image.open(p).size
            except Exception:
                w, h = None, None
            row = {
                "id": r["id"], "patient_id": r["patient_id"],
                "relative_path": r["relative_path"], "label": r["label"],
                "subtype": r.get("subtype","unknown"), "skin_tone": r.get("skin_tone","unknown"),
                "source": r.get("source",""), "license_url": r.get("license_url",""),
                "consent": r.get("consent","unknown"), "split": r.get("split",""),
                "width": w, "height": h, "hash": img_hash(p) if p.exists() else "",
            }
            upsert(cur, row)
    conn.commit(); conn.close()
    print("DB updated:", DB)

if __name__ == "__main__":
    main()
