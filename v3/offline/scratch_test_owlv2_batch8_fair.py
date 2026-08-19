"""Do lai batch=8 CUNG phuong phap voi batch16 test (warmup truoc, roi do rieng) de so sanh cong bang."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "share"))
import json
import modal
import pandas as pd
from config import DENSE_META_PATH, INDEX_DIR

LABEL_VI_PATH = INDEX_DIR / "label_vi.json"

with open(LABEL_VI_PATH, encoding="utf-8") as f:
    labels = sorted(json.load(f).keys())

meta = pd.read_parquet(DENSE_META_PATH)
sample = meta.iloc[::2000].head(8)
images_bytes = [Path(p).read_bytes() for p in sample["path"]]

Detector = modal.Cls.from_name("aic2026-owlv2", "OWLv2Detector")
detector = Detector()

detector.detect_batch.remote(images_bytes[:2], labels)  # warmup

t0 = time.time()
results = detector.detect_batch.remote(images_bytes, labels)
elapsed = time.time() - t0

n_det = sum(len(r) for r in results)
print(f"8 anh/1 call: {elapsed:.2f}s -> {8/elapsed*60:.1f} anh/phut/container, {n_det} detection")
