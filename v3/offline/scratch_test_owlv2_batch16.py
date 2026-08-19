"""Test tam thoi: do toc do OWLv2 khi batch=16 (so voi batch=8 dang dung trong production va
batch=32 da test truoc do va bi cham hon). Goi TACH BIET, khong dung chung container voi job
full corpus dang chay (Modal se tu autoscale container moi cho lenh nay)."""
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
sample = meta.iloc[::2000].head(16)  # 16 anh rai rac
images_bytes = [Path(p).read_bytes() for p in sample["path"]]

Detector = modal.Cls.from_name("aic2026-owlv2", "OWLv2Detector")
detector = Detector()

# warmup 8 anh (bang batch production) de container "am" truoc khi do
detector.detect_batch.remote(images_bytes[:8], labels)

t0 = time.time()
results = detector.detect_batch.remote(images_bytes, labels)
elapsed = time.time() - t0

n_det = sum(len(r) for r in results)
print(f"16 anh/1 call: {elapsed:.2f}s -> {16/elapsed*60:.1f} anh/phut/container, {n_det} detection")
print(f"So sanh: batch=8 hien tai ~{8/10.0*60:.1f} anh/phut/container (uoc tinh tu 10.0s/8 anh da do truoc)")
