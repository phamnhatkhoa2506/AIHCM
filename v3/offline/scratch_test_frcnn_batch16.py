"""Test tam thoi: do toc do Faster R-CNN Inception-ResNet-v2 khi goi 1 Modal call voi 16 anh
(lap for-loop trong container, KHONG batch GPU that - xem frcnn_incres_app.py) so voi baseline
batch=1 da chay (~245 anh/phut o 8000 anh dau). So sanh thoi gian .remote() cho 16 anh 1 lan
vs uoc tinh 16 lan goi rieng le (suy tu toc do da do)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "share"))
import modal
import pandas as pd
from config import INDEX_DIR

DENSE_META_PATH = INDEX_DIR / "dense" / "dense_meta.parquet"

meta = pd.read_parquet(DENSE_META_PATH)
sample = meta.iloc[::1000].head(16)  # 16 anh rai rac, tranh cache/trung anh
images_bytes = [Path(p).read_bytes() for p in sample["path"]]

Detector = modal.Cls.from_name("aic2026-frcnn-incres", "FasterRCNNDetector")
detector = Detector()

# warmup 1 anh de loai tru cold start container khoi phep do
detector.detect_batch.remote(images_bytes[:1])

t0 = time.time()
results = detector.detect_batch.remote(images_bytes)
elapsed = time.time() - t0

print(f"16 anh trong 1 call: {elapsed:.2f}s -> {16/elapsed*60:.1f} anh/phut (1 container)")
print(f"So sanh: baseline batch=1 thuc te ~245 anh/phut/toan bo 10 container "
      f"(~24.5 anh/phut/container)")
