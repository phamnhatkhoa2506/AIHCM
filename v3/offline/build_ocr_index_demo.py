"""DEMO/GIA LAP — sinh vai dong ocr_text.parquet gia de test luong UI (canvas keo-tha +
by_text()) TRUOC khi build pipeline OCR that (PaddleOCR + gom cum). KHONG dung o production,
chi de kiem tra day noi dung dung, thay the bang offline/build_ocr_index.py that sau nay.

Dung dung video_id/local_idx CO THAT trong corpus (L26_V246) de anh keyframe hien thi duoc
that trong UI luc test, khong phai vid gia."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))

import pandas as pd

from config import INDEX_DIR
from tiers.tier1_filter import _strip_accents

OUT_PATH = INDEX_DIR / "ocr_text.parquet"

# Moi dong: 1 "run" (cum lien tuc) gia lap - gan giong ket qua that se co sau khi gom cum theo
# thoi gian (_split_and_thin) tu PaddleOCR. video L26_V246 (157 frame, 0..156) dung lam mau.
_ROWS = [
    dict(video_id="L26_V246", local_idx_start=0, local_idx_end=40,
         text_raw="MÓN NGON MỖI NGÀY", ymin=0.85, xmin=0.05, ymax=0.98, xmax=0.55,
         score=0.97, source="ocr_demo"),
    dict(video_id="L26_V246", local_idx_start=40, local_idx_end=90,
         text_raw="GIÒ HEO HON DỪA NƯỚC", ymin=0.85, xmin=0.05, ymax=0.98, xmax=0.60,
         score=0.95, source="ocr_demo"),
    dict(video_id="L26_V246", local_idx_start=60, local_idx_end=110,
         text_raw="VIVU TV", ymin=0.02, xmin=0.80, ymax=0.10, xmax=0.98,
         score=0.99, source="ocr_demo"),
    dict(video_id="L26_V246", local_idx_start=100, local_idx_end=156,
         text_raw="NGUYÊN LIỆU: 500G GIÒ HEO", ymin=0.30, xmin=0.10, ymax=0.40, xmax=0.70,
         score=0.88, source="ocr_demo"),
]

for r in _ROWS:
    r["text_norm"] = _strip_accents(r["text_raw"])

df = pd.DataFrame(_ROWS)
df = df[["video_id", "local_idx_start", "local_idx_end", "text_raw", "text_norm",
         "ymin", "xmin", "ymax", "xmax", "score", "source"]]
df.to_parquet(OUT_PATH, index=False)
print(f"da ghi {len(df)} dong gia vao {OUT_PATH}")
print(df.to_string(index=False))
