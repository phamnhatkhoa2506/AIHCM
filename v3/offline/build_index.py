"""Bước đầu — Tier 1 (Vector): gộp toàn bộ clip-features-32 (.npy) + map-keyframes (.csv)
thành 1 ma trận feature + 1 bảng metadata, dùng chung cho search.py.

Giả định đã kiểm chứng: với mỗi video, thứ tự hàng trong file .npy khớp 1-1 với thứ tự
hàng (cột `n`) trong map-keyframes/<video_id>.csv (đã verify trên mẫu, không mismatch).
"""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/

import sys

import faiss
import numpy as np
import pandas as pd

from config import (
    CLIP_FEATURES_DIR,
    INDEX_DIR,
    INDEX_FAISS_PATH,
    INDEX_MATRIX_PATH,
    INDEX_META_PATH,
    MAP_KEYFRAMES_DIR,
)


def build() -> None:
    npy_files = sorted(CLIP_FEATURES_DIR.glob("*.npy"))
    if not npy_files:
        raise SystemExit(f"Không tìm thấy .npy nào trong {CLIP_FEATURES_DIR}")

    feats: list[np.ndarray] = []
    rows: list[dict] = []
    skipped: list[str] = []

    for i, fp in enumerate(npy_files, 1):
        video_id = fp.stem
        csv_path = MAP_KEYFRAMES_DIR / f"{video_id}.csv"
        if not csv_path.exists():
            skipped.append(video_id)
            continue

        feat = np.load(fp).astype(np.float32)
        df = pd.read_csv(csv_path)
        if len(df) != feat.shape[0]:
            skipped.append(video_id)
            continue

        feats.append(feat)
        for local_idx, r in enumerate(df.itertuples(index=False)):
            rows.append(
                {
                    "video_id": video_id,
                    "local_idx": local_idx,
                    "frame_idx": int(r.frame_idx),
                    "pts_time": float(r.pts_time),
                    "fps": float(r.fps),
                }
            )

        if i % 100 == 0 or i == len(npy_files):
            print(f"[{i}/{len(npy_files)}] video da gop", file=sys.stderr)

    matrix = np.concatenate(feats, axis=0)
    # chuẩn hoá L2 để search dùng dot product = cosine similarity
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    meta = pd.DataFrame(rows)
    assert len(meta) == matrix.shape[0], "meta va matrix lech hang"

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_MATRIX_PATH, matrix)
    meta.to_parquet(INDEX_META_PATH, index=False)

    # IndexFlatIP: vector đã L2-normalize -> inner product = cosine similarity, tìm kiếm chính xác
    # (chưa cần approximate/IVF ở quy mô ~177k, để dành khi corpus lớn hơn nhiều).
    matrix_c = np.ascontiguousarray(matrix, dtype=np.float32)
    dim = matrix_c.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(matrix_c)
    faiss.write_index(faiss_index, str(INDEX_FAISS_PATH))

    print(f"Xong: {matrix.shape[0]} frame, {matrix.shape[1]} dim, {len(npy_files) - len(skipped)} video")
    print(f"FAISS index: {faiss_index.ntotal} vector -> {INDEX_FAISS_PATH}")
    if skipped:
        print(f"Bo qua {len(skipped)} video (thieu csv hoac lech so hang): {skipped[:10]}{'...' if len(skipped) > 10 else ''}")


if __name__ == "__main__":
    build()
