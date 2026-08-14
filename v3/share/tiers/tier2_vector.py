"""TẦNG 2 — CLIP vector: xếp hạng theo độ khớp ngữ nghĩa TRONG tập ứng viên đã qua Tầng 1
(hoặc toàn corpus nếu Tầng 1 không lọc gì). ĐANG DÙNG — tầng chính tạo ra score hiện tại.

encode_query(): text -> vector (multilingual, khớp không gian embedding ảnh clip-ViT-B-32).
rank(): vector + tập ứng viên -> DataFrame kết quả đã xếp hạng.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import resources


def encode_query(query: str) -> np.ndarray:
    model = resources.get().model
    return model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)


def rank(
    query_vec: np.ndarray,
    candidates: set[tuple[str, int]] | None,
    top_k: int,
) -> pd.DataFrame:
    """candidates=None -> tìm trên toàn corpus qua FAISS (IndexFlatIP, cosine vì đã L2-normalize).
    candidates cụ thể -> tính cosine TRỰC TIẾP trên đúng tập đó, KHÔNG dùng FAISS-pool-rồi-lọc
    (nguyên tắc: Tầng 1 định nghĩa tập trước, Tầng 2 chỉ xếp hạng trong tập đó)."""
    res = resources.get()

    if candidates is None:
        scores, idx = res.index.search(query_vec, top_k)
        out = res.meta.iloc[idx[0]].copy()
        out["score"] = scores[0]
        return out.reset_index(drop=True)

    if not candidates:
        return res.meta.iloc[0:0].copy().assign(score=[])

    positions = np.array([res.row_pos[k] for k in candidates if k in res.row_pos], dtype=np.int64)
    sub_matrix = res.matrix[positions]
    scores = sub_matrix @ query_vec[0]

    order = np.argsort(-scores)[:top_k]
    out = res.meta.iloc[positions[order]].copy()
    out["score"] = scores[order]
    return out.reset_index(drop=True)
