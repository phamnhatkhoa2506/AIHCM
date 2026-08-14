"""Tang tim kiem THU 2, SONG SONG voi Tang 2 hien tai (tier2_vector.py) - dung tren bo
keyframe TU TRICH mat do cao hon (AutoShot shot-detection, xem hoi thoai 2026-08-11/13), encode
bang 3 model rieng (SigLIP2/PE-Core/BEiT-3, Modal apps: aic2026-siglip/aic2026-pe-core/
aic2026-beit3) + 1 che do fusion RRF (Reciprocal Rank Fusion).

KHAC Tang 1+2+3 hien tai: bo keyframe nay KHONG co objects_index/ASR di kem (van gac lai
Tier1 object detection - can chay lai Faster-RCNN/DINO tu dau, chua co Modal app). CO OCR
rieng (2026-08-14, build_dense_ocr_index.py, OCR THANG tren bo dense - xem OCR_TEXT_PATH duoi)
dung nhu HARD FILTER truoc khi rank (giong y het tier1_filter.by_text() nhung scope o bo dense).

frame_id nop bai = cot "frame_idx" trong dense_meta.parquet (frame tren truc VIDEO GOC, xem
merge_dense_embeddings.py va PDF muc 3) - dung THANG duoc, khong can quy doi.

4 che do (DENSE_MODES): "siglip" | "pe_core" | "beit3" | "rrf".
"""
from __future__ import annotations

from functools import lru_cache

import faiss
import numpy as np
import pandas as pd

from config import INDEX_DIR
from local_text_encoders import ENCODERS
from tiers.tier1_filter import _strip_accents

DENSE_DIR = INDEX_DIR / "dense"
OCR_TEXT_PATH = DENSE_DIR / "ocr_text.parquet"  # xem build_dense_ocr_index.py - schema:
# video_id, frame_idx, text_raw, text_norm, ymin/xmin/ymax/xmax, score. KHAC ocr_text.parquet
# sparse (co local_idx_start/end, da gom "run") - o day 1 dong/dong chu/frame THANG, khong gom,
# vi frame dense khong deu theo shot (xem docstring build_dense_ocr_index.py).
DENSE_MODES = ["siglip", "pe_core", "beit3", "rrf"]

RRF_K = 60  # hang so chuan trong cong thuc RRF: 1/(k+rank) - k=60 la gia tri pho bien trong
# literature (Cormack et al. 2009), lam mem anh huong cua rank thap ma khong can chuan hoa diem.


@lru_cache(maxsize=None)
def _load_dense_meta() -> pd.DataFrame:
    return pd.read_parquet(DENSE_DIR / "dense_meta.parquet")


@lru_cache(maxsize=None)
def _load_dense_row_pos() -> dict[tuple, int]:
    meta = _load_dense_meta()
    return {(vid, int(fid)): i for i, (vid, fid) in enumerate(zip(meta["video_id"], meta["frame_idx"]))}


@lru_cache(maxsize=None)
def _load_dense_index(model: str):
    matrix = np.load(DENSE_DIR / f"{model}_matrix.npy")
    index = faiss.read_index(str(DENSE_DIR / f"{model}_faiss.index"))
    return matrix, index


def _ocr_candidates(ocr_text: str) -> set[tuple[str, int]] | None:
    """Tra ve set (video_id, frame_idx) co chu KHOP CHINH XAC ocr_text (word-boundary, khong
    phan biet dau) - giong het tier1_filter.by_text(), scope o OCR_TEXT_PATH cua bo dense.
    None neu chua co du lieu OCR dense (chua chay xong build_dense_ocr_index.py)."""
    if not OCR_TEXT_PATH.exists():
        return None
    df = pd.read_parquet(OCR_TEXT_PATH)
    needle = f" {_strip_accents(ocr_text)} "
    hit = df[df["text_norm"].apply(lambda t: needle in f" {t} ")]
    return set(zip(hit["video_id"], hit["frame_idx"].astype(int)))


def _encode_query(query: str, model: str) -> np.ndarray:
    # SUA (2026-08-14, theo yeu cau nguoi dung): encode QUERY TEXT chay LOCAL (giong het
    # pattern tier2_vector.py::encode_query() dung cho CLIP hien tai) thay vi goi Modal
    # remote() moi lan - tranh phu thuoc Modal app con song/da deploy cho duong hoi/dap ONLINE
    # (chi anh CORPUS moi thuc su can Modal GPU, da lam xong o build_dense_embeddings.py).
    v = ENCODERS[model](query)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    return v


def _rank_single(
    query: str, model: str, top_k: int, candidates: set[tuple[str, int]] | None = None
) -> pd.DataFrame:
    """1 model - tra ve DataFrame [video_id, frame_id (=frame_idx), shot_idx, path, score].
    candidates=None -> tim tren TOAN BO index qua FAISS (giong tier2_vector.rank()). candidates
    cu the (vd tu OCR hard-filter) -> tinh cosine TRUC TIEP tren dung tap do, KHONG dung
    FAISS-pool-roi-loc (cung nguyen tac voi tier2_vector.rank())."""
    matrix, index = _load_dense_index(model)
    meta = _load_dense_meta()
    qvec = _encode_query(query, model)

    if candidates is None:
        scores, idx = index.search(qvec, top_k)
        out = meta.iloc[idx[0]].copy()
        out["score"] = scores[0]
        return out.rename(columns={"frame_idx": "frame_id"}).reset_index(drop=True)

    if not candidates:
        return meta.iloc[0:0].copy().rename(columns={"frame_idx": "frame_id"}).assign(score=[])

    row_pos = _load_dense_row_pos()
    positions = np.array([row_pos[k] for k in candidates if k in row_pos], dtype=np.int64)
    if len(positions) == 0:
        return meta.iloc[0:0].copy().rename(columns={"frame_idx": "frame_id"}).assign(score=[])
    sub_matrix = matrix[positions]
    scores = sub_matrix @ qvec[0]
    order = np.argsort(-scores)[:top_k]
    out = meta.iloc[positions[order]].copy()
    out["score"] = scores[order]
    return out.rename(columns={"frame_idx": "frame_id"}).reset_index(drop=True)


def _rank_rrf(
    query: str, top_k: int, candidates: set[tuple[str, int]] | None = None, pool_k: int = 200
) -> pd.DataFrame:
    """Fusion RRF: lay top pool_k tu MOI model rieng le, tinh RRF-score = sum(1/(RRF_K+rank_i))
    qua cac model co xuat hien (khong xuat hien trong top pool_k cua 1 model nao do coi nhu
    rank vo cung, dong gop 0 tu model do - KHONG loai anh, chi khong duoc cong tu nguon do)."""
    rrf_scores: dict[tuple, float] = {}
    per_model_row: dict[tuple, pd.Series] = {}
    for model in ("siglip", "pe_core", "beit3"):
        ranked = _rank_single(query, model, pool_k, candidates=candidates)
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            key = (row["video_id"], int(row["frame_id"]))
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            per_model_row.setdefault(key, row)

    order = sorted(rrf_scores.items(), key=lambda kv: -kv[1])[:top_k]
    rows = []
    for key, score in order:
        row = per_model_row[key].copy()
        row["score"] = score
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame(
        columns=["video_id", "frame_id", "shot_idx", "path", "score"]
    )


def search_dense(query: str, mode: str, top_k: int = 100, ocr_text: str | None = None) -> pd.DataFrame:
    """mode in DENSE_MODES ("siglip"/"pe_core"/"beit3"/"rrf"). ocr_text: hard-filter chu tren
    man hinh (giong tier1_filter.by_text) - None = khong loc, "" cung coi nhu None."""
    if mode not in DENSE_MODES:
        raise ValueError(f"mode phai la 1 trong {DENSE_MODES}, nhan '{mode}'")
    candidates = _ocr_candidates(ocr_text) if ocr_text else None
    if mode == "rrf":
        return _rank_rrf(query, top_k, candidates=candidates)
    return _rank_single(query, mode, top_k, candidates=candidates)
