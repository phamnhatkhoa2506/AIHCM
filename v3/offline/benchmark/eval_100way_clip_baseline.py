"""Baseline: CLIP hien tai tren bai toan retrieval - LUU embedding ra file de dung chung cho
SigLIP/PE-Core/VL-JEPA (khong encode lai), va bao ca 2 kieu do:
  (a) pool RIENG 10 frame/query (cung video, gan thoi gian) - test phan biet chi tiet nho.
  (b) pool CHUNG toan bo 1000 anh (het 100 query x 10 frame) - test quy mo LON hon nhieu,
      gan voi that hon (vd kis_001 that: canh tranh 37209 frame, khong phai 10).

SUA 2026-08-11 (lan 3): (a) rieng le van cho Recall@1=1.000 CA 2 lan thu (ngau nhien va gan
thoi gian) - ket luan: 10 candidate la qua it de thay ro "score clustering" (van de that la
CANH TRANH RONG qua NHIEU video/frame khac nhau, khong phai do gan/xa trong 1 video). Them (b)
de co phep do that su kho, TAN DUNG LAI embedding da encode (khong ton them chi phi Modal).

Chay: python offline/benchmark/eval_100way_clip_baseline.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "share"))

import io
import pickle
import time

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from config import CLIP_IMAGE_MODEL_NAME, MODEL_CACHE_DIR
from tiers.tier2_vector import encode_query

DATA_PATH = _Path(__file__).resolve().parent / "retrieval_100way_data.pkl"
EMB_OUT_PATH = _Path(__file__).resolve().parent / "embeddings_clip.pkl"


def compute_ranks(text_vecs: np.ndarray, img_vecs: np.ndarray, pool_size: int, n_queries: int) -> dict:
    """text_vecs: (n_queries, d). img_vecs: (n_queries*pool_size, d), GT cua query i o vi tri
    i*pool_size trong img_vecs (dung quy uoc build_100way_retrieval.py: GT luon index 0/pool).

    BUG THAT (2026-08-11): encode_query() tra ve mang (1,d) chu khong phai (d,) phang - neu
    KHONG ep phang truoc, text_vecs co the mang shape (n,1,d), khien sims = text_vecs[i]@pool.T
    ra (1,k) thay vi (k,) - np.where(order==0)[0][0] khi do lay nham CHI SO HANG (luon =0,
    vi chi co 1 hang) thay vi CHI SO COT (hang that) -> LUON tra ve rank=1 gia tao du dung sai
    thuc te. Da phat hien qua so sanh mau thuan voi eval_100way_margin.py (margin am nhieu nhung
    Recall@1 bao 1.000 - khong the dung ca 2). Ep phang o day de chan tai dien."""
    img_vecs = np.asarray(img_vecs).reshape(len(img_vecs), -1)
    text_vecs = np.asarray(text_vecs).reshape(len(text_vecs), -1)
    # (a) pool RIENG (10/query)
    ranks_small = []
    for i in range(n_queries):
        pool = img_vecs[i * pool_size: (i + 1) * pool_size]
        sims = text_vecs[i] @ pool.T
        order = np.argsort(-sims)
        ranks_small.append(int(np.where(order == 0)[0][0]) + 1)
    ranks_small = np.array(ranks_small)

    # (b) pool CHUNG (toan bo img_vecs, 1000 anh) - GT cua query i o vi tri i*pool_size
    sims_full = text_vecs @ img_vecs.T  # (n_queries, n_queries*pool_size)
    ranks_full = []
    for i in range(n_queries):
        gt_pos = i * pool_size
        order = np.argsort(-sims_full[i])
        ranks_full.append(int(np.where(order == gt_pos)[0][0]) + 1)
    ranks_full = np.array(ranks_full)

    return {
        "small_pool": {
            "Recall@1": float((ranks_small <= 1).mean()),
            "Recall@3": float((ranks_small <= 3).mean()),
            "Recall@5": float((ranks_small <= 5).mean()),
            "median_rank": float(np.median(ranks_small)),
            "pool_size": pool_size,
        },
        "full_pool": {
            "Recall@1": float((ranks_full <= 1).mean()),
            "Recall@10": float((ranks_full <= 10).mean()),
            "Recall@50": float((ranks_full <= 50).mean()),
            "median_rank": float(np.median(ranks_full)),
            "pool_size": len(img_vecs),
        },
    }


def print_report(name: str, r: dict, elapsed_s: float) -> None:
    print(f"\n=== {name} ===")
    print(f"[pool nho, 10 frame/query cung video]  R@1={r['small_pool']['Recall@1']:.3f}  "
          f"R@3={r['small_pool']['Recall@3']:.3f}  R@5={r['small_pool']['Recall@5']:.3f}  "
          f"median_rank={r['small_pool']['median_rank']:.1f}/{r['small_pool']['pool_size']}")
    print(f"[pool CHUNG, {r['full_pool']['pool_size']} anh]  R@1={r['full_pool']['Recall@1']:.3f}  "
          f"R@10={r['full_pool']['Recall@10']:.3f}  R@50={r['full_pool']['Recall@50']:.3f}  "
          f"median_rank={r['full_pool']['median_rank']:.1f}/{r['full_pool']['pool_size']}")
    print(f"Thoi gian encode: {elapsed_s:.1f}s")


def main() -> None:
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    query_texts, pools_bytes = data["query_texts"], data["pools_bytes"]
    pool_size = len(pools_bytes[0])
    n_queries = len(query_texts)

    t0 = time.perf_counter()
    img_model = SentenceTransformer(CLIP_IMAGE_MODEL_NAME, cache_folder=str(MODEL_CACHE_DIR))
    all_images = [Image.open(io.BytesIO(b)).convert("RGB") for pool in pools_bytes for b in pool]
    img_vecs = img_model.encode(all_images, convert_to_numpy=True, normalize_embeddings=True, batch_size=32)
    t1 = time.perf_counter()

    text_vecs = np.stack([encode_query(t) for t in query_texts])
    t2 = time.perf_counter()

    with open(EMB_OUT_PATH, "wb") as f:
        pickle.dump({"img_vecs": img_vecs, "text_vecs": text_vecs, "pool_size": pool_size}, f)
    print(f"da luu embedding -> {EMB_OUT_PATH}")

    r = compute_ranks(text_vecs, img_vecs, pool_size, n_queries)
    print_report("CLIP hien tai (clip-ViT-B-32 + multilingual text)", r, t1 - t0 + t2 - t1)


if __name__ == "__main__":
    main()
