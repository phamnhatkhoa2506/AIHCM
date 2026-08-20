"""Hàm đo Recall@k dùng CHUNG cho các script eval_100way_*.py (siglip/pe_core/...) — tách riêng
khỏi eval_100way_clip_baseline.py (đã XOÁ 2026-08-20, dọn dẹp pipeline CLIP-32 cũ - xem
share/config.py) vì đây là 2 hàm THUẦN TOÁN HỌC, không phụ thuộc gì vào model/pipeline cụ thể
nào — vẫn cần cho các script đo model MỚI (SigLIP2/PE-Core...), không phải "code cũ".
"""
from __future__ import annotations

import numpy as np


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
