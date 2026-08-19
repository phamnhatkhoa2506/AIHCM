"""Recall@k nhi phan da BAO HOA (CLIP dat 1.000 ca 2 lan thu thiet ke pool khac nhau - pool
qua de vi da so distractor thuoc video/chu de HOAN TOAN khac nhau). Doi sang MARGIN LIEN TUC:
sim(text, GT) - max(sim(text, distractor)) - cho biet MUC DO TU TIN thay vi chi dung/sai nhi
phan, van dung DUNG embedding da encode (khong ton them Modal).

Chay: python offline/benchmark/eval_100way_margin.py <embeddings_XXX.pkl> [<ten_model>]
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np


def main(emb_path: str, name: str) -> None:
    with open(emb_path, "rb") as f:
        d = pickle.load(f)
    img_vecs, text_vecs, pool_size = d["img_vecs"], d["text_vecs"], d["pool_size"]
    img_vecs = np.array(img_vecs).reshape(len(img_vecs), -1)
    text_vecs = np.array(text_vecs).reshape(len(text_vecs), -1)  # encode_query() tra (1,d),
    # np.stack them 1 chieu thua -> (n,1,d) - ep phang lai (n,d).
    n_queries = len(text_vecs)

    margins_small = []  # so voi 9 distractor CUNG VIDEO (pool nho)
    for i in range(n_queries):
        pool = img_vecs[i * pool_size: (i + 1) * pool_size]
        sims = text_vecs[i] @ pool.T
        margins_small.append(sims[0] - sims[1:].max())

    sims_full = text_vecs @ img_vecs.T
    margins_full = []
    for i in range(n_queries):
        gt_pos = i * pool_size
        row = sims_full[i].copy()
        gt_score = row[gt_pos]
        row[gt_pos] = -np.inf
        margins_full.append(gt_score - row.max())

    margins_small, margins_full = np.array(margins_small), np.array(margins_full)
    print(f"=== {name} — margin (sim(text,GT) - sim(text,doi_thu_manh_nhat)) ===")
    print(f"pool nho (9 distractor cung video):  mean={margins_small.mean():.4f}  "
          f"median={np.median(margins_small):.4f}  %am (thua)={100*(margins_small<0).mean():.1f}%")
    print(f"pool CHUNG ({len(img_vecs)-pool_size} doi thu):  mean={margins_full.mean():.4f}  "
          f"median={np.median(margins_full):.4f}  %am (thua)={100*(margins_full<0).mean():.1f}%")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "embeddings_clip.pkl")
    name = sys.argv[2] if len(sys.argv) > 2 else Path(path).stem
    main(path, name)
