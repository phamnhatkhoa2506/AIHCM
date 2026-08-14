"""Chuan bi du lieu chung cho viec so sanh CLIP hien tai vs SigLIP/PE-Core/VL-JEPA.

SUA 2026-08-11 (bug phuong phap phat hien ngay sau khi do CLIP baseline): ban dau chi lay
100 anh THUOC 100 VIDEO KHAC NHAU lam candidate pool chung - CLIP dat Recall@1=1.000 hoan hao,
nhung do la vi 100 canh nay qua khac biet nhau (salad vs hong hac vs phuong trinh...), KHONG
mo phong dung bai toan kho THAT (nhieu frame GIONG NHAU trong CUNG 1 video canh tranh - dung
"score clustering" da do sau truoc do voi kis_001). Test cu se cho MOI model ~1.0, vo nghia de
so sanh.

SUA: voi MOI candidate, them 9 frame KHAC cung video lam distractor - moi query gio phai phan
biet dung frame trong so 10 frame RAT GIONG NHAU (cung video, cung boi canh/nhan vat) thay vi
so voi ca 100 canh khac nhau. Day moi la dung bai toan da lam CLIP that bai truoc do.

SUA LAN 2 (2026-08-11, van con qua de - Recall@1 van 1.000): chon distractor NGAU NHIEN trong
CA VIDEO khong du - 1 video dai (10-45 phut) co THE co nhieu CANH khac nhau (intro/che bien/
phong van...), frame ngau nhien co the thuoc canh hoan toan khac GT, khac gi frame video khac.
Doi sang chon distractor GAN GT VE THOI GIAN NHAT (theo local_idx) - dung kieu canh tranh that
da do voi kis_001 (frame 77 vs 78, sat nhau, CUNG canh, chi khac chi tiet nho).

Chay: python offline/benchmark/build_100way_retrieval.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "share"))

import json
import pickle
import random

import pandas as pd

from config import INDEX_DIR
from keyframe_images import read_keyframe_bytes

MANIFEST_PATH = _Path(__file__).resolve().parent / "candidate_manifest.json"
QUERIES_PATH = _Path(__file__).resolve().parent / "queries_100.jsonl"
OUT_PATH = _Path(__file__).resolve().parent / "retrieval_100way_data.pkl"

N_DISTRACTORS = 9  # +GT = 10 frame/pool, deu CUNG VIDEO - mo phong dung "score clustering"
random.seed(42)


def main() -> None:
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    kis_queries = [json.loads(line) for line in open(QUERIES_PATH, encoding="utf-8") if line.strip()]
    kis_queries = [q for q in kis_queries if q["type"] == "KIS"]
    assert len(kis_queries) == len(manifest) == 100

    meta = pd.read_parquet(INDEX_DIR / "meta.parquet")

    query_texts = []
    pools_bytes: list[list[bytes]] = []  # 1 list/query, index 0 = GT, con lai la distractor
    pools_meta: list[list[dict]] = []
    for cand, q in zip(manifest, kis_queries):
        vid, li = cand["video_id"], cand["local_idx"]
        g = meta[meta["video_id"] == vid]
        other_idx = [int(x) for x in g["local_idx"] if int(x) != li]
        # GAN NHAT theo local_idx (khong phai ngau nhien) - mo phong dung "score clustering"
        # that: frame lien ke, cung canh, chi khac chi tiet nho -> canh tranh KHO nhat.
        other_idx.sort(key=lambda x: abs(x - li))
        n_take = min(N_DISTRACTORS, len(other_idx))
        distractor_idx = other_idx[:n_take]

        pool_bytes = [read_keyframe_bytes(vid, li)]
        pool_meta = [{"video_id": vid, "local_idx": li, "is_gt": True}]
        for di in distractor_idx:
            pool_bytes.append(read_keyframe_bytes(vid, di))
            pool_meta.append({"video_id": vid, "local_idx": di, "is_gt": False})

        query_texts.append(q["query"])
        pools_bytes.append(pool_bytes)
        pools_meta.append(pool_meta)

    with open(OUT_PATH, "wb") as f:
        pickle.dump({
            "query_texts": query_texts,
            "pools_bytes": pools_bytes,  # list[list[bytes]] - 1 pool/query, GT o index 0
            "pools_meta": pools_meta,
            "manifest": manifest,
        }, f)
    sizes = [len(p) for p in pools_bytes]
    print(f"da luu {len(query_texts)} query, pool size {min(sizes)}-{max(sizes)} (muc tieu {N_DISTRACTORS+1}) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
