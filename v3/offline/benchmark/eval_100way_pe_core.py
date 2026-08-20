"""Encode 1000 anh + 100 query qua Perception Encoder (PE-Core-B16-224, Meta), luu embedding,
tinh Recall@k + margin - so voi CLIP/SigLIP tren CUNG bo du lieu.

Chay: python offline/benchmark/eval_100way_pe_core.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "share"))
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

import pickle
import time

import modal
import numpy as np

from eval_100way_utils import compute_ranks, print_report

DATA_PATH = _Path(__file__).resolve().parent / "retrieval_100way_data.pkl"
EMB_OUT_PATH = _Path(__file__).resolve().parent / "embeddings_pe_core.pkl"

BATCH = 32


def main() -> None:
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
    query_texts, pools_bytes = data["query_texts"], data["pools_bytes"]
    pool_size = len(pools_bytes[0])
    all_images_bytes = [b for pool in pools_bytes for b in pool]

    Enc = modal.Cls.from_name("aic2026-pe-core", "PECoreEncoder")
    enc = Enc()

    t0 = time.perf_counter()
    img_vecs_chunks = []
    for i in range(0, len(all_images_bytes), BATCH):
        chunk = all_images_bytes[i: i + BATCH]
        img_vecs_chunks.extend(enc.encode_images.remote(chunk))
    img_vecs = np.array(img_vecs_chunks)
    t1 = time.perf_counter()
    print(f"encode {len(all_images_bytes)} anh: {t1-t0:.1f}s")

    text_vecs_chunks = []
    for i in range(0, len(query_texts), BATCH):
        chunk = query_texts[i: i + BATCH]
        text_vecs_chunks.extend(enc.encode_texts.remote(chunk))
    text_vecs = np.array(text_vecs_chunks)
    t2 = time.perf_counter()
    print(f"encode {len(query_texts)} query text: {t2-t1:.1f}s")

    with open(EMB_OUT_PATH, "wb") as f:
        pickle.dump({"img_vecs": img_vecs, "text_vecs": text_vecs, "pool_size": pool_size}, f)
    print(f"da luu embedding -> {EMB_OUT_PATH}")

    r = compute_ranks(text_vecs, img_vecs, pool_size, len(query_texts))
    print_report("Perception Encoder (PE-Core-B16-224)", r, t1 - t0 + t2 - t1)


if __name__ == "__main__":
    main()
