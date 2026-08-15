"""Gop cac shard region_siglip_shard{N}.npy/region_siglip_ids{N}.npy (xem
build_dense_region_embeddings_shard.py) thanh 1 file duy nhat:
  index/dense/region_embeddings_siglip.npy (vector, dim=768)
  index/dense/region_embeddings_siglip_detection_ids.npy (detection_id, RangeIndex GOC cua
    index/dense/objects_index.parquet)

Dedupe theo detection_id (uu tien dong XUAT HIEN SAU trong danh sach shard - phong truong hop
1 detection_id lo bi encode lai o 2 shard khac nhau, khong nen xay ra binh thuong vi
iloc[shard::num_shards] chia deu khong trung, nhung van dedupe cho chac).

Chay: python offline/merge_dense_region_embeddings_shards.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/

import os
import sys

import numpy as np

from config import INDEX_DIR

DENSE_DIR = INDEX_DIR / "dense"
SHARD_DIR = DENSE_DIR / "shards"
OUT_VEC_PATH = DENSE_DIR / "region_embeddings_siglip.npy"
OUT_ID_PATH = DENSE_DIR / "region_embeddings_siglip_detection_ids.npy"


def main() -> None:
    vec_paths = sorted(SHARD_DIR.glob("region_siglip_shard*.npy"))
    if not vec_paths:
        print("Khong tim thay shard nao (region_siglip_shard*.npy).", file=sys.stderr)
        return

    all_ids: dict[int, np.ndarray] = {}
    for vp in vec_paths:
        idx_str = vp.stem.replace("region_siglip_shard", "")
        id_path = SHARD_DIR / f"region_siglip_ids{idx_str}.npy"
        if not id_path.exists():
            print(f"BO QUA {vp.name} - thieu file id di kem ({id_path.name})", file=sys.stderr)
            continue
        vecs = np.load(vp)
        ids = np.load(id_path)
        print(f"  {vp.name}: {len(ids)} embedding", file=sys.stderr)
        for i, did in enumerate(ids):
            all_ids[int(did)] = vecs[i]  # dong SAU de (neu trung) - xem docstring dedupe

    if not all_ids:
        print("Khong co embedding nao de gop.", file=sys.stderr)
        return

    sorted_dids = sorted(all_ids)
    merged_ids = np.array(sorted_dids, dtype=np.int64)
    merged_vecs = np.array([all_ids[d] for d in sorted_dids], dtype=np.float32)

    # xem _atomic_write trong build_dense_region_embeddings_shard.py - np.save() tu them duoi
    # ".npy" neu ten CHUA ket thuc dung ".npy", nen tmp phai tu ket thuc bang ".npy" that su.
    tmp_v = OUT_VEC_PATH.with_name(OUT_VEC_PATH.stem + "_tmp.npy")
    tmp_i = OUT_ID_PATH.with_name(OUT_ID_PATH.stem + "_tmp.npy")
    np.save(tmp_v, merged_vecs)
    np.save(tmp_i, merged_ids)
    os.replace(tmp_v, OUT_VEC_PATH)
    os.replace(tmp_i, OUT_ID_PATH)
    print(f"Da gop: {len(merged_ids)} embedding (dim={merged_vecs.shape[1]}) -> {OUT_VEC_PATH}")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
