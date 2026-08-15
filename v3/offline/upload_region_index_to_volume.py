"""Upload 3 file can thiet cho Region-CLIP rerank server (offline/modal_infra/region_rerank_app.py)
len Modal Volume "aic2026-region-index":
  - scope_detections_cache.parquet  (tu index/dense/shards/_scope_detections_cache.parquet,
    2.6 trieu dong - CHI cac object trong scope Region-CLIP, NHO HON nhieu so voi
    objects_index.parquet goc 14.5 trieu dong - xem build_dense_region_embeddings_shard.py)
  - region_embeddings_siglip.npy (vector, 5.3GB)
  - region_embeddings_siglip_detection_ids.npy

Chay lai script nay MOI LAN sau khi chay tiep job build_dense_region_embeddings_shard.py (embedding
moi) + merge_dense_region_embeddings_shards.py - de server tren Modal luon dung du lieu MOI NHAT
(khong tu dong dong bo, phai upload lai thu cong).

Chay: python offline/upload_region_index_to_volume.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/

import sys

import modal

from config import INDEX_DIR

DENSE_DIR = INDEX_DIR / "dense"
SCOPE_CACHE_PATH = DENSE_DIR / "shards" / "_scope_detections_cache.parquet"
REGION_EMB_PATH = DENSE_DIR / "region_embeddings_siglip.npy"
REGION_EMB_IDS_PATH = DENSE_DIR / "region_embeddings_siglip_detection_ids.npy"

VOLUME_NAME = "aic2026-region-index"


def main() -> None:
    for p in (SCOPE_CACHE_PATH, REGION_EMB_PATH, REGION_EMB_IDS_PATH):
        if not p.exists():
            print(f"LOI: khong tim thay {p}", file=sys.stderr)
            sys.exit(1)

    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    with vol.batch_upload(force=True) as batch:
        batch.put_file(str(SCOPE_CACHE_PATH), "/scope_detections_cache.parquet")
        batch.put_file(str(REGION_EMB_PATH), "/region_embeddings_siglip.npy")
        batch.put_file(str(REGION_EMB_IDS_PATH), "/region_embeddings_siglip_detection_ids.npy")
    print(f"Da upload 3 file len Volume '{VOLUME_NAME}'.")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
