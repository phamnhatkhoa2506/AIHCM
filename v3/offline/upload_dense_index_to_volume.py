"""Upload 3 model embedding + FAISS index + dense_meta cần cho server Modal
(offline/modal_infra/dense_index_app.py) lên Volume "aic2026-dense-index":
  - siglip_matrix.npy / siglip_faiss.index
  - pe_core_matrix.npy / pe_core_faiss.index
  - beit3_matrix.npy / beit3_faiss.index
  - dense_meta.parquet

2026-08-16 (theo yêu cầu người dùng: "máy mình mỗi lần chạy lên nó chiếm gần hết RAM ... bỏ
lên modal") — 3 file matrix + 3 file faiss (~7.4GB, thường xuyên nạp CẢ 3 model khi dùng mode
"All"/rrf) là phần RAM nặng nhất còn lại trên máy local, giống hệt lý do
upload_region_index_to_volume.py ra đời trước đó (Region-CLIP, 5.3GB).

Chạy lại script này MỖI LẦN có embedding mới (build_dense_embeddings.py chạy lại) — server
KHÔNG tự động đồng bộ.

Chạy: python offline/upload_dense_index_to_volume.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/

import sys

import modal

from config import DENSE_DIR, DENSE_META_PATH, INDEX_DIR

MODELS = ("siglip", "pe_core", "beit3")

VOLUME_NAME = "aic2026-dense-index"


def main() -> None:
    files: list[tuple[_Path, str]] = [(DENSE_META_PATH, "/dense_meta.parquet")]
    for model in MODELS:
        files.append((DENSE_DIR / f"{model}_matrix.npy", f"/{model}_matrix.npy"))
        files.append((DENSE_DIR / f"{model}_faiss.index", f"/{model}_faiss.index"))

    for p, _ in files:
        if not p.exists():
            print(f"LOI: khong tim thay {p}", file=sys.stderr)
            sys.exit(1)

    vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    with vol.batch_upload(force=True) as batch:
        for p, dest in files:
            batch.put_file(str(p), dest)
    print(f"Da upload {len(files)} file len Volume '{VOLUME_NAME}'.")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
