"""Sua lai region_siglip_done{N}.jsonl cho DUNG THUC TE (2026-08-17, bug that phat hien: shard 1
va 9 bao "da xong 100%" nhung ~64% ID trong do KHONG HE co embedding that - chac chan tu 1 su co
crash TRUOC CA phien lam viec nay, done.jsonl da sai san tu truoc). Resume logic cu TIN done.jsonl
mu quang -> khong bao gio thu lai cac ID "ma" nay, ket qua ket qua CUOI CUNG kho vuot qua ~86.5%
du chay lai bao nhieu lan.

Fix: tinh lai done_ids{N} = (pham vi that cua shard N) GIAO (detection_id THAT SU co trong file
merge region_embeddings_siglip_detection_ids.npy) - ghi DE LEN done{N}.jsonl. Sau khi chay script
nay, resume binh thuong (build_dense_region_embeddings_shard.py --shard N --num-shards 10) se tu
dong phat hien dung phan con thieu va lam lai.

Chay: python offline/repair_region_done_jsonl.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/

import json
import shutil
import sys

import numpy as np
import pandas as pd

from config import INDEX_DIR

DENSE_DIR = INDEX_DIR / "dense"
SHARD_DIR = DENSE_DIR / "shards"
SCOPE_CACHE_PATH = SHARD_DIR / "_scope_detections_cache.parquet"
MERGED_IDS_PATH = DENSE_DIR / "region_embeddings_siglip_detection_ids.npy"
NUM_SHARDS = 10


def main() -> None:
    df = pd.read_parquet(SCOPE_CACHE_PATH)
    merged_ids = set(np.load(MERGED_IDS_PATH).tolist())
    print(f"Tong embedding that su co: {len(merged_ids)}", file=sys.stderr)

    grand_total_fixed = 0
    for shard in range(NUM_SHARDS):
        shard_df = df.iloc[shard::NUM_SHARDS]
        shard_ids = set(shard_df.detection_id.tolist())
        true_done = shard_ids & merged_ids

        done_path = SHARD_DIR / f"region_siglip_done{shard}.jsonl"
        old_done: set[int] = set()
        if done_path.exists():
            with open(done_path, encoding="utf-8") as f:
                for line in f:
                    old_done.add(json.loads(line)["detection_id"])

        phantom = old_done - true_done  # danh dau "done" nhung KHONG co embedding that
        print(f"shard{shard}: pham vi {len(shard_ids)}, done.jsonl CU co {len(old_done)}, "
              f"THAT SU xong {len(true_done)}, PHANTOM (se bi xoa de lam lai) {len(phantom)}",
              file=sys.stderr)

        if phantom:
            backup_path = done_path.with_suffix(".jsonl.bak")
            if not backup_path.exists():  # chi backup 1 lan, tranh de len ban backup that
                shutil.copy2(done_path, backup_path)
            tmp_path = done_path.with_name(f"region_siglip_done{shard}_tmp.jsonl")
            with open(tmp_path, "w", encoding="utf-8") as f:
                for did in sorted(true_done):
                    f.write(json.dumps({"detection_id": did}) + "\n")
            tmp_path.replace(done_path)
            grand_total_fixed += len(phantom)

    print(f"\nDa sua xong - tong {grand_total_fixed} ID phantom da duoc go khoi done.jsonl, "
          f"se duoc chay lai o lan resume tiep theo.", file=sys.stderr)


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
