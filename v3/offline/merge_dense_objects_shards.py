"""Gop ket qua tu cac shard (offline/build_dense_objects_shard.py) vao objects_index.parquet +
objects_dense_done.jsonl chinh. Chay SAU KHI tat ca shard da chay xong (hoac chay giua chung de
gop tam thoi cung duoc - AN TOAN, khong ghi de shard data).

Chay: python offline/merge_dense_objects_shards.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))

import json
import os
import sys
from pathlib import Path

import pandas as pd

from config import INDEX_DIR

DENSE_DIR = INDEX_DIR / "dense"
SHARD_DIR = DENSE_DIR / "shards"
OUT_PATH = DENSE_DIR / "objects_index.parquet"
DONE_PATH = DENSE_DIR / "objects_dense_done.jsonl"


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)


def run() -> None:
    shard_parquets = sorted(SHARD_DIR.glob("objects_shard*.parquet"))
    shard_dones = sorted(SHARD_DIR.glob("objects_shard*_done.jsonl"))
    print(f"Tim thay {len(shard_parquets)} shard parquet, {len(shard_dones)} shard done log.",
          file=sys.stderr)

    main_df = pd.read_parquet(OUT_PATH) if OUT_PATH.exists() else pd.DataFrame()
    main_done_keys = set()
    if DONE_PATH.exists():
        with open(DONE_PATH, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                main_done_keys.add((r["video_id"], r["frame_idx"]))

    all_dfs = [main_df] if len(main_df) else []
    for p in shard_parquets:
        df = pd.read_parquet(p)
        if len(df):
            all_dfs.append(df)
        print(f"  {p.name}: {len(df)} detection", file=sys.stderr)

    merged_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    if len(merged_df):
        merged_df = merged_df.drop_duplicates(subset=["video_id", "frame_idx", "label",
                                                        "xmin", "ymin", "xmax", "ymax"])
    _atomic_write_parquet(merged_df, OUT_PATH)

    new_keys = set(main_done_keys)
    with open(DONE_PATH, "a", encoding="utf-8") as done_f:
        for p in shard_dones:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    key = (r["video_id"], r["frame_idx"])
                    if key not in new_keys:
                        new_keys.add(key)
                        done_f.write(line if line.endswith("\n") else line + "\n")

    print(f"Da gop: {len(merged_df)} detection tong, {len(new_keys)} anh da xong -> "
          f"{OUT_PATH}, {DONE_PATH}", file=sys.stderr)


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    run()
