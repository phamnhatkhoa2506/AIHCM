"""Chay 1 SHARD (mot phan chia deu cua so anh CON LAI) bang cach goi .remote() TUAN TU truc
tiep (KHONG qua .map()) - theo de xuat nguoi dung (2026-08-14) de vong qua nghen bang thong/
IOPS Volume dung chung ma .map() gap phai khi nhieu container chay dong thoi (da do: cang
nhieu container thi toc do/container cang giam manh).

Chay N tien trinh song song (moi tien trinh 1 shard) tu nhieu terminal/background task khac
nhau - Modal se tu dieu phoi cac container rieng cho tung tien trinh (khong dam bao 1-1 nhung
it nhat khong bi chan boi 1 cong .map() duy nhat).

Output RIENG cho tung shard (khong dung chung file voi driver .map() cu - tranh ghi de):
  index/dense/shards/objects_shard{N}.parquet
  index/dense/shards/objects_shard{N}_done.jsonl

Sau khi CA N shard chay xong, dung offline/merge_dense_objects_shards.py de gop lai.

Chay: python offline/build_dense_objects_shard.py --shard 0 --num-shards 10
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))

import argparse
import json
import os
import sys
from pathlib import Path

import modal
import pandas as pd

from config import INDEX_DIR
from dense_volume_map import to_volume_rel_path

DENSE_DIR = INDEX_DIR / "dense"
DENSE_META_PATH = DENSE_DIR / "dense_meta.parquet"
SHARD_DIR = DENSE_DIR / "shards"
MAIN_DONE_PATH = DENSE_DIR / "objects_dense_done.jsonl"  # de biet anh nao MAIN driver da xong

MAX_IMAGES_PER_CALL = 64
CHECKPOINT_EVERY = 5  # 5*64=320 anh/checkpoint


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)


def _load_done_keys(*paths: Path) -> set[tuple[str, int]]:
    keys = set()
    for p in paths:
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                keys.add((r["video_id"], r["frame_idx"]))
    return keys


def run(shard: int, num_shards: int) -> None:
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SHARD_DIR / f"objects_shard{shard}.parquet"
    done_path = SHARD_DIR / f"objects_shard{shard}_done.jsonl"

    meta = pd.read_parquet(DENSE_META_PATH)
    recs_all = meta[["video_id", "frame_idx", "path"]].to_dict("records")

    # loai bo anh MAIN driver da lam xong TRUOC khi chia shard, roi loai bo anh SHARD nay da
    # lam xong (resume) - dam bao khop dung 1 lan chia deu cho toan bo phan con lai.
    main_done = _load_done_keys(MAIN_DONE_PATH)
    remaining = [r for r in recs_all if (r["video_id"], r["frame_idx"]) not in main_done]
    shard_recs = remaining[shard::num_shards]  # chia deu kieu round-robin

    shard_done = _load_done_keys(done_path)
    before = len(shard_recs)
    shard_recs = [r for r in shard_recs if (r["video_id"], r["frame_idx"]) not in shard_done]
    print(f"[shard {shard}] Tong {before} anh cua shard nay, da xong {len(shard_done)}, "
          f"con lai {len(shard_recs)}", file=sys.stderr)
    if not shard_recs:
        print(f"[shard {shard}] Khong con gi de lam.", file=sys.stderr)
        return

    Detector = modal.Cls.from_name("aic2026-owlv2", "OWLv2Detector")
    detector = Detector()

    all_rows: list[dict] = (
        pd.read_parquet(out_path).to_dict("records") if out_path.exists() else []
    )
    n_errors = 0
    done_f = open(done_path, "a", encoding="utf-8")

    batches = [shard_recs[i:i + MAX_IMAGES_PER_CALL] for i in range(0, len(shard_recs), MAX_IMAGES_PER_CALL)]

    try:
        for ci, batch in enumerate(batches, 1):
            rel_paths = [to_volume_rel_path(r["path"]) for r in batch]
            try:
                results = detector.detect_batch_from_volume.remote(rel_paths)
            except Exception as e:
                n_errors += len(batch)
                print(f"[shard {shard}] [LOI] batch {ci}/{len(batches)}: {type(e).__name__} {str(e)[:120]}",
                      file=sys.stderr)
                continue

            for rec, dets in zip(batch, results):
                for d in dets:
                    if d["score"] < 0.15:
                        continue
                    all_rows.append({
                        "video_id": rec["video_id"], "frame_idx": rec["frame_idx"],
                        "label": d["label"], "score": d["score"],
                        "ymin": d["ymin"], "xmin": d["xmin"],
                        "ymax": d["ymax"], "xmax": d["xmax"], "source": "owlv2",
                    })
                done_f.write(json.dumps({"video_id": rec["video_id"], "frame_idx": rec["frame_idx"]},
                                         ensure_ascii=False) + "\n")
            done_f.flush()
            os.fsync(done_f.fileno())

            if ci % CHECKPOINT_EVERY == 0 or ci == len(batches):
                _atomic_write_parquet(pd.DataFrame(all_rows), out_path)
                done_count = min(ci * MAX_IMAGES_PER_CALL, len(shard_recs))
                print(f"[shard {shard}] [{done_count}/{len(shard_recs)}] checkpoint {len(all_rows)} "
                      f"detection, {n_errors} loi -> {out_path}", file=sys.stderr)
    finally:
        done_f.close()

    _atomic_write_parquet(pd.DataFrame(all_rows), out_path)
    print(f"[shard {shard}] Da luu {len(all_rows)} detection, {n_errors} loi -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    run(args.shard, args.num_shards)
