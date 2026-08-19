"""Precompute Region-CLIP (SigLIP2) cho object trong bo dense (OWLv2 closed-set, xem
index/dense/objects_index.parquet) - CHAY THEO SHARD (goi .remote() TUAN TU, KHONG qua
.map()) - dung dung pattern da xac nhan nhanh hon ~8x cho OWLv2 dense detection (xem
[[feedback_modal_map_vs_sharded_remote]] trong memory, hoi thoai 2026-08-14).

SCOPE (2026-08-15, theo yeu cau nguoi dung - 14.5 trieu detection qua nhieu de encode het,
va muon da dang hon person/animal):
  - Category: person + animal + vehicle + clothing_accessory + surface (BO qua "container" -
    it khi la doi tuong bi hoi thuoc tinh mau/kieu dang) - xem index/label_types.json.
  - score >= 0.25 (RIENG cho region-embedding, KHONG doi threshold hard-filter goc 0.15 cua
    objects_index.parquet).
  -> ~2.61 trieu object (18% tong 14.5M) - gap ~5.8x scope cu tren BTC (449,356 object,
  person+animal, BTC it anh hon).

Model: SigLIP2 (aic2026-siglip/SigLIPEncoder.encode_images, xem modal_infra/siglip_app.py) -
CHON SigLIP2 TRUOC (nguoi dung yeu cau) vi thang benchmark ro ret tren bo dense (R@1=0.30 vs
CLIP goc 0.16) - PE-Core/BEiT-3 se lam sau neu can, dung lai script nay doi MODEL_APP_NAME.

Crop anh: doc THANG tu dia local (dense_meta.parquet "path") - KHAC BTC (phai giai nen zip),
don gian hon nhieu, khong can ThreadPoolExecutor rieng cho I/O zip nhu build_region_embeddings.py.

Output RIENG cho tung shard (index/dense/shards/):
  region_siglip_shard{N}.npy (vector, dim=768) + region_siglip_ids{N}.npy (detection_id, tra
  ve theo RangeIndex GOC cua objects_index.parquet, KHONG doi khi reset_index) +
  region_siglip_done{N}.jsonl (resume).

Chay: python offline/build_dense_region_embeddings_shard.py --shard 0 --num-shards 10
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/

import argparse
import io
import json
import os
import sys

import modal
import numpy as np
import pandas as pd
from PIL import Image

from config import DENSE_DIR, DENSE_META_PATH, INDEX_DIR

DENSE_DIR = INDEX_DIR / "dense"
OBJECTS_INDEX_PATH = DENSE_DIR / "objects_index.parquet"
DENSE_META_PATH = DENSE_DIR / "dense_meta.parquet"
SHARD_DIR = DENSE_DIR / "shards"
LABEL_TYPES_PATH = INDEX_DIR / "label_types.json"

SCOPE_CATEGORIES = {"person", "animal", "vehicle", "clothing_accessory", "surface"}
SCORE_THRESHOLD = 0.25
EMBED_DIM = 768  # SigLIP2 base patch16-224 hidden size (KHAC 512 cua CLIP-ViT-B-32 cu)

MODAL_APP_NAME = "aic2026-siglip"
MODAL_CLASS_NAME = "SigLIPEncoder"
MODAL_METHOD_NAME = "encode_images"

BATCH_SIZE = 64
CHECKPOINT_EVERY = 10  # 10*64 = 640 object/checkpoint


SCOPE_CACHE_PATH = SHARD_DIR / "_scope_detections_cache.parquet"


def _load_scope_detections() -> pd.DataFrame:
    """detection_id = vi tri dong GOC trong objects_index.parquet (RangeIndex, TRUOC khi loc
    scope) - dung khop voi cach resources.py/query_planner.py tra cuu region_embeddings theo
    detection_id (xem build_region_embeddings.py ban BTC, cung nguyen tac).

    BUG THAT (2026-08-15, phat hien khi chay that: shard crash ArrayMemoryError NGAY CA sau
    khi da fix memory o _atomic_write): MOI trong 10 process TU DOC LAI toan bo 14.5 trieu
    dong objects_index.parquet (~vai GB/process) chi de loc con ~2.6 trieu dong - 10 process
    song song nhan len thanh chuc GB RAM cung luc, day may vao trang thai thieu RAM du chi
    can cap phat vai tram MB. Fix: cache ket qua DA LOC (nho hon ~5.6x) ra 1 file rieng, doc
    file NHO nay cho CAC LAN SAU thay vi doc lai file goc."""
    if SCOPE_CACHE_PATH.exists():
        return pd.read_parquet(SCOPE_CACHE_PATH)

    df = pd.read_parquet(OBJECTS_INDEX_PATH).reset_index(drop=False).rename(columns={"index": "detection_id"})
    with open(LABEL_TYPES_PATH, encoding="utf-8") as f:
        label_types = json.load(f)
    labels_in_scope = {lb for lb, cats in label_types.items() if set(cats) & SCOPE_CATEGORIES}
    df = df[df.label.isin(labels_in_scope) & (df.score >= SCORE_THRESHOLD)]

    meta = pd.read_parquet(DENSE_META_PATH)[["video_id", "frame_idx", "path"]]
    df = df.merge(meta, on=["video_id", "frame_idx"], how="inner").reset_index(drop=True)

    # ghi cache ATOMIC (tranh file dang ghi do bi doc nham boi shard khac chay dong thoi) -
    # neu 2 process cung race vao day (chua co cache) thi ai ghi sau se de len, khong sao vi
    # noi dung giong het nhau (deterministic tu cung input).
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SCOPE_CACHE_PATH.with_name(SCOPE_CACHE_PATH.stem + "_tmp.parquet")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, SCOPE_CACHE_PATH)
    return df


def _crop_bytes(path: str, box: tuple) -> bytes:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    ymin, xmin, ymax, xmax = box
    left, top = max(0, int(xmin * w)), max(0, int(ymin * h))
    right, bottom = min(w, int(xmax * w)), min(h, int(ymax * h))
    right, bottom = max(right, left + 1), max(bottom, top + 1)  # box qua nho -> ep >=1px
    crop = img.crop((left, top, right, bottom))
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _atomic_write(vecs: np.ndarray, ids: np.ndarray, vec_path, id_path) -> None:
    # BUG THAT (2026-08-15): np.save() TU DONG them duoi ".npy" neu ten file CHUA ket thuc
    # dung ".npy" - dat ten tmp la "*.npy.tmp" bi numpy bien thanh "*.npy.tmp.npy" (sai ten,
    # os.replace() sau do khong tim thay). Fix: tmp phai TU KET THUC bang ".npy" that su.
    tmp_v = vec_path.with_name(vec_path.stem + "_tmp.npy")
    tmp_i = id_path.with_name(id_path.stem + "_tmp.npy")
    np.save(tmp_v, vecs)
    np.save(tmp_i, ids)
    os.replace(tmp_v, vec_path)
    os.replace(tmp_i, id_path)


def run(shard: int, num_shards: int) -> None:
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    out_vec_path = SHARD_DIR / f"region_siglip_shard{shard}.npy"
    out_id_path = SHARD_DIR / f"region_siglip_ids{shard}.npy"
    done_path = SHARD_DIR / f"region_siglip_done{shard}.jsonl"

    df = _load_scope_detections()
    shard_df = df.iloc[shard::num_shards].reset_index(drop=True)

    done_ids: set[int] = set()
    if done_path.exists():
        with open(done_path, encoding="utf-8") as f:
            for line in f:
                done_ids.add(json.loads(line)["detection_id"])

    todo = shard_df[~shard_df.detection_id.isin(done_ids)].reset_index(drop=True)
    print(f"[shard {shard}] Pham vi {len(shard_df)} object, da xong {len(done_ids)}, "
          f"con lai {len(todo)}", file=sys.stderr)
    if todo.empty:
        print(f"[shard {shard}] Khong con gi de lam.", file=sys.stderr)
        return

    Encoder = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)
    encoder = Encoder()
    encode_fn = getattr(encoder, MODAL_METHOD_NAME)

    old_vecs = np.load(out_vec_path) if out_vec_path.exists() else np.zeros((0, EMBED_DIM), dtype=np.float32)
    old_ids = np.load(out_id_path) if out_id_path.exists() else np.zeros((0,), dtype=np.int64)

    # BUG THAT (2026-08-15, phat hien khi chay that: 4/10 shard bi ArrayMemoryError giua chung)
    # - giu vector duoi dang list Python long nhau (moi vec la list 768 float object rieng, tu
    # Modal JSON response) TON RAM GAP ~7x so voi giu THANG numpy float32 array (moi float
    # object CPython ~28 byte + overhead list, thay vi 4 byte lien tuc trong ndarray) - nhan
    # voi 10 process chay song song lam qua tai RAM may that. Fix: convert NGAY tung batch
    # thanh numpy array nho, gop bang np.concatenate() (khong np.array(list_long_python)).
    vec_chunks: list[np.ndarray] = [old_vecs]
    id_chunks: list[np.ndarray] = [old_ids]
    n_since_checkpoint = 0

    n_errors = 0
    done_f = open(done_path, "a", encoding="utf-8")
    batches = [todo.iloc[i:i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]

    try:
        for ci, batch in enumerate(batches, 1):
            crops = [_crop_bytes(row.path, (row.ymin, row.xmin, row.ymax, row.xmax))
                     for row in batch.itertuples(index=False)]
            try:
                vecs = encode_fn.remote(crops)
            except Exception as e:
                n_errors += len(batch)
                print(f"[shard {shard}] [LOI] batch {ci}/{len(batches)}: "
                      f"{type(e).__name__} {str(e)[:120]}", file=sys.stderr)
                continue
            batch_ids = [int(row.detection_id) for row in batch.itertuples(index=False)]
            vec_chunks.append(np.asarray(vecs, dtype=np.float32))
            id_chunks.append(np.asarray(batch_ids, dtype=np.int64))
            n_since_checkpoint += len(batch_ids)
            for did in batch_ids:
                done_f.write(json.dumps({"detection_id": did}) + "\n")
            done_f.flush()
            os.fsync(done_f.fileno())
            if ci % CHECKPOINT_EVERY == 0 or ci == len(batches):
                # gop cac chunk lai thanh 1 array roi RESET danh sach chunk ve dung 1 phan tu
                # (chinh no) - tranh danh sach chunk phinh to VO HAN qua nhieu checkpoint (10
                # shard chay lau, hang nghin batch) trong khi van it ton hon list-of-float cu.
                merged_vecs = np.concatenate(vec_chunks) if len(vec_chunks) > 1 else vec_chunks[0]
                merged_ids = np.concatenate(id_chunks) if len(id_chunks) > 1 else id_chunks[0]
                _atomic_write(merged_vecs, merged_ids, out_vec_path, out_id_path)
                vec_chunks, id_chunks = [merged_vecs], [merged_ids]
                done_count = min(ci * BATCH_SIZE, len(todo))
                print(f"[shard {shard}] [{done_count}/{len(todo)}] checkpoint {len(merged_ids)} "
                      f"embedding, {n_errors} loi -> {out_vec_path}", file=sys.stderr)
    finally:
        done_f.close()
    final_vecs = np.concatenate(vec_chunks) if len(vec_chunks) > 1 else vec_chunks[0]
    final_ids = np.concatenate(id_chunks) if len(id_chunks) > 1 else id_chunks[0]
    _atomic_write(final_vecs, final_ids, out_vec_path, out_id_path)
    print(f"[shard {shard}] Da luu {len(final_ids)} embedding, {n_errors} loi -> {out_vec_path}", file=sys.stderr)


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
