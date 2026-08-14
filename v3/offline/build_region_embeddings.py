"""Precompute region-CLIP embedding cho object trong objects_index.parquet (closed-set) VA
open_vocab_detections.parquet (Grounding DINO) — encode qua Modal GPU
(`modal_infra/region_clip_app.py::Encoder`), KHONG dung CPU local (da do: CPU may nay chi
~4.5 box/s, qua cham).

Local chi lam phan CROP anh (doc zip + decode JPEG - da do ~16.8 box/s don luong, thuan I/O)
- song song hoa bang ThreadPoolExecutor giong pattern build_spatial_edges.py/build_objects_index.py,
roi gui batch bytes anh da crop len Modal (khong gui frame goc).

Checkpoint/resume: bo qua detection_id da co trong file ket qua khi chay lai.

Mac dinh SCOPE_CATEGORIES (closed-set) = person + animal (449,356 object, 40% tong - doi tuong
hay bi hoi thuoc tinh nhat). Doi SCOPE_CATEGORIES=None de lam toan bo 1.1 trieu object closed
(lau hon nhieu).

BO SUNG 2026-08-07 (bug that phat hien: query "con lan mau vang" -> Region-CLIP mat 46s vi
0/147 box dung duoc embedding san, phai fallback CPU): open_vocab_detections.parquet (Grounding
DINO, 87,649 object) truoc day KHONG nam trong pham vi precompute nao ca - nhan moi phat hien
(vd "lion dance costume") luon roi vao nhanh cham. INCLUDE_OPEN_VOCAB=True bay gio LUON precompute
HET open_vocab (khong loc theo category, vi day la tap nho, khong can gioi han nhu closed-set).
detection_id cua open_vocab = N_CLOSED_TOTAL + vi_tri_dong_trong_file - dung KHOP voi cach
resources.py gop objects_index (concat closed roi noi open_vocab vao sau, ignore_index=True)."""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/

import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import modal
import numpy as np
import pandas as pd

from config import INDEX_DIR, OBJECTS_INDEX_PATH
from region_clip import crop_region

SCOPE_CATEGORIES: set[str] | None = {"person", "animal"}
INCLUDE_OPEN_VOCAB = True
OPEN_VOCAB_DETECTIONS_PATH = INDEX_DIR / "open_vocab_detections.parquet"
BATCH_SIZE = 64
N_CROP_WORKERS = 32
CHECKPOINT_EVERY = 20000

EMBEDDINGS_PATH = INDEX_DIR / "region_embeddings.npy"
DETECTION_IDS_PATH = INDEX_DIR / "region_embeddings_detection_ids.npy"


def _load_scope_detections(limit: int | None = None) -> pd.DataFrame:
    closed_full = pd.read_parquet(OBJECTS_INDEX_PATH).reset_index(drop=False).rename(columns={"index": "detection_id"})
    n_closed_total = len(closed_full)  # offset cho detection_id cua open_vocab - PHAI tinh TRUOC khi loc scope

    df = closed_full
    if SCOPE_CATEGORIES is not None:
        with open(INDEX_DIR / "label_types.json", encoding="utf-8") as f:
            label_types = json.load(f)
        labels_in_scope = {lb for lb, cats in label_types.items() if set(cats) & SCOPE_CATEGORIES}
        df = df[df.label.isin(labels_in_scope)]

    if INCLUDE_OPEN_VOCAB and OPEN_VOCAB_DETECTIONS_PATH.exists():
        ov = pd.read_parquet(OPEN_VOCAB_DETECTIONS_PATH).reset_index(drop=False).rename(columns={"index": "_pos"})
        ov["detection_id"] = n_closed_total + ov["_pos"]
        ov = ov.drop(columns=["_pos"])
        df = pd.concat([df, ov], ignore_index=True)

    df = df.reset_index(drop=True)
    return df.head(limit) if limit else df


def _crop_bytes(video_id: str, local_idx: int, box: tuple) -> bytes:
    crop = crop_region(video_id, local_idx, box)
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _load_existing() -> tuple[np.ndarray, np.ndarray]:
    if EMBEDDINGS_PATH.exists() and DETECTION_IDS_PATH.exists():
        return np.load(EMBEDDINGS_PATH), np.load(DETECTION_IDS_PATH)
    return np.zeros((0, 512), dtype=np.float32), np.zeros((0,), dtype=np.int64)


def _save(vecs: np.ndarray, ids: np.ndarray) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, vecs)
    np.save(DETECTION_IDS_PATH, ids)


def main(limit: int | None = None) -> None:
    df = _load_scope_detections(limit)
    old_vecs, old_ids = _load_existing()
    done_ids = set(old_ids.tolist())
    print(f"Da co san {len(done_ids)} embedding tu lan chay truoc", file=sys.stderr)

    todo = df[~df.detection_id.isin(done_ids)].reset_index(drop=True)
    print(f"Pham vi: {len(df)} object, con lai {len(todo)} can encode", file=sys.stderr)
    if todo.empty:
        print("Khong con gi de lam.", file=sys.stderr)
        return

    Encoder = modal.Cls.from_name("aic2026-region-clip", "Encoder")
    encoder = Encoder()

    rows = list(todo.itertuples(index=False))
    crops_bytes: list[bytes | None] = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=N_CROP_WORKERS) as pool:
        futures = {
            pool.submit(_crop_bytes, r.video_id, int(r.local_idx), (r.ymin, r.xmin, r.ymax, r.xmax)): i
            for i, r in enumerate(rows)
        }
        for i, fut in enumerate(as_completed(futures), 1):
            crops_bytes[futures[fut]] = fut.result()
            if i % 5000 == 0 or i == len(rows):
                print(f"[crop {i}/{len(rows)}]", file=sys.stderr)

    batches = [crops_bytes[i : i + BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]
    id_batches = [
        [r.detection_id for r in rows[i : i + BATCH_SIZE]] for i in range(0, len(rows), BATCH_SIZE)
    ]

    print(f"Goi Modal (aic2026-region-clip/Encoder): {len(batches)} batch, {BATCH_SIZE} box/batch...", file=sys.stderr)
    all_vecs: list[list[float]] = []
    all_ids: list[int] = []
    n_done = 0
    n_batches_done = 0
    last_checkpoint = 0
    for i, vecs in enumerate(encoder.encode_batch.map(batches)):
        all_vecs.extend(vecs)
        all_ids.extend(id_batches[i])
        n_done += len(vecs)
        n_batches_done += 1
        print(f"  [Modal batch {n_batches_done}/{len(batches)}] -> {n_done}/{len(rows)} object encoded", file=sys.stderr)

        if n_done - last_checkpoint >= CHECKPOINT_EVERY:
            last_checkpoint = n_done
            print(f"[encode {n_done}/{len(rows)}] checkpoint...", file=sys.stderr)
            merged_vecs = np.concatenate([old_vecs, np.array(all_vecs, dtype=np.float32)])
            merged_ids = np.concatenate([old_ids, np.array(all_ids, dtype=np.int64)])
            _save(merged_vecs, merged_ids)

    merged_vecs = np.concatenate([old_vecs, np.array(all_vecs, dtype=np.float32)])
    merged_ids = np.concatenate([old_ids, np.array(all_ids, dtype=np.int64)])
    _save(merged_vecs, merged_ids)
    print(f"Xong: {len(merged_ids)} embedding (tong, gom ca cu) -> {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit)
