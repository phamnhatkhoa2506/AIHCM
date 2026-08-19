"""OCR cho bo keyframe TU TRICH mat do cao (369.589 anh, xem merge_dense_embeddings.py) -
dung LAI NGUYEN Modal app PaddleOCR (modal_infra/ocr_app.py) da co, chi khac driver:
  - Doc anh tu DUONG DAN LOCAL (index/dense/dense_meta.parquet cot "path"), khong qua zip BTC.
  - KHONG gom "run" theo khoang cach frame co dinh nhu build_ocr_index.py (RUN_GAP_THRESHOLD=20
    hieu chinh cho stride=15 CO DINH cua sample thua sparse) - frame dense KHONG deu (theo
    shot-detection, khoang cach tu vai frame den hang tram frame) - nguong co dinh se sai o day.
    Luu THANG 1 dong/dong chu/frame (khong merge) - moi frame dense DA LA 1 ung vien that su
    hien thi cho nguoi dung (khac sparse phai suy ra frame giua 2 mocsample), nen khong can
    "bu" khoang trong bang merge nua.

Schema OUT_PATH (index/dense/ocr_text.parquet): video_id, frame_idx, text_raw, text_norm,
ymin, xmin, ymax, xmax, score - dung boi share/tiers/dense_search.py cho hard-filter OCR.

QUY MO LON HON HAN OCR sparse cu (369k anh vs ~vai chuc nghin frame sample thua truoc) - chay
co the mat NHIEU GIO du song song 8 container (PaddleOCR khong batch GPU that, xem ocr_app.py).
Co checkpoint/resume (ocr_dense_done.jsonl) - an toan khi bi ngat giua chung.

Chay: python offline/build_dense_ocr_index.py
      python offline/build_dense_ocr_index.py --limit 500   # test nhanh
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal
import pandas as pd

from atomic_io import atomic_write_parquet

from config import DENSE_DIR, DENSE_META_PATH, INDEX_DIR
from tiers.tier1_filter import _strip_accents

OUT_PATH = DENSE_DIR / "ocr_text.parquet"
DONE_PATH = DENSE_DIR / "ocr_dense_done.jsonl"

MIN_SCORE = 0.5  # giong het build_ocr_index.py - loc detection do tin cay thap
MAX_IMAGES_PER_CALL = 16
CHECKPOINT_EVERY_CHUNKS = 20
N_READ_WORKERS = 32


def _load_done_keys() -> set[tuple[str, int]]:
    if not DONE_PATH.exists():
        return set()
    keys = set()
    with open(DONE_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            keys.add((r["video_id"], r["frame_idx"]))
    return keys


def run(limit: int | None = None) -> None:
    meta = pd.read_parquet(DENSE_META_PATH)
    if limit:
        meta = meta.head(limit)
    recs = meta[["video_id", "frame_idx", "path"]].to_dict("records")
    print(f"Tong {len(recs)} anh dense can OCR", file=sys.stderr)

    done_keys = _load_done_keys()
    if done_keys:
        before = len(recs)
        recs = [r for r in recs if (r["video_id"], r["frame_idx"]) not in done_keys]
        print(f"Resume: da xong {len(done_keys)} anh tu lan chay truoc, "
              f"con lai {len(recs)}/{before} can xu ly", file=sys.stderr)
    if not recs:
        print("Khong con gi de lam.", file=sys.stderr)
        return

    def _read_one(rec: dict) -> bytes:
        return Path(rec["path"]).read_bytes()

    OCRDetector = modal.Cls.from_name("aic2026-ocr", "OCRDetector")
    detector = OCRDetector()

    all_rows: list[dict] = (
        pd.read_parquet(OUT_PATH).to_dict("records") if OUT_PATH.exists() else []
    )

    n_errors = 0
    done_f = open(DONE_PATH, "a", encoding="utf-8")
    read_pool = ThreadPoolExecutor(max_workers=N_READ_WORKERS)

    batches = [recs[i:i + MAX_IMAGES_PER_CALL] for i in range(0, len(recs), MAX_IMAGES_PER_CALL)]

    def _read_batch(batch_recs: list[dict]) -> list[bytes]:
        return [_read_one(r) for r in batch_recs]

    # prefetch nho (giong build_dense_embeddings.py sau khi sua bug MemoryError doc het 1 luc)
    # - khong nap toan bo 369k anh vao RAM cung luc.
    futures_iter = iter(read_pool.submit(_read_batch, b) for b in batches)
    pending: list = []
    for _ in range(min(N_READ_WORKERS + 2, len(batches))):
        try:
            pending.append(next(futures_iter))
        except StopIteration:
            break

    def _batches_bytes_gen():
        nonlocal pending
        while pending:
            fut = pending.pop(0)
            try:
                pending.append(next(futures_iter))
            except StopIteration:
                pass
            yield fut.result()

    try:
        map_results = detector.ocr_batch.map(
            _batches_bytes_gen(), order_outputs=True, return_exceptions=True, wrap_returned_exceptions=False
        )
        for ci, (results, batch) in enumerate(zip(map_results, batches), 1):
            if isinstance(results, Exception):
                n_errors += len(batch)
                print(f"  [LOI] batch {ci}/{len(batches)}: {type(results).__name__} {str(results)[:120]}",
                      file=sys.stderr)
                continue

            for rec, lines in zip(batch, results):
                for ln in lines:
                    if ln["score"] < MIN_SCORE:
                        continue
                    all_rows.append({
                        "video_id": rec["video_id"],
                        "frame_idx": rec["frame_idx"],
                        "text_raw": ln["text"],
                        "text_norm": _strip_accents(ln["text"]),
                        "score": ln["score"],
                        "ymin": ln["ymin"], "xmin": ln["xmin"],
                        "ymax": ln["ymax"], "xmax": ln["xmax"],
                    })
                done_f.write(json.dumps({"video_id": rec["video_id"], "frame_idx": rec["frame_idx"]},
                                         ensure_ascii=False) + "\n")
            done_f.flush()
            os.fsync(done_f.fileno())

            if ci % CHECKPOINT_EVERY_CHUNKS == 0 or ci == len(batches):
                atomic_write_parquet(pd.DataFrame(all_rows), OUT_PATH, index=False)
                done_count = min(ci * MAX_IMAGES_PER_CALL, len(recs))
                print(f"[{done_count}/{len(recs)}] checkpoint {len(all_rows)} dong chu, "
                      f"{n_errors} loi -> {OUT_PATH}", file=sys.stderr)
    finally:
        done_f.close()
        read_pool.shutdown(wait=False)

    atomic_write_parquet(pd.DataFrame(all_rows), OUT_PATH, index=False)
    print(f"\nDa luu {len(all_rows)} dong chu, {n_errors} loi -> {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="chi OCR N anh dau (test nhanh)")
    args = parser.parse_args()
    run(limit=args.limit)
