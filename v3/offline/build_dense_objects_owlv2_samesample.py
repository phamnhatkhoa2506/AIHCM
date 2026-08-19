"""So sanh cong bang voi baseline Faster R-CNN Inception-ResNet-v2: chay OWLv2 tren DUNG 8000
anh cung mau (stride giong build_dense_objects_baseline_frcnn.py --n-samples 8000, khong random)
- KHONG phai full corpus, chi de doi chieu chat luong/toc do 2 model truoc khi quyet dinh chay
full bang model nao (nguoi dung yeu cau 2026-08-14).

Output RIENG: index/dense/objects_owlv2_samesample.parquet
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal
import pandas as pd

from config import DENSE_DIR, DENSE_META_PATH, INDEX_DIR

LABEL_VI_PATH = INDEX_DIR / "label_vi.json"
OUT_PATH = DENSE_DIR / "objects_owlv2_samesample.parquet"
DONE_PATH = DENSE_DIR / "objects_owlv2_samesample_done.jsonl"

MIN_SCORE = 0.15
MAX_IMAGES_PER_CALL = 8
CHECKPOINT_EVERY_CHUNKS = 20
N_READ_WORKERS = 32
N_SAMPLES = 8000


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)


def _load_done_keys() -> set[tuple[str, int]]:
    if not DONE_PATH.exists():
        return set()
    keys = set()
    with open(DONE_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            keys.add((r["video_id"], r["frame_idx"]))
    return keys


def run() -> None:
    with open(LABEL_VI_PATH, encoding="utf-8") as f:
        labels = sorted(json.load(f).keys())
    print(f"Dung {len(labels)} nhan closed-set (OpenImages, tu label_vi.json)", file=sys.stderr)

    meta = pd.read_parquet(DENSE_META_PATH)
    total = len(meta)
    stride = max(1, total // N_SAMPLES)
    sampled = meta.iloc[::stride].head(N_SAMPLES)  # CUNG cach lay mau voi baseline FRCNN
    recs = sampled[["video_id", "frame_idx", "path"]].to_dict("records")
    print(f"Mau deu (stride={stride}) {len(recs)}/{total} anh - CUNG mau voi baseline FRCNN",
          file=sys.stderr)

    done_keys = _load_done_keys()
    if done_keys:
        before = len(recs)
        recs = [r for r in recs if (r["video_id"], r["frame_idx"]) not in done_keys]
        print(f"Resume: da xong {len(done_keys)} anh tu lan chay truoc, "
              f"con lai {len(recs)}/{before} can xu ly", file=sys.stderr)
    if not recs:
        print("Khong con gi de lam.", file=sys.stderr)
        return

    Detector = modal.Cls.from_name("aic2026-owlv2", "OWLv2Detector")
    detector = Detector()

    all_rows: list[dict] = (
        pd.read_parquet(OUT_PATH).to_dict("records") if OUT_PATH.exists() else []
    )

    n_errors = 0
    done_f = open(DONE_PATH, "a", encoding="utf-8")
    read_pool = ThreadPoolExecutor(max_workers=N_READ_WORKERS)

    batches = [recs[i:i + MAX_IMAGES_PER_CALL] for i in range(0, len(recs), MAX_IMAGES_PER_CALL)]

    def _read_batch(batch_recs: list[dict]) -> list[bytes]:
        return [Path(r["path"]).read_bytes() for r in batch_recs]

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
        map_results = detector.detect_batch.map(
            _batches_bytes_gen(), [labels] * len(batches),
            order_outputs=True, return_exceptions=True, wrap_returned_exceptions=False,
        )
        for ci, (results, batch) in enumerate(zip(map_results, batches), 1):
            if isinstance(results, Exception):
                n_errors += len(batch)
                print(f"  [LOI] batch {ci}/{len(batches)}: {type(results).__name__} {str(results)[:120]}",
                      file=sys.stderr)
                continue

            for rec, dets in zip(batch, results):
                for d in dets:
                    if d["score"] < MIN_SCORE:
                        continue
                    all_rows.append({
                        "video_id": rec["video_id"],
                        "frame_idx": rec["frame_idx"],
                        "label": d["label"],
                        "score": d["score"],
                        "ymin": d["ymin"], "xmin": d["xmin"],
                        "ymax": d["ymax"], "xmax": d["xmax"],
                        "source": "owlv2",
                    })
                done_f.write(json.dumps({"video_id": rec["video_id"], "frame_idx": rec["frame_idx"]},
                                         ensure_ascii=False) + "\n")
            done_f.flush()
            os.fsync(done_f.fileno())

            if ci % CHECKPOINT_EVERY_CHUNKS == 0 or ci == len(batches):
                _atomic_write_parquet(pd.DataFrame(all_rows), OUT_PATH)
                done_count = min(ci * MAX_IMAGES_PER_CALL, len(recs))
                print(f"[{done_count}/{len(recs)}] checkpoint {len(all_rows)} detection, "
                      f"{n_errors} loi -> {OUT_PATH}", file=sys.stderr)
    finally:
        done_f.close()
        read_pool.shutdown(wait=False)

    _atomic_write_parquet(pd.DataFrame(all_rows), OUT_PATH)
    print(f"\nDa luu {len(all_rows)} detection, {n_errors} loi -> {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    run()
