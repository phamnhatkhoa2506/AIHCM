"""Closed-set object detection cho bo keyframe DENSE (369.589 anh) bang OWLv2, dung DUNG 514
nhan OpenImages da co san anh xa tieng Viet (index/label_vi.json) - xem modal_infra/owlv2_app.py
va hoi thoai 2026-08-14 (ly do chon OWLv2 thay Grounding DINO/Faster-RCNN).

2026-08-14 (thay doi kien truc lon): da chan doan nghen co chai la LOCAL CPU (doc/giai nen anh
+ gui bytes qua mang cho Modal, khong phai Modal GPU container - da kiem chung bang cach giam
container 10->5 KHONG anh huong toc do, va tang worker doc file 8->32 CUNG khong nhanh hon).
Giai phap: anh da duoc upload san len Modal Volume "aic2026-dense-keyframes" (xem
offline/upload_dense_to_volume.py + share/dense_volume_map.py) - driver nay CHI gui DUONG DAN
TUONG DOI (string, vai chuc byte/anh) thay vi doc+gui bytes anh (~70KB/anh) - loai bo hoan toan
tai CPU/mang local, container Modal tu doc thang tu Volume.

Schema OUT_PATH (index/dense/objects_index.parquet): video_id, frame_idx, label, score,
ymin/xmin/ymax/xmax (normalize [0,1]), source="owlv2" - CUNG QUY UOC voi objects_index.parquet
sparse (khong co local_idx, chi khac frame_idx thay local_idx).

Chay: python offline/build_dense_objects_index.py
      python offline/build_dense_objects_index.py --limit 500   # test nhanh
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/

import argparse
import json
import os
import sys
from pathlib import Path

import modal
import pandas as pd

from config import DENSE_DIR, DENSE_META_PATH, INDEX_DIR
from dense_volume_map import to_volume_rel_path

LABEL_VI_PATH = INDEX_DIR / "label_vi.json"
OUT_PATH = DENSE_DIR / "objects_index.parquet"
DONE_PATH = DENSE_DIR / "objects_dense_done.jsonl"


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Ghi qua file .tmp roi os.replace() (atomic tren cung ổ đĩa) - tranh hong file khi
    tien trinh bi kill DUNG LUC dang ghi (2026-08-14: objects_index.parquet bi hong "Parquet
    magic bytes not found" sau 1 lan bi kill giua chung to_parquet() ghi thang vao OUT_PATH,
    mat toan bo ~122k anh da xu ly du DONE_PATH van ghi la da xong)."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)


MIN_SCORE = 0.15  # nguong OWLv2 (khac PaddleOCR/closed-set BTC - model+kien truc khac nen
# nguong khac, da thu nghiem 0.15 cho ket qua hop ly o anh test that, xem hoi thoai).
# 2026-08-14 (lan 2, sau khi chuyen sang doc tu Modal Volume): do lai batch size vi ban chat
# chi phi khac han (khong con gui bytes anh, chi gui path string - rat re) - batch=16 (176.2
# anh/phut/container) < 32 (174.3) < 128 (199.2) < 64 (244.7, TOT NHAT). Doi tu 16 (ket luan cu
# khi con gui bytes, khong con dung nua) len 64.
MAX_IMAGES_PER_CALL = 64
CHECKPOINT_EVERY_CHUNKS = 3  # 3*64=192 anh/checkpoint, gan tuong duong tan suat cu (~160)


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
    # 2026-08-14: KHONG con doc/gui labels tu day nua - container Modal tu doc label_vi.json
    # (nhung san trong image) + tokenize 1 LAN duy nhat trong load() (xem owlv2_app.py). Truoc
    # do gui [labels]*so_batch qua .map() khien Owlv2Processor tokenize lai 514 nhan MOI BATCH
    # (dung do: 3.3-3.5s/64-anh, LON HON CA forward GPU 1.44s) - day la nguyen nhan chinh khien
    # toc do khong tang du da sua nghen local CPU (Volume) va tang batch size.
    print("Nhan closed-set: container tu doc/tokenize 1 lan (xem owlv2_app.py::load())", file=sys.stderr)

    meta = pd.read_parquet(DENSE_META_PATH)
    if limit:
        meta = meta.head(limit)
    recs = meta[["video_id", "frame_idx", "path"]].to_dict("records")
    print(f"Tong {len(recs)} anh dense can detect object", file=sys.stderr)

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

    batches = [recs[i:i + MAX_IMAGES_PER_CALL] for i in range(0, len(recs), MAX_IMAGES_PER_CALL)]

    def _rel_paths_gen():
        # KHONG con doc bytes/dung ThreadPoolExecutor - chi tinh duong dan tuong doi (string,
        # rat re) - Modal tu doc anh thang tu Volume da upload san (xem docstring dau file).
        for batch in batches:
            yield [to_volume_rel_path(r["path"]) for r in batch]

    try:
        # BUG DA GAP NHIEU LAN (OCR/ASR/embeddings) - .remote() tuan tu khong cho Modal ly do
        # de mo them container. PHAI .map() de autoscale that qua nhieu container song song.
        map_results = detector.detect_batch_from_volume.map(
            _rel_paths_gen(),
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

    _atomic_write_parquet(pd.DataFrame(all_rows), OUT_PATH)
    print(f"\nDa luu {len(all_rows)} detection, {n_errors} loi -> {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="chi detect N anh dau (test nhanh)")
    args = parser.parse_args()
    run(limit=args.limit)
