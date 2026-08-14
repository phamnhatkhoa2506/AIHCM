"""OCR — chữ trên màn hình (lower-third bản tin, watermark kênh, biển hiệu/màn hình trong
khung hình). Dùng PaddleOCR qua Modal (modal_infra/ocr_app.py), TOẠ ĐỘ TRỰC TIẾP (không qua
object detection — xem docstring ocr_app.py).

2 giai đoạn TÁCH RIÊNG (giống vocab_discovery.py run_discovery/aggregate_vocab):
  GIAI ĐOẠN 1 (run_ocr):    gọi Modal, lưu MỌI dòng chữ thô/frame -> ocr_raw_detections.parquet.
  GIAI ĐOẠN 2 (cluster_runs): gộp các frame liên tiếp CÙNG video + CÙNG chữ (đã chuẩn hoá)
                              thành 1 "run" (chuỗi đứng yên) -> ocr_text.parquet (schema dùng
                              cho query, xem share/tiers/tier1_filter.py::by_text).
Tách riêng để tune RUN_GAP_THRESHOLD/MIN_SCORE sau này KHÔNG cần gọi lại Modal (tốn tiền).

Sample thưa (không chạy toàn bộ 177,321 frame) — dùng LẠI đúng chiến lược của
vocab_discovery._sampled_frames() (SAMPLE_SKIP_INTRO=5, SAMPLE_STRIDE=15): lower-third bản tin
đứng yên hàng chục giây, không cần OCR mọi frame, gom cụm ở giai đoạn 2 sẽ bù lại các frame bị
bỏ qua giữa 2 mốc sample.

Chạy: python offline/build_ocr_index.py             # full (theo sample thua)
      python offline/build_ocr_index.py --limit 200 # test nhanh vai video dau
      python offline/build_ocr_index.py --cluster-only  # chi gom cum lai tu raw da co (khong goi Modal)
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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import modal
import pandas as pd

from atomic_io import atomic_write_parquet
from PIL import Image

from config import INDEX_DIR
from keyframe_images import read_keyframe_bytes
from tiers.tier1_filter import _strip_accents
from vocab_discovery import _sampled_frames

RAW_PATH = INDEX_DIR / "ocr_raw_detections.parquet"  # 1 dong/dong chu/frame, CHUA gom cum
DONE_PATH = INDEX_DIR / "ocr_done.jsonl"  # resume - (video_id,local_idx) DA goi OCR (ke ca 0
# dong chu - phai track rieng, khong dua vao co dong trong RAW_PATH hay khong)
OUT_PATH = INDEX_DIR / "ocr_text.parquet"  # KET QUA CUOI, dung cho by_text() - schema:
# video_id, local_idx_start, local_idx_end, text_raw, text_norm, ymin,xmin,ymax,xmax, score, source

MIN_SCORE = 0.5  # loc bo nhan dien do tin cay thap (chu mo/nhieu/bi che) - PaddleOCR tra score
# theo tung dong, khong phai nguong tuy y - de o day de tune rieng ma khong can goi lai Modal.

MAX_IMAGES_PER_CALL = 16  # OCR nhe hon Grounding DINO (khong batch GPU that, xem ocr_app.py)
# nen chon cao hon MAX_IMAGES_PER_CALL=8 cua run_grounding_dino.py - van du an toan tranh 1
# lan goi qua lau (timeout) hay 1 loi lam mat qua nhieu progress.
CHECKPOINT_EVERY_CHUNKS = 20

RUN_GAP_THRESHOLD = 20  # giong het run_grounding_dino.py - 2 frame sample lien tiep (stride=15)
# van coi la "cung 1 lan chu dung yen" neu khoang cach local_idx <= nguong nay.


def _load_done_keys() -> set[tuple[str, int]]:
    if not DONE_PATH.exists():
        return set()
    keys = set()
    with open(DONE_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            keys.add((r["video_id"], r["local_idx"]))
    return keys


N_READ_WORKERS = 32


def run_ocr(limit: int | None = None) -> None:
    """GIAI DOAN 1 - goi Modal, checkpoint MOI dong chu tho vao RAW_PATH (chua gom cum)."""
    frames = _sampled_frames()
    if limit:
        frames = frames.head(limit)
    recs = frames[["video_id", "local_idx"]].to_dict("records")
    print(f"Tong {len(recs)} frame (sample thua) can OCR", file=sys.stderr)

    done_keys = _load_done_keys()
    if done_keys:
        before = len(recs)
        recs = [r for r in recs if (r["video_id"], r["local_idx"]) not in done_keys]
        print(f"Resume: da xong {len(done_keys)} frame tu lan chay truoc, "
              f"con lai {len(recs)}/{before} can xu ly", file=sys.stderr)
    if not recs:
        print("Khong con gi de lam.", file=sys.stderr)
        return

    # PHA 1 (CPU, I/O-bound) - doc HET anh truoc, tach han khoi PHA 2 (GPU) - giong pattern
    # run_grounding_dino.py._prepare_chunks() (tranh round-trip lap lai CPU<->GPU moi vong lap).
    def _read_one(rec: dict) -> bytes:
        return read_keyframe_bytes(rec["video_id"], rec["local_idx"])

    print(f"Dang doc {len(recs)} anh (song song {N_READ_WORKERS} luong)...", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=N_READ_WORKERS) as pool:
        images = list(pool.map(_read_one, recs))
    print("Doc xong, bat dau PHA 2 (GPU, goi Modal.map() song song nhieu container)...", file=sys.stderr)

    chunks_recs = [recs[i:i + MAX_IMAGES_PER_CALL] for i in range(0, len(recs), MAX_IMAGES_PER_CALL)]
    chunks_imgs = [images[i:i + MAX_IMAGES_PER_CALL] for i in range(0, len(images), MAX_IMAGES_PER_CALL)]

    OCRDetector = modal.Cls.from_name("aic2026-ocr", "OCRDetector")
    detector = OCRDetector()

    all_rows: list[dict] = (
        pd.read_parquet(RAW_PATH).to_dict("records") if RAW_PATH.exists() else []
    )

    n_errors = 0
    done_f = open(DONE_PATH, "a", encoding="utf-8")
    try:
        map_results = detector.ocr_batch.map(chunks_imgs, order_outputs=True, return_exceptions=True)
        for ci, (results, chunk) in enumerate(zip(map_results, chunks_recs), 1):
            if isinstance(results, Exception):
                n_errors += 1
                print(f"  [LOI] chunk {ci}/{len(chunks_recs)}: {type(results).__name__} {str(results)[:120]}",
                      file=sys.stderr)
                continue

            for rec, lines in zip(chunk, results):
                for ln in lines:
                    if ln["score"] < MIN_SCORE:
                        continue
                    all_rows.append({
                        "video_id": rec["video_id"],
                        "local_idx": rec["local_idx"],
                        "text_raw": ln["text"],
                        "score": ln["score"],
                        "ymin": ln["ymin"], "xmin": ln["xmin"],
                        "ymax": ln["ymax"], "xmax": ln["xmax"],
                    })
                done_f.write(json.dumps({"video_id": rec["video_id"], "local_idx": rec["local_idx"]},
                                         ensure_ascii=False) + "\n")
            done_f.flush()
            os.fsync(done_f.fileno())

            if ci % CHECKPOINT_EVERY_CHUNKS == 0 or ci == len(chunks_recs):
                atomic_write_parquet(pd.DataFrame(all_rows), RAW_PATH, index=False)
                print(f"[chunk {ci}/{len(chunks_recs)}] checkpoint {len(all_rows)} dong chu, "
                      f"{n_errors} chunk loi -> {RAW_PATH}", file=sys.stderr)
    finally:
        done_f.close()

    atomic_write_parquet(pd.DataFrame(all_rows), RAW_PATH, index=False)
    print(f"\nDa luu {len(all_rows)} dong chu tho, {n_errors} chunk loi -> {RAW_PATH}", file=sys.stderr)


def cluster_runs() -> None:
    """GIAI DOAN 2 - gom RAW_PATH thanh cac "run" (chuoi chu dung yen lien tuc) -> OUT_PATH.
    Khong goi Modal - chi xu ly du lieu da co, chay lai duoc bao nhieu lan tuy y de tune
    RUN_GAP_THRESHOLD/MIN_SCORE ma khong ton tien."""
    if not RAW_PATH.exists():
        print(f"Chua co {RAW_PATH} - chay run_ocr() truoc.", file=sys.stderr)
        return

    raw = pd.read_parquet(RAW_PATH)
    raw = raw[raw["score"] >= MIN_SCORE].copy()
    raw["text_norm"] = raw["text_raw"].apply(_strip_accents)
    print(f"Dang gom cum {len(raw)} dong chu (sau loc score>={MIN_SCORE})...", file=sys.stderr)

    # gom theo (video_id, text_norm) - CHI coi la "cung 1 lan chu xuat hien" neu ca video LAN
    # noi dung chu (da chuan hoa) deu trung - giong het _split_and_thin() cua
    # run_grounding_dino.py (nhom theo video+prompt truoc khi tach doan lien tuc theo thoi gian).
    # DON GIAN HOA co y thuc (chua kiem tra chong lan vi tri khong gian giua cac run cung text) -
    # neu sau nay phat hien case that 2 vung chu KHAC NHAU tren cung frame co CUNG noi dung
    # (hiem, vd watermark lap lai 2 goc) thi them dieu kien IoU luc do, khong lam truoc khi co
    # bang chung that.
    by_video_text: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in raw.itertuples(index=False):
        by_video_text[(row.video_id, row.text_norm)].append(row._asdict())

    out_rows = []
    for (vid, text_norm), recs in by_video_text.items():
        recs.sort(key=lambda r: r["local_idx"])
        run: list[dict] = [recs[0]]
        for r in recs[1:]:
            if r["local_idx"] - run[-1]["local_idx"] <= RUN_GAP_THRESHOLD:
                run.append(r)
            else:
                out_rows.append(_run_to_row(vid, text_norm, run))
                run = [r]
        out_rows.append(_run_to_row(vid, text_norm, run))

    df = pd.DataFrame(out_rows)
    atomic_write_parquet(df, OUT_PATH, index=False)
    print(f"Da gom {len(raw)} dong chu tho thanh {len(df)} run -> {OUT_PATH}", file=sys.stderr)


def _run_to_row(video_id: str, text_norm: str, run: list[dict]) -> dict:
    return {
        "video_id": video_id,
        "local_idx_start": min(r["local_idx"] for r in run),
        "local_idx_end": max(r["local_idx"] for r in run),
        "text_raw": max(run, key=lambda r: r["score"])["text_raw"],  # ban co score cao nhat
        "text_norm": text_norm,
        "ymin": sum(r["ymin"] for r in run) / len(run),
        "xmin": sum(r["xmin"] for r in run) / len(run),
        "ymax": sum(r["ymax"] for r in run) / len(run),
        "xmax": sum(r["xmax"] for r in run) / len(run),
        "score": sum(r["score"] for r in run) / len(run),
        "source": "ocr_paddle",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="chi OCR N frame dau (test nhanh)")
    parser.add_argument("--cluster-only", action="store_true", help="chi gom cum lai tu raw da co, khong goi Modal")
    args = parser.parse_args()

    if args.cluster_only:
        cluster_runs()
    else:
        run_ocr(limit=args.limit)
        cluster_runs()
