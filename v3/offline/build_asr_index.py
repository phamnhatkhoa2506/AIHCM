"""ASR — transcript tiếng nói (bản tin/phóng viên/MC), dùng PhoWhisper qua Modal
(modal_infra/asr_app.py). Audio trích LOCAL bằng ffmpeg (share/video_audio.py) trước khi gửi
Modal — không gửi nguyên video (nặng hơn audio rất nhiều).

Khác OCR: PhoWhisper trả timestamp theo GIÂY (không phải toạ độ khung hình) — phải map sang
local_idx qua meta.parquet["pts_time"] (tìm frame gần nhất với mốc giây đó) mới ra được cùng
schema (video_id, local_idx_start, local_idx_end, ...) để dùng chung cơ chế query với OCR.

Chạy: python offline/build_asr_index.py                # full 873 video
      python offline/build_asr_index.py --limit 5       # test nhanh vai video dau
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

import modal
import numpy as np
import pandas as pd

from atomic_io import atomic_write_parquet

from config import INDEX_DIR, INDEX_META_PATH
from tiers.tier1_filter import _strip_accents
from video_audio import extract_audio_bytes

OUT_PATH = INDEX_DIR / "asr_text.parquet"  # schema GIONG ocr_text.parquet (source="asr",
# ymin/xmin/ymax/xmax = NaN vi khong co vi tri khong gian) - de sau nay gop chung neu can
# (xem resources.py, chua gop tu dong, doc rieng qua resources.get().asr_index).
DONE_PATH = INDEX_DIR / "asr_done.jsonl"  # resume - video_id DA transcribe xong

N_EXTRACT_WORKERS = 8  # trich audio (ffmpeg) la CPU-bound nang hon doc keyframe - it luong hon
BATCH_SIZE = 16  # ~2x MAX_CONTAINERS (8, xem asr_app.py) - moi batch chia deu qua .map()


def _load_done_videos() -> set[str]:
    if not DONE_PATH.exists():
        return set()
    with open(DONE_PATH, encoding="utf-8") as f:
        return {json.loads(line)["video_id"] for line in f}


def _seconds_to_local_idx_range(video_id: str, start_s: float, end_s: float, meta: pd.DataFrame) -> tuple[int, int]:
    """Tim local_idx gan nhat voi moc giay start/end trong DUNG video (tra ve range bao trum,
    dam bao local_idx_start<=local_idx_end ke ca khi start==end hoac video qua ngan)."""
    g = meta[meta["video_id"] == video_id]
    if g.empty:
        return 0, 0
    pts = g["pts_time"].to_numpy()
    idxs = g["local_idx"].to_numpy()
    i_start = int(idxs[np.argmin(np.abs(pts - start_s))])
    i_end = int(idxs[np.argmin(np.abs(pts - end_s))])
    return min(i_start, i_end), max(i_start, i_end)


def run(limit: int | None = None) -> None:
    meta = pd.read_parquet(INDEX_META_PATH)
    video_ids = sorted(meta["video_id"].unique())
    if limit:
        video_ids = video_ids[:limit]
    print(f"Tong {len(video_ids)} video can ASR", file=sys.stderr)

    done = _load_done_videos()
    if done:
        before = len(video_ids)
        video_ids = [v for v in video_ids if v not in done]
        print(f"Resume: da xong {len(done)} video tu lan chay truoc, "
              f"con lai {len(video_ids)}/{before} can xu ly", file=sys.stderr)
    if not video_ids:
        print("Khong con gi de lam.", file=sys.stderr)
        return

    Transcriber = modal.Cls.from_name("aic2026-asr", "Transcriber")
    transcriber = Transcriber()

    all_rows: list[dict] = pd.read_parquet(OUT_PATH).to_dict("records") if OUT_PATH.exists() else []
    done_f = open(DONE_PATH, "a", encoding="utf-8")
    n_errors = 0
    batches = [video_ids[i: i + BATCH_SIZE] for i in range(0, len(video_ids), BATCH_SIZE)]
    extract_pool = ThreadPoolExecutor(max_workers=N_EXTRACT_WORKERS)
    try:
        # BUG THAT (2026-08-11, nguoi dung hoi "sao chi co 1 container chay"): ban truoc goi
        # transcriber.transcribe.remote() TUAN TU tung video 1 trong vong lap - Modal KHONG CO
        # LY DO gi mo them container vi chi co DUNG 1 request dang bay tai 1 thoi diem (giong
        # het bug da gap va sua o run_grounding_dino.py truoc day). SUA LAN 1: chia BATCH_SIZE
        # video 1 lan, trich audio CA BATCH truoc roi moi goi .map().
        #
        # SUA LAN 2 (2026-08-11, nguoi dung hoi tiep - trich audio TUAN TU truoc moi batch van
        # de 8 container GPU ngoi KHONG lam gi (van bi tinh phi "warm") trong luc cho PHA 1 CPU
        # xong): PREFETCH - submit trich audio cho batch KE TIEP ngay SAU KHI da co du lieu
        # batch HIEN TAI (khong doi ket qua), de no chay NEN song song voi luc Modal dang xu ly
        # batch hien tai tren GPU. Trich 16 video qua 8 luong CPU luon nhanh hon nhieu so voi
        # Modal xu ly 16 video tren GPU (~176-250s), nen den luc can batch ke tiep gan nhu chac
        # chan da xong san - gan nhu xoa het khoang trong GPU idle giua 2 batch.
        extract_futures: dict[int, list] = {}
        if batches:
            extract_futures[0] = [extract_pool.submit(extract_audio_bytes, v) for v in batches[0]]

        for bi, batch_vids in enumerate(batches):
            audio_results = [f.result() for f in extract_futures.pop(bi)]
            if bi + 1 < len(batches):
                extract_futures[bi + 1] = [
                    extract_pool.submit(extract_audio_bytes, v) for v in batches[bi + 1]
                ]

            try:
                map_results = list(transcriber.transcribe.map(
                    audio_results, order_outputs=True, return_exceptions=True,
                    wrap_returned_exceptions=False,
                ))
            except Exception as e:
                n_errors += len(batch_vids)
                print(f"  [LOI CA BATCH] {type(e).__name__} {str(e)[:200]}", file=sys.stderr)
                continue

            for vid, chunks in zip(batch_vids, map_results):
                if isinstance(chunks, Exception):
                    n_errors += 1
                    print(f"  [LOI] {vid}: {type(chunks).__name__} {str(chunks)[:120]}", file=sys.stderr)
                    continue
                for c in chunks:
                    li_start, li_end = _seconds_to_local_idx_range(vid, c["start"], c["end"], meta)
                    text_norm = _strip_accents(c["text"])
                    all_rows.append({
                        "video_id": vid,
                        "local_idx_start": li_start,
                        "local_idx_end": li_end,
                        "text_raw": c["text"],
                        "text_norm": text_norm,
                        "ymin": np.nan, "xmin": np.nan, "ymax": np.nan, "xmax": np.nan,
                        "score": np.nan,  # pipeline ASR khong tra confidence tin cay per-chunk
                        "source": "asr",
                    })
                done_f.write(json.dumps({"video_id": vid}, ensure_ascii=False) + "\n")
            done_f.flush()
            os.fsync(done_f.fileno())

            atomic_write_parquet(pd.DataFrame(all_rows), OUT_PATH, index=False)
            done_count = sum(len(batches[j]) for j in range(bi + 1))
            print(f"[{done_count}/{len(video_ids)}] checkpoint {len(all_rows)} doan transcript, "
                  f"{n_errors} loi -> {OUT_PATH}", file=sys.stderr)
    finally:
        done_f.close()
        extract_pool.shutdown(wait=False)

    atomic_write_parquet(pd.DataFrame(all_rows), OUT_PATH, index=False)
    print(f"\nDa luu {len(all_rows)} doan transcript, {n_errors} loi -> {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="chi ASR N video dau (test nhanh)")
    args = parser.parse_args()
    run(limit=args.limit)
