"""Pilot do toc do GPU thuc te cua Grounding DINO (A10G, da deploy) cho 2 che do:
  - OPEN: prompt ngan (cum VLM de xuat that, lay tu vocab_discovery_flagged_frames.jsonl)
  - CLOSED: prompt day du 514 nhan OpenImages (nhu detector goc, nhung qua Grounding DINO)
Do giay/anh (batch=8, giong MAX_IMAGES_PER_CALL dang dung trong run_grounding_dino.py), roi
suy ra uoc tinh cho toan bo pham vi tuong ung (17,871 frame cho OPEN, 177,321 cho CLOSED).
KHONG dung so doan - chi bao cao so do that.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))

import io
import json
import time

import modal
import pandas as pd
from PIL import Image

from config import INDEX_DIR, INDEX_META_PATH
from keyframe_images import read_keyframe_bytes

BATCH_SIZE = 8
N_BATCHES_PER_MODE = 5  # 5 batch x 8 anh = 40 anh/che do, du de co so trung binh on dinh

with open(INDEX_DIR / "label_vi.json", encoding="utf-8") as f:
    _LABELS = sorted(json.load(f).keys())

# PHAT HIEN THAT (2026-08-06): Grounding DINO text encoder gioi han CUNG 256 token - prompt
# ca 514 nhan tokenize ra 1342 token -> RuntimeError ngay lap tuc. Phai chia nho thanh nhieu
# prompt <256 token, goi NHIEU LUOT/anh de phu het 514 nhan. 60 nhan/chunk la uoc luong an
# toan (trung binh ~2 token/nhan + dau cham -> ~120-150 token/chunk, con du bien).
LABELS_PER_CHUNK = 60
CLOSED_PROMPT_CHUNKS = [
    ". ".join(lb.lower() for lb in _LABELS[i:i + LABELS_PER_CHUNK]) + "."
    for i in range(0, len(_LABELS), LABELS_PER_CHUNK)
]

FLAGGED_PATH = INDEX_DIR / "vocab_discovery_flagged_frames.jsonl"


def _sample_open_batches():
    """Lay N_BATCHES_PER_MODE*BATCH_SIZE frame that tu flagged_frames.jsonl, kem dung prompt
    (cum phrase that cua tung frame, GOM theo cung prompt giong y het pipeline that)."""
    from collections import defaultdict
    groups = defaultdict(list)
    with open(FLAGGED_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            prompt = ". ".join(sorted({p.strip().lower() for p in r["phrases"]})) + "."
            groups[prompt].append(r)
            # du 1 nhom >= BATCH_SIZE la co the dung roi, khong can quet het file
            if len(groups[prompt]) >= N_BATCHES_PER_MODE * BATCH_SIZE:
                break

    batches = []
    for prompt, items in groups.items():
        for i in range(0, len(items), BATCH_SIZE):
            chunk = items[i:i + BATCH_SIZE]
            if len(chunk) == BATCH_SIZE:
                batches.append((prompt, chunk))
            if len(batches) >= N_BATCHES_PER_MODE:
                break
        if len(batches) >= N_BATCHES_PER_MODE:
            break
    return batches


def _sample_closed_batches():
    """Do dung 1 batch anh CO DINH, lap qua vai chunk prompt khac nhau (dai dien cho viec
    can bao nhieu luot/anh de phu het 514 nhan) - khong can moi chunk 1 bo anh rieng."""
    meta = pd.read_parquet(INDEX_META_PATH)
    sample = meta.sample(n=BATCH_SIZE, random_state=123)
    recs = [{"video_id": r.video_id, "local_idx": int(r.local_idx)} for r in sample.itertuples(index=False)]
    n_test_chunks = min(N_BATCHES_PER_MODE, len(CLOSED_PROMPT_CHUNKS))
    return [(CLOSED_PROMPT_CHUNKS[i], recs) for i in range(n_test_chunks)]


def _time_batches(name, batches, detector):
    print(f"\n=== {name} ===")
    times = []
    for i, (prompt, recs) in enumerate(batches, 1):
        images_bytes = [read_keyframe_bytes(r["video_id"], r["local_idx"]) for r in recs]
        t0 = time.perf_counter()
        results = detector.detect_batch.remote(images_bytes, prompt, 0.25, 0.20)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"  batch {i}: {len(recs)} anh, {elapsed:.2f}s ({elapsed/len(recs):.3f}s/anh), "
              f"prompt_len={len(prompt)} ky tu, {sum(len(d) for d in results)} detection")
    avg_per_img = sum(times) / (len(times) * BATCH_SIZE)
    print(f"  -> trung binh: {avg_per_img:.3f}s/anh (tren {len(times)} batch, {len(times)*BATCH_SIZE} anh)")
    return avg_per_img


def main():
    Detector = modal.Cls.from_name("aic2026-grounding-dino", "Detector")
    detector = Detector()

    open_batches = _sample_open_batches()
    closed_batches = _sample_closed_batches()

    open_rate = _time_batches("OPEN (prompt ngan, cum VLM that)", open_batches, detector)
    closed_rate = _time_batches("CLOSED (prompt day du 514 nhan)", closed_batches, detector)

    N_OPEN_TOTAL = 17871
    N_CLOSED_TOTAL = 177321
    N_CHUNKS = len(CLOSED_PROMPT_CHUNKS)
    closed_rate_full = closed_rate * N_CHUNKS  # can N_CHUNKS luot/anh de phu het 514 nhan

    print("\n" + "=" * 60)
    print(f"CLOSED can {N_CHUNKS} chunk/anh ({LABELS_PER_CHUNK} nhan/chunk) de phu het 514 nhan")
    print(f"  -> rate CLOSED thuc (da nhan {N_CHUNKS} luot): {closed_rate_full:.3f}s/anh")
    print("\nUOC TINH TONG THOI GIAN (tuan tu, CHUA tinh song song nhieu container):")
    print(f"  OPEN   ({N_OPEN_TOTAL} frame): {open_rate * N_OPEN_TOTAL / 60:.1f} phut")
    print(f"  CLOSED ({N_CLOSED_TOTAL} frame): {closed_rate_full * N_CLOSED_TOTAL / 60:.1f} phut "
          f"({closed_rate_full * N_CLOSED_TOTAL / 3600:.1f} gio)")
    print("  (chia cho so container song song de ra thoi gian thuc te)")


if __name__ == "__main__":
    if _sys.stdout.encoding and _sys.stdout.encoding.lower() != "utf-8":
        try:
            _sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
