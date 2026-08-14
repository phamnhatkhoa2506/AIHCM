"""Bước 2: build inverted index Objects (dạng long: 1 dòng/detection) sau khi lọc theo
ngưỡng confidence đã chốt từ build_object_stats.py (THRESHOLD=0.3, xem log kết quả:
trung bình 3.94 nhãn khác nhau/frame, 514/600 nhãn còn xuất hiện — điểm cân bằng, không
phải đoán). Dùng ThreadPoolExecutor vì bottleneck là I/O đọc 177k file nhỏ, không phải CPU.
"""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from config import INDEX_DIR, INDEX_META_PATH, OBJECTS_DIR, OBJECTS_INDEX_PATH

THRESHOLD = 0.3
N_WORKERS = 32


def _read_one(video_id: str, local_idx: int) -> list[dict] | None:
    fp = OBJECTS_DIR / video_id / f"{local_idx + 1:03d}.json"
    if not fp.exists():
        return None
    with open(fp, encoding="utf-8") as f:
        d = json.load(f)

    rows = []
    for score, label, box in zip(d["detection_scores"], d["detection_class_entities"], d["detection_boxes"]):
        s = float(score)
        if s < THRESHOLD:
            continue
        ymin, xmin, ymax, xmax = (float(v) for v in box)
        rows.append(
            {
                "video_id": video_id,
                "local_idx": local_idx,
                "label": label,
                "score": s,
                "ymin": ymin,
                "xmin": xmin,
                "ymax": ymax,
                "xmax": xmax,
            }
        )
    return rows


def main() -> None:
    meta = pd.read_parquet(INDEX_META_PATH, columns=["video_id", "local_idx"])
    tasks = list(zip(meta["video_id"].tolist(), meta["local_idx"].tolist()))
    n_frames = len(tasks)

    all_rows: list[dict] = []
    missing = 0

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_read_one, vid, li): (vid, li) for vid, li in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res is None:
                missing += 1
                continue
            all_rows.extend(res)

            if i % 20000 == 0 or i == n_frames:
                print(f"[{i}/{n_frames}]", file=sys.stderr)

    df = pd.DataFrame(all_rows)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OBJECTS_INDEX_PATH, index=False)

    print(f"Xong: {len(df)} detection (nguong>={THRESHOLD}), {df['label'].nunique()} nhan, "
          f"{n_frames - missing} frame co detection, {missing} frame thieu file")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
