"""Bước 1: quét TOÀN corpus Objects (177k file JSON) để có căn cứ chọn ngưỡng confidence
và tần suất theo nhãn — thay vì đoán ngưỡng. Chạy 1 lần, kết quả dùng cho build_objects_index.py.
"""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from config import INDEX_DIR, INDEX_META_PATH, OBJECTS_DIR

STATS_PATH = INDEX_DIR / "object_score_stats.json"
LABEL_FREQ_PATH = INDEX_DIR / "label_freq_by_threshold.parquet"

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
N_WORKERS = 32


def _read_one(video_id: str, local_idx: int) -> tuple[list[float], list[str]] | None:
    fp = OBJECTS_DIR / video_id / f"{local_idx + 1:03d}.json"
    if not fp.exists():
        return None
    with open(fp, encoding="utf-8") as f:
        d = json.load(f)
    return [float(s) for s in d["detection_scores"]], d["detection_class_entities"]


def main() -> None:
    meta = pd.read_parquet(INDEX_META_PATH, columns=["video_id", "local_idx"])
    tasks = list(zip(meta["video_id"].tolist(), meta["local_idx"].tolist()))
    n_frames = len(tasks)

    all_scores: list[float] = []
    # đếm số FRAME (không phải số box) có best-score theo nhãn >= ngưỡng
    label_frame_count = {t: {} for t in THRESHOLDS}
    missing = 0

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_read_one, vid, li): (vid, li) for vid, li in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res is None:
                missing += 1
                continue
            scores, entities = res
            all_scores.extend(scores)

            # best score mỗi nhãn trong frame (1 label có thể có nhiều box)
            best_by_label: dict[str, float] = {}
            for s, e in zip(scores, entities):
                if s > best_by_label.get(e, -1.0):
                    best_by_label[e] = s

            for t in THRESHOLDS:
                counts = label_frame_count[t]
                for lb, best in best_by_label.items():
                    if best >= t:
                        counts[lb] = counts.get(lb, 0) + 1

            if i % 20000 == 0 or i == n_frames:
                print(f"[{i}/{n_frames}]", file=sys.stderr)

    scores_arr = np.array(all_scores)
    pct = {p: float(np.percentile(scores_arr, p)) for p in [50, 75, 90, 95, 99]}
    stats = {
        "n_frames_scanned": n_frames - missing,
        "n_frames_missing": missing,
        "n_detections_total": len(scores_arr),
        "score_percentiles": pct,
        "frame_coverage_by_threshold": {
            str(t): float(sum(label_frame_count[t].values()) / (n_frames - missing) / 1)
            for t in THRESHOLDS
        },
    }

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    rows = []
    for t in THRESHOLDS:
        for lb, cnt in label_frame_count[t].items():
            rows.append({"threshold": t, "label": lb, "n_frames": cnt})
    pd.DataFrame(rows).to_parquet(LABEL_FREQ_PATH, index=False)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nSo nhan (label) khac nhau tung nguong:")
    for t in THRESHOLDS:
        print(f"  >= {t}: {len(label_frame_count[t])} nhan")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
