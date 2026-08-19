"""Chạy P1 (extract_frame) trên vài frame ở GIỮA video (tránh đoạn dạo đầu chương trình,
thường lặp khuôn/logo), lưu kết quả (relation + extra_observations) ra JSON để soát tay.
"""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/

import json
import sys

from config import INDEX_DIR
from p1_extract import SpatialIndex, extract_frame

FRAMES = [
    ("L26_V475", 78),
    ("L26_V255", 81),
    ("L24_V040", 33),
    ("L26_V086", 71),
    ("L26_V264", 81),
    ("L26_V246", 78),
    ("L30_V044", 47),
    ("L26_V069", 79),
]

OUT_PATH = INDEX_DIR / "sample_middle_frames_review.json"


def main() -> None:
    sp = SpatialIndex.load()
    results = []

    for i, (vid, li) in enumerate(FRAMES, 1):
        print(f"[{i}/{len(FRAMES)}] {vid}/{li} ...", file=sys.stderr)
        try:
            relations, extra = extract_frame(vid, li, sp)
            results.append(
                {"video_id": vid, "local_idx": li, "relations": relations, "extra_observations": extra}
            )
        except Exception as e:
            print(f"  loi: {e}", file=sys.stderr)
            results.append({"video_id": vid, "local_idx": li, "error": str(e)})

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Da luu -> {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
