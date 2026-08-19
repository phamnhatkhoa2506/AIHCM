"""Parse media-info/*.json (YouTube metadata) thành 1 bảng nhỏ dùng để lọc theo
kênh (author)/từ khoá (keywords)/ngày đăng (publish_date) trước khi vào Tier 1 vector search.

Không phải mọi kênh đều có keywords hữu ích (kênh tin tức "60 Giây" description/tiêu đề
lặp khuôn, keywords=None — chỉ publish_date phân biệt được); kênh khác (vd ViVU TV,
chiếm đa số corpus) có keywords khá đặc trưng theo nội dung. Bảng này giữ nguyên cả 2
loại, để search.py quyết định lọc theo field nào tuỳ query.
"""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/

import json
import sys

import pandas as pd

from config import INDEX_DIR, MEDIA_INFO_DIR

METADATA_PATH = INDEX_DIR / "video_metadata.parquet"


def build() -> None:
    files = sorted(MEDIA_INFO_DIR.glob("*.json"))
    if not files:
        raise SystemExit(f"Không tìm thấy media-info nào trong {MEDIA_INFO_DIR}")

    rows = []
    for fp in files:
        video_id = fp.stem
        with open(fp, encoding="utf-8") as f:
            d = json.load(f)

        publish_date = pd.to_datetime(d.get("publish_date"), format="%d/%m/%Y", errors="coerce")
        keywords = d.get("keywords") or []

        rows.append(
            {
                "video_id": video_id,
                "author": d.get("author"),
                "title": d.get("title"),
                "publish_date": publish_date,
                "keywords": keywords,
                "keywords_text": " ".join(keywords).lower() if keywords else "",
            }
        )

    df = pd.DataFrame(rows)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(METADATA_PATH, index=False)

    print(f"Xong: {len(df)} video, {df['author'].nunique()} kenh")
    print(df["author"].value_counts().to_string())


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    build()
