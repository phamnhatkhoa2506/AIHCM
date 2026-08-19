"""Lưu các mốc TRAKE đã được người dùng TINH CHỈNH bằng tay + DUYỆT thủ công (2026-08-20, theo
yêu cầu người dùng: "chức năng xem video của kết quả temporal để tinh chỉnh lại frame sau khi
đã có kết quả... approve frame mong muốn khi đã kiểm duyệt qua video") — GIỐNG HỆT pattern
`vlm_corrections.py` (JSONL append-only, CHỈ ghi khi người dùng bấm nút Duyệt, KHÔNG tự động
áp dụng lại vào thuật toán xếp hạng — đây là bản ghi tham khảo/cải thiện dữ liệu về sau, xem
`app.py::_render_trake_frame_tune` cho phần override HIỂN THỊ ngay trong phiên hiện tại)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from config import DENSE_DIR

TRAKE_APPROVED_PATH = DENSE_DIR / "trake_approved_corrections.jsonl"


def save_trake_correction(
    video_id: str,
    anchor_index: int,
    anchor_text: str,
    old_frame_id: int,
    new_frame_id: int,
    new_pts_time: float,
) -> None:
    """Ghi THÊM (append, KHÔNG ghi đè) 1 dòng JSONL — giữ lại lịch sử nếu 1 mốc được tinh
    chỉnh/duyệt nhiều lần (khác lần trước khác nhau), không mất bản ghi cũ."""
    TRAKE_APPROVED_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "video_id": video_id,
        "anchor_index": anchor_index,
        "anchor_text": anchor_text,
        "old_frame_id": old_frame_id,
        "new_frame_id": new_frame_id,
        "new_pts_time": new_pts_time,
        "approved_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(TRAKE_APPROVED_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_trake_corrections() -> list[dict]:
    if not TRAKE_APPROVED_PATH.exists():
        return []
    with open(TRAKE_APPROVED_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
