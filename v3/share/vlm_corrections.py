"""Lưu các bản VLM đọc chữ ĐÃ ĐƯỢC NGƯỜI DÙNG DUYỆT THỦ CÔNG (2026-08-18, theo yêu cầu người
dùng: "thêm 1 cái trong quá trình dev, đó là chức năng approved text để lưu lại vào hệ thống
nhằm cải thiện ngôn ngữ, cái này chỉ được chấp thuận khi có sự đồng ý của mình, có nút để
approved chứ không để tự động").

KHÔNG BAO GIỜ tự động ghi — CHỈ ghi khi người dùng bấm nút "Duyệt" rõ ràng trong UI (xem
online/app.py::_render_vlm_ocr_verify). Mục đích: tích luỹ 1 tập dữ liệu "chữ đọc đúng đã người
xác nhận" cho các case PaddleOCR đọc sai/bỏ sót (xem hội thoại 2026-08-17/18 — case "AN HÒA
TỰ"/"CHÙA XỨ THÁNH MIẾU") — dùng làm tham khảo/patch thủ công sau này, KHÔNG tự động ghi đè
ocr_text.parquet (an toàn — chưa có cơ chế review/merge tự động nào ở đây)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from config import INDEX_DIR

VLM_APPROVED_OCR_PATH = INDEX_DIR / "dense" / "vlm_approved_ocr.jsonl"


def save_approved_vlm_text(video_id: str, frame_id: int, text: str, model: str, image_path: str) -> None:
    """Ghi THÊM (append, KHÔNG ghi đè) 1 dòng JSONL — giữ lại lịch sử nếu 1 frame được duyệt
    nhiều lần (vd duyệt lại sau khi đổi model VLM, hoặc PaddleOCR được chạy lại sau này)."""
    VLM_APPROVED_OCR_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "video_id": video_id,
        "frame_id": int(frame_id),
        "text": text,
        "model": model,
        "image_path": image_path,
        "approved_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(VLM_APPROVED_OCR_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_approved_vlm_texts() -> list[dict]:
    """Đọc lại TOÀN BỘ bản đã duyệt — dùng offline sau này nếu muốn soát/merge thủ công vào
    OCR index (chưa có script tự động — cố tình để người kiểm tra tay trước khi áp dụng, đúng
    tinh thần "chỉ chấp thuận khi có sự đồng ý")."""
    if not VLM_APPROVED_OCR_PATH.exists():
        return []
    out = []
    with open(VLM_APPROVED_OCR_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
