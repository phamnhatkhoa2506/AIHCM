"""Tín hiệu củng cố chéo giữa OCR (chữ hiện trên màn hình) và ASR (lời nói) — việc C trong 3
việc khai thác ASR (xem hội thoại 2026-08-10). Nếu CÙNG LÚC lời nói VÀ chữ trên màn hình nhắc
tới cùng nội dung (vd ASR nói "bão số 3" VÀ OCR hiện chữ "BÃO SỐ 3" cùng khoảng thời gian) —
2 tín hiệu ĐỘC LẬP trùng khớp đáng tin hơn hẳn 1 tín hiệu đơn lẻ.

ƯU TIÊN THẤP (xem hội thoại) — chỉ nên wire vào scoring thật sau khi có dữ liệu OCR+ASR thật
chạy song song và thấy case cần, không thiết kế trước khi có bằng chứng. File này chỉ cung
cấp HÀM TIỆN ÍCH độc lập, chưa gọi từ query_planner.py.
"""
from __future__ import annotations

import resources

_STOPWORD_MIN_LEN = 3  # tu ngan hon bo qua khi so sanh trung lap - tranh khop nham qua tu
# chuc nang chung chung ("va", "la", "co"...). THAP HON nguong 4 dung o label_translate.py
# (BUG THAT phat hien 2026-08-10 khi test: "bao" trong "bao so 3" - tu NOI DUNG that, khong
# phai tu chuc nang - chi co 3 ky tu sau khi bo dau, bi loai oan boi nguong 4).


def _significant_tokens(text_norm: str) -> set[str]:
    # so LUON giu du ngan (vd "3" trong "bao so 3") - hau nhu khong bao gio la tu chuc nang.
    return {w for w in text_norm.split() if len(w) >= _STOPWORD_MIN_LEN or w.isdigit()}


def ocr_asr_agreement(video_id: str, local_idx: int, asr_window: int = 30) -> float:
    """Tra ve ty le token chung (0.0-1.0, kieu Jaccard-nhe: |giao| / min(|A|,|B|)) giua chu
    OCR dung frame nay va loi noi ASR gan frame nay. 0.0 neu thieu 1 trong 2 nguon hoac khong
    co token chung nao. KHONG phai xac suat that - chi la 1 tin hieu tho de dung THEM sau nay
    neu can (chua wire vao scoring o day)."""
    res = resources.get()
    ocr_index = res.ocr_index
    asr_index = res.asr_index
    if ocr_index is None or asr_index is None:
        return 0.0

    ocr_hit = ocr_index[
        (ocr_index["video_id"] == video_id)
        & (ocr_index["local_idx_start"] <= local_idx)
        & (ocr_index["local_idx_end"] >= local_idx)
    ]
    if ocr_hit.empty:
        return 0.0

    asr_hit = asr_index[
        (asr_index["video_id"] == video_id)
        & (asr_index["local_idx_start"] - asr_window <= local_idx)
        & (asr_index["local_idx_end"] + asr_window >= local_idx)
    ]
    if asr_hit.empty:
        return 0.0

    ocr_tokens: set[str] = set()
    for t in ocr_hit["text_norm"]:
        ocr_tokens |= _significant_tokens(t)
    asr_tokens: set[str] = set()
    for t in asr_hit["text_norm"]:
        asr_tokens |= _significant_tokens(t)

    if not ocr_tokens or not asr_tokens:
        return 0.0

    shared = ocr_tokens & asr_tokens
    return len(shared) / min(len(ocr_tokens), len(asr_tokens))
