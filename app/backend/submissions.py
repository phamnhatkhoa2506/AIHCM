"""Quản lý danh sách nộp bài — tương đương `st.session_state["submissions_by_query"]` bên
Streamlit cũ (1 bucket/câu hỏi, xem online/app.py `_active_submission_list`). Lưu IN-MEMORY
theo session_id (header `X-Session-Id`, frontend tự sinh + lưu localStorage) — đủ dùng cho 1
máy chấm thi chạy local; không cần DB cho giai đoạn 1."""
from __future__ import annotations

from collections import defaultdict

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse

from backend.schemas import AutofillRequest, AutofillResponse, SubmissionItem

router = APIRouter()

# session_id -> query_key -> list[SubmissionItem]
_STORE: dict[str, dict[str, list[SubmissionItem]]] = defaultdict(lambda: defaultdict(list))

# Khớp đúng SUBMISSION_MAX bên v3 (online/app.py) - tối đa 100 dòng/câu hỏi (= SUBMISSION_TOP_K
# bên submission_pipeline.py, đúng số BTC nhận R@{1,5,20,50,100}).
SUBMISSION_MAX = 100


def _row_key(it: SubmissionItem) -> tuple:
    """Key chống trùng 1 dòng nộp bài - giống hệt `_submission_row_key` bên v3 (theo mode vì
    kis/qa/trake có schema khác nhau: trake so theo frame_ids, còn lại theo frame_id)."""
    if it.frame_ids:
        return (it.mode, it.video_id, tuple(it.frame_ids))
    return (it.mode, it.video_id, it.frame_id)


def _sid(x_session_id: str | None) -> str:
    if not x_session_id:
        raise HTTPException(400, "Thiếu header X-Session-Id")
    return x_session_id


@router.get("/{query_key}")
def list_items(query_key: str, x_session_id: str | None = Header(default=None)):
    return _STORE[_sid(x_session_id)][query_key]


@router.post("/{query_key}")
def add_item(query_key: str, item: SubmissionItem, x_session_id: str | None = Header(default=None)):
    lst = _STORE[_sid(x_session_id)][query_key]
    lst.append(item)
    return lst


@router.post("/{query_key}/toggle")
def toggle_item(query_key: str, item: SubmissionItem, x_session_id: str | None = Header(default=None)):
    """Y HET `_render_submit_button` bên v3 - nút "📤 Nộp" TỰ CHUYỂN thành "✅ Đã nộp" khi frame
    này đã có trong danh sách, bấm lại = HOÀN TÁC (xoá khỏi danh sách, không phải khoá cứng).
    Xoá theo NỘI DUNG (_row_key: mode/video_id/frame_id(s)), không phải index - card kết quả
    không biết vị trí của nó trong danh sách nộp bài (danh sách có thể đã bị sắp xếp lại)."""
    lst = _STORE[_sid(x_session_id)][query_key]
    key = _row_key(item)
    for idx, r in enumerate(lst):
        if _row_key(r) == key:
            lst.pop(idx)
            return {"submitted": False, "items": lst}
    if len(lst) >= SUBMISSION_MAX:
        raise HTTPException(400, f"Danh sách nộp bài đã đủ {SUBMISSION_MAX} dòng.")
    lst.append(item)
    return {"submitted": True, "items": lst}


@router.delete("/{query_key}/{index}")
def remove_item(query_key: str, index: int, x_session_id: str | None = Header(default=None)):
    lst = _STORE[_sid(x_session_id)][query_key]
    if not (0 <= index < len(lst)):
        raise HTTPException(404, "Index không hợp lệ")
    lst.pop(index)
    return lst


@router.post("/{query_key}/move/{index}")
def move_item(query_key: str, index: int, direction: str, x_session_id: str | None = Header(default=None)):
    lst = _STORE[_sid(x_session_id)][query_key]
    j = index - 1 if direction == "up" else index + 1
    if not (0 <= index < len(lst) and 0 <= j < len(lst)):
        raise HTTPException(400, "Không thể di chuyển")
    lst[index], lst[j] = lst[j], lst[index]
    return lst


@router.post("/{query_key}/autofill", response_model=AutofillResponse)
def autofill(query_key: str, req: AutofillRequest, x_session_id: str | None = Header(default=None)):
    """Tự động điền — thêm các dòng CHƯA CÓ trong danh sách, theo ĐÚNG THỨ TỰ rank (top -> thấp,
    thứ tự `req.items` truyền vào = thứ tự đang hiển thị trên lưới kết quả), cho tới khi đủ
    SUBMISSION_MAX (không tách quota riêng với nộp tay - "tổng frame submit thủ công + tự động
    = 100"), giống hệt `_render_autofill_button` bên v3."""
    lst = _STORE[_sid(x_session_id)][query_key]
    existing_keys = {_row_key(it) for it in lst}
    added = 0
    remaining = SUBMISSION_MAX - len(lst)
    for it in req.items:
        if added >= remaining:
            break
        key = _row_key(it)
        if key in existing_keys:
            continue
        lst.append(it)
        existing_keys.add(key)
        added += 1
    return AutofillResponse(added=added, items=lst)


@router.delete("/{query_key}")
def clear(query_key: str, x_session_id: str | None = Header(default=None)):
    _STORE[_sid(x_session_id)][query_key] = []
    return []


def _submission_to_csv(lst: list[SubmissionItem]) -> str:
    """Y HET `_submission_to_csv` ben v3 (online/app.py) - KHONG dung csv.writer (dialect excel
    mac dinh noi "\\r\\n" + QUOTE_MINIMAL, khac hanh vi v3) - noi dong thu cong bang "\\n", KIS/
    TRAKE khong bao gio quote, QA LUON boc ngoac kep + tu escape ngoac kep BEN TRONG (nhan doi
    "" - chuan CSV) de dung dinh dang BTC yeu cau tuyet doi, khong lech du chi 1 ky tu."""
    lines = []
    for it in lst:
        if it.frame_ids:  # TRAKE
            lines.append(",".join([it.video_id, *(str(f) for f in it.frame_ids)]))
        elif it.answer_text is not None:  # Q&A
            answer = str(it.answer_text).replace('"', '""')
            lines.append(f'{it.video_id},{it.frame_id},"{answer}"')
        else:  # KIS
            lines.append(f"{it.video_id},{it.frame_id}")
    return "\n".join(lines)


@router.get("/{query_key}/export.csv")
def export_csv(
    query_key: str, filename: str | None = None, session_id: str | None = None,
    x_session_id: str | None = Header(default=None),
):
    """filename: TUỲ CHỌN, lấy từ ô "Name" phía frontend (thuần cosmetic, xem sp-name trong
    index.html) - KHÔNG dùng để tra cứu dữ liệu (khác `query_key`, vốn là ID ổn định) - đúng
    tinh thần tách "Name" khỏi khoá lưu trữ như bên v3 (xem _active_submission_key).

    session_id (query param): 2026-08-21 (bug thật - "vẫn chưa tải được") - endpoint này được
    frontend gọi qua `window.location.href` (điều hướng trình duyệt thường để trigger tải file
    kèm đúng Content-Disposition/filename) - CHỨ KHÔNG QUA fetch(), nên KHÔNG gắn được header
    tuỳ chỉnh X-Session-Id như mọi endpoint /api/submissions/* khác -> luôn 400 "Thiếu header
    X-Session-Id". Chấp nhận thêm session id qua query param (chỉ endpoint tải file này cần,
    các endpoint khác vẫn gọi qua fetch() nên header vẫn hoạt động bình thường)."""
    lst = _STORE[_sid(x_session_id or session_id)][query_key]
    name = (filename or query_key).strip() or query_key
    if not name.lower().endswith(".csv"):
        name += ".csv"
    content = _submission_to_csv(lst)
    return StreamingResponse(
        iter([content]), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )
