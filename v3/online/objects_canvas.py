"""Custom Streamlit component: canvas vẽ khung + GÕ CHỮ TRỰC TIẾP ngay trên khung (2026-08-15,
theo yêu cầu người dùng "kéo thả như ảnh mẫu, chữ in trực tiếp chứ không nhập input ngoài").

THAY THẾ streamlit-drawable-canvas (fabric.js) — thư viện đó không có drawing_mode="text" và
initial_drawing chỉ áp dụng 1 LẦN LÚC MOUNT nên không thể "bake" chữ vào canvas đang vẽ được.
Ở đây viết thẳng HTML5 canvas + JS thuần (không React/npm build) — giao tiếp với Streamlit qua
đúng giao thức postMessage nội bộ (isStreamlitMessage/streamlit:setComponentValue/...) mà mọi
custom component chính thức đều dùng dưới lớp React, chỉ là viết tay thay vì qua thư viện.

KHÔNG cần build step: components.declare_component(path=...) phục vụ thẳng index.html tĩnh.
"""
from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

from label_translate import list_labels_vi

_COMPONENT_DIR = Path(__file__).parent / "objects_canvas_component"
_component_func = components.declare_component("objects_canvas", path=str(_COMPONENT_DIR))


def objects_canvas(
    key: str,
    width: int = 480,
    height: int = 270,
    boxes: list[dict] | None = None,
) -> list[dict]:
    """Vẽ 1 canvas 2 CHẾ ĐỘ (2026-08-15, theo yêu cầu người dùng "không tự suy luận, custom 2
    loại box: xanh lá = OCR gõ tự do, vàng = Object có dropdown 514 nhãn gắn liền trên box" —
    thay hẳn cơ chế suy luận tự động trước đó, vì người dùng thấy KHÔNG AN TOÀN/lãng phí và dễ
    bị giám khảo trừ điểm nếu suy đoán sai).

    Trả list[dict], mỗi box:
      {"x0","y0","x1","y1": px thô trong không gian width x height,
       "kind": "ocr" | "object",
       "text": chữ OCR (kind="ocr"),
       "label": nhãn tiếng Việt đã chọn từ dropdown (kind="object", "" nếu chưa chọn),
       "minCount": số lượng tối thiểu (kind="object", mặc định 1)}

    `boxes`: giá trị khởi tạo (chỉ áp dụng LÚC MOUNT đầu tiên bên phía JS — đổi sau đó KHÔNG
    ghi đè lên khung người dùng đang thao tác, xem index.html)."""
    result = _component_func(
        key=key, width=width, height=height, boxes=boxes or [],
        labels=list_labels_vi(), default=boxes or [],
    )
    return result or []
