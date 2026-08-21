"""Chuyển khung vẽ (px thô trong canvas_w x canvas_h) sang `spatial_boxes` mà search_dense()/
dense_temporal.search() nhận (xem tiers/dense_search.py) — y hệt logic build spatial_boxes cuối
`_render_filter_canvas` bên v3/online/app.py (phần vẽ canvas HTML5 đã port sang static/js/
canvas.js, không còn qua custom Streamlit component/objects_canvas.py nữa)."""
from __future__ import annotations

from label_translate import resolve as resolve_label_vi


def boxes_to_spatial(boxes: list, canvas_w: int, canvas_h: int) -> list[dict]:
    """boxes: list[CanvasBox-like] (x0,y0,x1,y1 px thô, kind "ocr"|"object", text/label/minCount).
    Trả spatial_boxes: [{"type":"ocr","text":...,"region":(ymin,xmin,ymax,xmax)}, ...] hoặc
    [{"type":"object","labels":[...],"region":...,"min_count":...}, ...] — region chuẩn hoá về
    [0,1], nhãn Object resolve qua đúng 514 nhãn closed-set (label_vi.json)."""
    spatial_boxes: list[dict] = []
    for b in boxes:
        x0, y0, x1, y1 = b.x0, b.y0, b.x1, b.y1
        xmin = max(0.0, min(x0, x1) / canvas_w)
        ymin = max(0.0, min(y0, y1) / canvas_h)
        xmax = min(1.0, max(x0, x1) / canvas_w)
        ymax = min(1.0, max(y0, y1) / canvas_h)
        region = (ymin, xmin, ymax, xmax)
        if b.kind == "ocr":
            text = (b.text or "").strip()
            if not text:
                continue
            spatial_boxes.append({"type": "ocr", "text": text, "region": region})
        else:
            label = (b.label or "").strip()
            if not label:
                continue
            hits = resolve_label_vi(label)
            min_count = max(1, int(b.minCount or 1))
            spatial_boxes.append({"type": "object", "labels": hits, "region": region, "min_count": min_count})
    return spatial_boxes
