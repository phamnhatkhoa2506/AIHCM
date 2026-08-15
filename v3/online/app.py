"""Giao diện đơn giản: nhập query text -> hiển thị top-k frame khớp nhất (ảnh + info).

Chạy: streamlit run app.py
"""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/

import time

import streamlit as st

from label_translate import resolve as resolve_label_vi
from objects_canvas import objects_canvas
from query_planner import extract_entities
from steplog import StepLog
from submission_pipeline import answer_qa
from tiers import dense_temporal
from tiers.dense_search import _fps_by_video, apply_region_clip_rerank, search_dense
from video_clip import get_clip_bytes

st.set_page_config(page_title="AIC 2026 - Dense Search", layout="wide")

# Nhan hien thi (UI) -> gia tri that search_dense() can (DENSE_MODES trong dense_search.py) -
# radio doi sang nhan than thien ("All"/"SigLIP2"/...) nhung backend van nhan "rrf"/"siglip"/...
_DENSE_MODEL_DISPLAY_TO_VALUE = {"All": "rrf", "SigLIP2": "siglip", "PE-Core": "pe_core", "BEiT-3": "beit3"}


def _render_video_toggle(video_id: str, frame_id: int, widget_key: str) -> None:
    """Nút "▶ Video" dưới mỗi keyframe — bấm mới cắt/phát đoạn video ngắn quanh frame đó (2026-
    08-15, theo yêu cầu người dùng "keyframe hiển thị có thể chạy video ... BTC chấm UI cao
    hơn"). CHỦ ĐỘNG lazy: KHÔNG tự cắt clip cho mọi dòng kết quả (top_k có thể tới 500 dòng) —
    chỉ cắt khi người dùng bấm đúng dòng đó, xem video_clip.py."""
    show_key = f"show_video_{widget_key}"
    if st.button("▶ Video", key=f"vidbtn_{widget_key}"):
        st.session_state[show_key] = not st.session_state.get(show_key, False)
    if st.session_state.get(show_key):
        fps = _fps_by_video().get(video_id, 25.0)
        t = frame_id / fps
        with st.spinner("Đang cắt đoạn video quanh frame..."):
            try:
                st.video(get_clip_bytes(video_id, t))
            except Exception as e:
                st.error(f"Không cắt được video ({e})")

with st.sidebar:
    st.title("AIC 2026 — VAIKADAI")
    st.caption("Hệ thống video search cực kỳ thông minh do Anh Khoa, Kiên, Đức, Viên và The Liem làm.")

    # 2026-08-15 (theo yeu cau nguoi dung): "Loai truy van" chuyen len DAU sidebar, tren cung.
    mode = st.radio(
        "Loại truy vấn",
        ["KIS", "TRAKE", "Q&A"],
    )
    is_temporal = mode == "TRAKE"
    is_qa = mode == "Q&A"

    st.divider()
    st.header("Bộ lọc (tuỳ chọn)")
    author_options = ["ViVU TV", "Báo Tuổi Trẻ", "Báo Thanh Niên", "HTV Sports", "60 Giây Official", "HTV Entertainment", "HTV Giải Trí"]
    authors = st.multiselect("Kênh", author_options)
    keywords_raw = st.text_input(
        "Từ khoá / tên chương trình (phân cách bằng dấu phẩy)",
        help="Khớp cả tiêu đề video lẫn tag YouTube gốc, không phân biệt dấu (gõ 'mon ngon' vẫn khớp 'Món Ngon').",
    )
    col1, col2 = st.columns(2)
    date_from = col1.date_input("Từ ngày", value=None)
    date_to = col2.date_input("Đến ngày", value=None)

    st.divider()
    dense_model_display = st.radio(
        "Model xếp hạng", list(_DENSE_MODEL_DISPLAY_TO_VALUE), horizontal=True,
        help="All: fusion Reciprocal Rank Fusion trên cả 3 model — ổn định hơn vì không phụ "
        "thuộc 1 model duy nhất. SigLIP2/PE-Core/BEiT-3: dùng đúng 1 model (SigLIP2 thắng "
        "benchmark rõ rệt trong 3 model riêng lẻ).",
    )
    dense_model = _DENSE_MODEL_DISPLAY_TO_VALUE[dense_model_display]

    # Sync 2 chieu slider <-> number_input (2026-08-15, theo yeu cau nguoi dung: max tang len
    # 500 + can 1 o nhap tay de kiem soat chinh xac hon keo slider) - dung pattern callback
    # ghi CHEO qua key cua widget kia (Streamlit khong tu dong bo 2 widget doc lap).
    def _sync_top_k_from_slider():
        st.session_state.top_k_input = st.session_state.top_k_slider

    def _sync_top_k_from_input():
        st.session_state.top_k_slider = st.session_state.top_k_input

    if "top_k_slider" not in st.session_state:
        st.session_state.top_k_slider = 12
        st.session_state.top_k_input = 12

    col_topk_slider, col_topk_input = st.columns([3, 1])
    col_topk_slider.slider(
        "Số kết quả", min_value=5, max_value=500, step=1,
        key="top_k_slider", on_change=_sync_top_k_from_slider,
    )
    col_topk_input.number_input(
        "Số kết quả", min_value=5, max_value=500, step=1,
        key="top_k_input", on_change=_sync_top_k_from_input, label_visibility="collapsed",
    )
    top_k = st.session_state.top_k_slider

spatial_op_display = st.radio(
    "Quan hệ giữa các khung", ["AND", "OR"],
    horizontal=True,
    help="Áp dụng cho NỘI DUNG (chữ/nhãn) — vẫn lọc cứng: AND = frame phải có ĐỦ nội dung mọi "
    "khung, OR = chỉ cần khớp 1 khung bất kỳ. VỊ TRÍ vẽ (khung ở đâu trên ảnh) KHÔNG còn lọc "
    "cứng nữa (2026-08-15) — chỉ cộng thêm điểm nếu khớp đúng vị trí thật, vẽ lệch tay vẫn giữ "
    "nguyên kết quả đúng, không bị loại oan. Áp dụng chung cho mọi canvas (kể cả từng mốc TRAKE).",
)
spatial_op = "and" if spatial_op_display.startswith("AND") else "or"
_CANVAS_W, _CANVAS_H = 480, 270  # ti le 16:9, co dinh de chuan hoa toa do ve [0,1] - thu nho
# lai (tu 720x405) de canvas + panel input NAM NGANG vua 1 hang (theo yeu cau nguoi dung
# 2026-08-15), khong con full-width nhu truoc.


def _render_filter_canvas(key_suffix: str) -> list[dict]:
    """1 canvas OCR+Object dùng lại được (KIS/Q&A dùng 1 lần với key_suffix="global", TRAKE
    dùng N lần - 1 lần/mốc, key_suffix=f"trake_{i}").

    2026-08-15 (theo yêu cầu người dùng — CUSTOM 2 LOẠI BOX rõ ràng thay vì tự suy luận: "làm
    thế mình thấy ko an toàn và là sự lãng phí, chưa kể là điểm trừ trong mắt giám khảo"):
    chuyển từ streamlit-drawable-canvas (fabric.js, không hỗ trợ gõ chữ ngay trên khung) sang
    objects_canvas — component HTML5 canvas viết tay (xem online/objects_canvas_component/
    index.html) với 2 CHẾ ĐỘ VẼ tường minh:
      - 🟢 OCR: gõ chữ tự do ngay trên khung, bake trực tiếp vào canvas.
      - 🟡 Object: dropdown CHỌN THẲNG từ đúng 514 nhãn closed-set (label_vi.json) gắn liền
        ngay trên khung + ô số lượng tối thiểu — KHÔNG còn suy luận/gợi ý mờ nào, người dùng tự
        chọn màu/loại trước khi vẽ nên không có khả năng nhận nhầm loại khung."""
    # 2026-08-15 (theo yeu cau nguoi dung "ben phai bi du nhieu qua") - component canvas co
    # chieu rong CO DINH (_CANVAS_W=480px, khong gian ra theo container) nen neu de full-width
    # se thua rat nhieu khoang trong ben phai trong the (container) rong. Dat canvas vao cot
    # trai HEP VUA DU, danh sach tom tat khung ve sang cot phai TAN DUNG khoang trong do.
    col_canvas, col_list = st.columns([1, 1])
    with col_canvas:
        raw_boxes = objects_canvas(key=f"objects_canvas_{key_suffix}", width=_CANVAS_W, height=_CANVAS_H)

    spatial_boxes: list[dict] = []
    with col_list:
        if raw_boxes:
            st.caption(f"{len(raw_boxes)} khung đã vẽ:")
            for i, b in enumerate(raw_boxes):
                xmin = max(0.0, min(b["x0"], b["x1"]) / _CANVAS_W)
                ymin = max(0.0, min(b["y0"], b["y1"]) / _CANVAS_H)
                xmax = min(1.0, max(b["x0"], b["x1"]) / _CANVAS_W)
                ymax = min(1.0, max(b["y0"], b["y1"]) / _CANVAS_H)
                region = (ymin, xmin, ymax, xmax)
                if b["kind"] == "ocr":
                    text = b["text"].strip()
                    if not text:
                        st.caption(f"Khung #{i + 1}: 🟢 _(chưa gõ chữ — bỏ qua)_")
                        continue
                    st.caption(f"Khung #{i + 1}: 🔤 OCR \"{text}\"")
                    spatial_boxes.append({"type": "ocr", "text": text, "region": region})
                else:
                    label = b["label"].strip()
                    if not label:
                        st.caption(f"Khung #{i + 1}: 🟡 _(chưa chọn nhãn — bỏ qua)_")
                        continue
                    # dropdown chi cho chon dung 514 nhan co san -> resolve LUON thanh cong,
                    # khong can canh bao "khong map duoc" nhu duong text tu do truoc day.
                    hits = resolve_label_vi(label)
                    min_count = int(b.get("minCount") or 1)
                    disp = f"{label} ×{min_count}" if min_count > 1 else label
                    st.caption(f"Khung #{i + 1}: 🔲 Object {disp}")
                    spatial_boxes.append({"type": "object", "labels": hits, "region": region, "min_count": min_count})
        else:
            st.caption("_Chưa vẽ khung nào._")
    return spatial_boxes


if not is_temporal:
    st.caption("Khung kéo thả object/OCR áp dụng cho TOÀN BỘ kết quả:")
    spatial_boxes = _render_filter_canvas("global")
else:
    spatial_boxes = None  # TRAKE: khong con canvas chung - moi moc co canvas RIENG (xem duoi)
st.divider()

if is_temporal:
    query = ""
    st.caption("Mỗi ô = 1 khoảnh khắc, ĐÚNG thứ tự thời gian (vd: giậm nhảy / bay qua xà / tiếp đất)")
    # UI dong (2026-08-11) - thay the textarea tinh bang danh sach co the them/xoa/sap xep
    # tung moc rieng, dung st.session_state de giu trang thai giua cac lan rerun cua Streamlit.
    # moi moc gio la dict {"text","labels_raw","ocr_raw"} - khong con string tinh, de mang
    # theo filter RIENG cho tung khoanh khac (2026-08-11, xem tier3_temporal.py::_normalize_anchor)
    if "trake_anchors" not in st.session_state:
        st.session_state.trake_anchors = [
            {"text": "", "labels_raw": "", "ocr_raw": "", "spatial_boxes": []},
            {"text": "", "labels_raw": "", "ocr_raw": "", "spatial_boxes": []},
        ]

    for i in range(len(st.session_state.trake_anchors)):
        anc = st.session_state.trake_anchors[i]
        # 2026-08-15 (theo yeu cau nguoi dung "nhieu o roi rac qua, dinh kem input/nhan/khung
        # ve lai voi nhau") - BOC CA MOC (text mo ta + nut sap xep + bo loc rieng + canvas vao
        # 1 st.container(border=True) DUY NHAT -> doc thanh 1 THE lien khoi thay vi cac dong
        # thanh phan roi rac tach biet nhu truoc. Canvas van nam TRUC TIEP trong container nay
        # (KHONG long trong expander) de khong dinh bug mount height=0 (xem docstring
        # _render_filter_canvas) - container thuong (khong collapse duoc) khong dinh bug do.
        with st.container(border=True):
            col_badge, col_text, col_up, col_down, col_del = st.columns([1, 9, 1, 1, 1])
            col_badge.markdown(f"**Mốc {i + 1}**")
            anc["text"] = col_text.text_input(
                f"Mốc {i + 1}", value=anc["text"], key=f"trake_anchor_{i}",
                label_visibility="collapsed", placeholder=f"Mô tả mốc {i + 1} (vd: giậm nhảy)",
            )
            if col_up.button("↑", key=f"trake_up_{i}", disabled=(i == 0)):
                st.session_state.trake_anchors[i - 1], st.session_state.trake_anchors[i] = (
                    st.session_state.trake_anchors[i], st.session_state.trake_anchors[i - 1]
                )
                st.rerun()
            if col_down.button("↓", key=f"trake_down_{i}", disabled=(i == len(st.session_state.trake_anchors) - 1)):
                st.session_state.trake_anchors[i + 1], st.session_state.trake_anchors[i] = (
                    st.session_state.trake_anchors[i], st.session_state.trake_anchors[i + 1]
                )
                st.rerun()
            if col_del.button("🗑", key=f"trake_del_{i}", disabled=(len(st.session_state.trake_anchors) <= 2)):
                st.session_state.trake_anchors.pop(i)
                st.rerun()

            with st.expander("Lọc chữ/nhãn phụ (tuỳ chọn, ngoài khung vẽ bên dưới)", expanded=False):
                anc["labels_raw"] = st.text_input(
                    "Nhãn bắt buộc riêng cho mốc này (vd: người, xà)", value=anc["labels_raw"],
                    key=f"trake_labels_{i}",
                    help="Chỉ áp dụng cho khoảnh khắc này, KHÔNG áp dụng cho cả chuỗi.",
                )
                anc["ocr_raw"] = st.text_input(
                    "Chữ trên màn hình riêng cho mốc này", value=anc["ocr_raw"], key=f"trake_ocr_{i}",
                )

            st.caption("Khung vẽ object/OCR cho mốc này:")
            anc["spatial_boxes"] = _render_filter_canvas(f"trake_{i}")

    if st.button("+ Thêm mốc"):
        st.session_state.trake_anchors.append({"text": "", "labels_raw": "", "ocr_raw": "", "spatial_boxes": []})
        st.rerun()

    anchors_raw = ""  # khong con dung duong text-join cu - xem cho xay dung `anchors` ben duoi
    question = ""
elif is_qa:
    query = st.text_input("Mô tả sự kiện", placeholder="vd: một người đang nấu ăn với chảo và thìa")
    question = st.text_input("Câu hỏi", placeholder="vd: người trong ảnh đang cầm vật gì?")
    vqa_top_n = st.number_input(
        "Số ứng viên đầu gọi VQA thật (tốn phí/lần)", min_value=1, max_value=20, value=3, step=1
    )
    # 2026-08-15 (theo yeu cau nguoi dung "tich hop Rerank nhu KIS") - cung 1 checkbox/logic voi
    # nhanh KIS ben duoi (else), chi khac la hien o day cho mode Q&A.
    use_region_clip_rerank = st.checkbox(
        "Region-CLIP rerank theo thuộc tính (màu/quần áo..., SigLIP2)",
        value=True,
        help="Ép ưu tiên frame khớp thuộc tính LLM trích được (vd 'áo dài màu tím') TRƯỚC khi "
        "hỏi VQA thật, không chỉ dựa vào CLIP đoán cả câu. Chạy qua server Modal riêng, ~2s/lần.",
    )
    anchors_raw = ""
else:
    query = st.text_input("Query", placeholder="VD: Một diễn giả mặc áo đỏ phát biểu tại cuộc họp báo ngoài trời")
    anchors_raw = ""
    question = ""
    use_region_clip_rerank = st.checkbox(
        "Region-CLIP rerank theo thuộc tính (màu/quần áo..., SigLIP2)",
        value=True,
        help="Ép ưu tiên frame khớp thuộc tính LLM trích được (vd 'áo dài màu tím'), không chỉ "
        "dựa vào CLIP đoán cả câu. Chạy qua server Modal riêng (aic2026-region-rerank, luôn "
        "giữ ấm — 2026-08-15) nên không còn tốn ~1-2 phút nạp file 5,3GB trên máy bạn nữa, chỉ "
        "còn ~2s/lần. Tắt nếu không cần thuộc tính.",
    )

run = st.button("Tìm kiếm", type="primary")

if is_temporal:
    # xay dung anchors dang dict THAT (khong qua text-join/split nua) - tu session_state,
    # bo qua moc rong (text trong) giong het logic cu (a.strip() if a.strip()).
    anchors = []
    for anc in st.session_state.trake_anchors:
        text = anc["text"].strip()
        if not text:
            continue
        labels_terms = [t.strip() for t in anc["labels_raw"].split(",") if t.strip()]
        resolved_labels = []
        for t in labels_terms:
            resolved_labels.extend(resolve_label_vi(t))
        entry: dict = {"text": text}
        if resolved_labels:
            entry["must_have_labels"] = resolved_labels
        if anc["ocr_raw"].strip():
            entry["ocr_text"] = anc["ocr_raw"].strip()
        if anc.get("spatial_boxes"):
            entry["spatial_boxes"] = anc["spatial_boxes"]
        anchors.append(entry)
else:
    anchors = []

if is_temporal:
    ready = len(anchors) >= 2
elif is_qa:
    ready = bool(query.strip() and question.strip())
else:
    ready = bool(query.strip())

if run and not ready:
    st.warning("Nhập query (hoặc >=2 dòng anchor nếu ở chế độ chuỗi sự kiện) trước đã.")

if run and ready:
    keywords_any = [k.strip() for k in keywords_raw.split(",") if k.strip()] or None

    # 2026-08-15: ca 3 mode (KIS/Q&A/Temporal) gio DEU chay tren bo dense, DEU nhan spatial_boxes
    # day du (nhieu khung, co vi tri, AND/OR) - khong con can flatten rieng must_have_labels/
    # ocr_text_first cho "nhanh BTC cu" nua (da migrate het). must_have_labels/min_count o day
    # CHI con dung lam gia tri KHOI TAO cho merge voi LLM entities o nhanh KIS/Q&A ben duoi
    # (xem "merged_must_have").
    must_have_labels = None
    min_count = None
    _obj_boxes = [b for b in spatial_boxes if b["type"] == "object"]
    if _obj_boxes:
        must_have_labels = sorted({lb for b in _obj_boxes for lb in b["labels"]})
        min_count = {}
        for b in _obj_boxes:
            for lb in b["labels"]:
                min_count[lb] = max(min_count.get(lb, 1), b["min_count"])

    common_filters = dict(
        authors=authors or None,
        date_from=str(date_from) if date_from else None,
        date_to=str(date_to) if date_to else None,
        keywords_any=keywords_any,
    )

    log = StepLog()
    t_start = time.perf_counter()
    plan = None
    with st.spinner("Đang tìm..." if not is_qa else "Đang tìm + hỏi VQA (có thể mất chút thời gian)..."):
        if is_qa:
            # 2026-08-15: truyen THEM spatial_boxes/spatial_op (khung ve OCR/Object da nang,
            # AND/OR, soft-boost vi tri) - truoc do BI THIEU (common_filters chi mang ocr_text
            # tu khung OCR DAU TIEN kieu cu, khong mang het cac khung tren canvas) - phat hien
            # qua cau hoi nguoi dung "Q&A co dung duoc khung ve khong".
            results = answer_qa(query, question, top_k=top_k, vqa_top_n=int(vqa_top_n),
                                 dense_model=dense_model, use_region_clip_rerank=use_region_clip_rerank,
                                 spatial_boxes=spatial_boxes or None, spatial_op=spatial_op,
                                 log=log, **common_filters)
        elif is_temporal:
            # 2026-08-15: TRAKE gio co 1 canvas RIENG cho MOI moc (xem vong lap trake_anchors o
            # tren) - spatial_boxes o day la None (khong con canvas CHUNG cho ca chuoi), tung
            # anchor dict trong `anchors` da tu mang theo "spatial_boxes" cua rieng no (xem
            # dense_temporal.py::_normalize_anchor/search - merge per-anchor, khong con global).
            results = dense_temporal.search(
                anchors, top_k=top_k, dense_model=dense_model,
                spatial_boxes=spatial_boxes, spatial_op=spatial_op, log=log,
            )
        else:
            # 2026-08-15 (theo yeu cau nguoi dung: bo checkbox, LUON BAT LLM tach entity) -
            # extract_entities() (query_planner.py) LUON chay truoc, gop must_have_labels/
            # min_count trich duoc VOI cai tu khung Object tren canvas (UNION nhan, MAX
            # min_count) - roi giao het cho search_dense (bo dense, KHONG qua planned_search/
            # search() BTC cu nua).
            plan = extract_entities(query, log=log)
            merged_must_have = list({*(must_have_labels or []), *plan["resolved_must_have_labels"]}) or None
            merged_min_count = {**(min_count or {}), **plan["resolved_min_count"]} or None

            # Region-CLIP rerank (2026-08-15, MOI wire, CHI SigLIP2 - xem dense_search.py::
            # apply_region_clip_rerank) - TUY CHON (checkbox use_region_clip_rerank, theo yeu
            # cau nguoi dung: lan dau/phien co the cham ~1-2 phut do nap file embedding 5.3GB,
            # nen de nguoi dung tu bat/tat thay vi luon chay). Can pool RONG HON top_k de
            # re-rank co gi ma chon (giong pattern query_planner.py cu), roi cat lai dung top_k.
            attributes = (plan.get("attributes") or []) if use_region_clip_rerank else []
            search_top_k = top_k * 4 if attributes else top_k
            results = search_dense(
                query, dense_model, top_k=search_top_k, ocr_text=None,
                authors=authors or None,
                date_from=str(date_from) if date_from else None,
                date_to=str(date_to) if date_to else None,
                keywords_any=keywords_any, must_have_labels=merged_must_have, min_count=merged_min_count,
                spatial_boxes=spatial_boxes or None, spatial_op=spatial_op,
                audio_mentions=plan.get("audio_mentions") or None, log=log,
            )
            if attributes and not results.empty:
                results = apply_region_clip_rerank(results, attributes, top_k, log=log)
            elif log and plan.get("attributes") and not use_region_clip_rerank:
                log.add("Region-CLIP", "TẮT (checkbox người dùng) — bỏ qua bước rerank theo thuộc tính", 0.0)
    elapsed = time.perf_counter() - t_start

    if plan is not None:
        with st.expander("Xem plan đã phân rã (LLM)", expanded=True):
            st.write(f"**Entity trích được:** {plan.get('entities')}")
            st.write(f"**Đã resolve sang nhãn:** {plan.get('resolved_must_have_labels')} "
                     f"(min_count={plan.get('resolved_min_count')})")
            if plan.get("unresolved"):
                st.warning(f"Không resolve được (bỏ qua, chỉ CLIP xử lý): {plan['unresolved']}")
            if plan.get("attributes"):
                st.write(f"**Thuộc tính (Region-CLIP, SigLIP2):** {plan.get('attributes')}")
            if "region_score" in results.columns:
                st.caption("Có cột `score_before_rerank`/`region_score` trong bảng log bước "
                           "region-CLIP — xem chi tiết ở log từng bước bên dưới.")

    st.caption(f"⏱ Tổng thời gian xử lý: {elapsed:.2f}s")
    with st.expander(f"📋 Log từng bước ({len(log.steps)} bước)", expanded=True):
        for i, s in enumerate(log.steps, 1):
            st.markdown(f"**{i}. {s['step']}** — `{s['elapsed_s']}s`  \n{s['detail']}")

    # 2026-08-15 (theo yeu cau nguoi dung: "cac he thong khac tra ve DU top_k") - search_dense()
    # gio LUON bu du top_k bang ket qua KHONG qua loc cung khi thieu (ke ca thieu 1 phan, khong
    # chi 0 hoan toan) - danh dau tung DONG qua cot "is_backfill". CANH BAO RO neu CO dong bu
    # them, ke ca ket qua tren van hien BINH THUONG (khong an di) - gap-honesty.
    n_backfill = int(results["is_backfill"].sum()) if not results.empty and "is_backfill" in results.columns else 0
    if n_backfill > 0:
        if n_backfill == len(results):
            st.warning("⚠ Bộ lọc cứng (nhãn/OCR/khung vẽ) không khớp kết quả nào — TOÀN BỘ danh "
                       "sách dưới đây là gợi ý CLIP thuần (BỎ QUA bộ lọc), không phải kết quả đã lọc đúng.")
        else:
            st.warning(f"⚠ Bộ lọc cứng chỉ khớp {len(results) - n_backfill}/{len(results)} kết quả — "
                       f"{n_backfill} kết quả còn lại (đánh dấu 🔓 bên dưới ảnh) là gợi ý CLIP thuần "
                       f"BỔ SUNG cho đủ số lượng, KHÔNG khớp bộ lọc.")

    if results.empty:
        st.warning("Không tìm thấy kết quả nào khớp bộ lọc.")
    elif is_qa:
        # 2026-08-15 (cutover): Q&A gio chay tren bo dense - anh doc THANG tu row["path"]
        # (dia local), khong qua Keyframes_*.zip cua BTC nua (xem submission_pipeline.answer_qa).
        st.caption(f"{len(results)} kết quả — chỉ {min(int(vqa_top_n), len(results))} ứng viên đầu được hỏi VQA thật, "
                   f"các ứng viên sau dùng lại câu trả lời của ứng viên đầu")
        cols = st.columns(4)
        for i, row in results.iterrows():
            col = cols[i % 4]
            with col:
                try:
                    st.image(row["path"], width="stretch")
                except Exception as e:
                    st.error(f"Không đọc được ảnh ({e})")
                vqa_tag = "🔎 VQA thật" if i < vqa_top_n else "↪ dùng lại rank 1"
                backfill_tag = " 🔓" if row.get("is_backfill") else ""
                st.markdown(
                    f"**{row['video_id']}**{backfill_tag} · frame `{row['frame_id']}` ({vqa_tag})  \n"
                    f"**Trả lời:** {row['answer']}"
                )
                _render_video_toggle(row["video_id"], int(row["frame_id"]), f"qa_{i}")
    elif is_temporal:
        st.caption(f"{len(results)} video khớp chuỗi {len(anchors)} anchor (đúng thứ tự thời gian)")
        for _, row in results.iterrows():
            st.markdown(f"**{row['video_id']}** · score={row['score']:.3f}")

            # Timeline truc quan (2026-08-11) - cham diem tung moc theo dung vi tri thoi gian
            # trong video, thay vi chi liet ke anh theo hang - de soat NHANH thu tu co dung
            # khong (dac biet khi cac moc gan nhau ve thoi gian, kho thay qua text).
            times = [float(row[f"anchor{i}_pts_time"]) for i in range(len(anchors))]
            t_max = max(times) * 1.05 if max(times) > 0 else 1.0
            timeline_html = (
                '<div style="position:relative;height:36px;background:#eee;border-radius:4px;margin:8px 0;">'
            )
            for i, t in enumerate(times):
                pct = 100 * t / t_max
                timeline_html += (
                    f'<div style="position:absolute;left:{pct:.1f}%;top:0;transform:translateX(-50%);'
                    f'text-align:center;">'
                    f'<div style="width:10px;height:10px;border-radius:50%;background:#d33;'
                    f'margin:0 auto;"></div>'
                    f'<div style="font-size:11px;white-space:nowrap;">#{i + 1} {t:.1f}s</div></div>'
                )
            timeline_html += "</div>"
            st.markdown(timeline_html, unsafe_allow_html=True)

            cols = st.columns(len(anchors))
            for i, anchor in enumerate(anchors):
                with cols[i]:
                    # 2026-08-15 (migrate sang bo dense): doc anh THANG tu row[f"anchor{i}_path"]
                    # (dia local), khong qua Keyframes_*.zip cua BTC nua (xem tiers/dense_temporal.py).
                    try:
                        st.image(row[f"anchor{i}_path"], width="stretch")
                    except Exception as e:
                        st.error(f"Không đọc được ảnh ({e})")
                    extra = []
                    if anchor.get("must_have_labels"):
                        extra.append(f"nhãn: {anchor['must_have_labels']}")
                    if anchor.get("ocr_text"):
                        extra.append(f"OCR: \"{anchor['ocr_text']}\"")
                    extra_str = f"  \n_{' · '.join(extra)}_" if extra else ""
                    st.caption(
                        f"{anchor['text']}{extra_str}  \n"
                        f"frame `{int(row[f'anchor{i}_frame_id'])}` · t={row[f'anchor{i}_pts_time']:.2f}s"
                    )
                    _render_video_toggle(
                        row["video_id"], int(row[f"anchor{i}_frame_id"]),
                        f"trake_{row['video_id']}_{i}_{int(row[f'anchor{i}_frame_id'])}",
                    )
            st.divider()
    else:
        # duong mac dinh (2026-08-15, cutover): search_dense() - cot path/frame_id (xem
        # tiers/dense_search.py), anh nam THANG tren dia local, khong qua Keyframes_*.zip.
        st.caption(f"{len(results)} kết quả — model: {dense_model}")
        cols = st.columns(4)
        for i, row in results.iterrows():
            col = cols[i % 4]
            with col:
                try:
                    st.image(row["path"], width="stretch")
                except Exception as e:
                    st.error(f"Không đọc được ảnh ({e})")
                # 🔓 = ket qua BU THEM (khong khop bo loc cung, xem is_backfill o dense_search.py)
                backfill_tag = " 🔓" if row.get("is_backfill") else ""
                st.markdown(
                    f"**{row['video_id']}**{backfill_tag} · frame `{row['frame_id']}` · score={row['score']:.3f}"
                )
                _render_video_toggle(row["video_id"], int(row["frame_id"]), f"kis_{i}")
