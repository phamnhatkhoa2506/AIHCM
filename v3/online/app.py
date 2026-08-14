"""Giao diện đơn giản: nhập query text -> hiển thị top-k frame khớp nhất (ảnh + info).

Chạy: streamlit run app.py
"""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/

import time

import streamlit as st
from streamlit_drawable_canvas import st_canvas

from keyframe_images import read_keyframe_bytes
from label_translate import resolve as resolve_label_vi
from label_translate import suggest as suggest_label_vi
from query_planner import planned_search
from search import search
from steplog import StepLog
from submission_pipeline import answer_qa
from tiers.dense_search import search_dense

st.set_page_config(page_title="AIC 2026 - Tier 1 Search", layout="wide")
st.title("AIC 2026 — Tier 1 Vector Search (CLIP + FAISS)")

with st.sidebar:
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
    st.caption("Lọc thô theo Object (gõ tiếng Việt, tự map sang nhãn OpenImages V4)")
    labels_raw = st.text_input("Nhãn bắt buộc (vd: người, xe hơi — phân cách dấu phẩy)")
    min_person = st.number_input("Số lượng 'người' tối thiểu (0 = không lọc)", min_value=0, value=0, step=1)

    include_open_vocab = st.checkbox(
        "Dùng cả nhãn open-vocab (Grounding DINO) khi lọc Object, không chỉ 514 nhãn gốc BTC",
        value=True,
        help="Tắt: chỉ khớp trên đúng 514 nhãn closed-set BTC cấp — dùng để so sánh/fallback "
        "về hành vi trước khi có Grounding DINO. Bật: query có thể khớp thêm nhãn mới phát "
        "hiện (vd 'lion dance costume') không nằm trong 514 nhãn gốc.",
    )
    use_suppression = st.checkbox(
        "Đè nhãn closed-set bị open-vocab (Grounding DINO) xác định lại",
        value=True,
        disabled=not include_open_vocab,
        help="Bật: nhãn cũ sai (vd 'Dog' gán cho đầu lân múa lân) bị loại khỏi kết quả khớp "
        "nếu có detection open-vocab cùng vị trí đè lên nó. Tắt để so sánh với hành vi cũ "
        "(không suppress gì cả) — dữ liệu gốc không đổi, chỉ đổi cách lọc lúc truy vấn. "
        "(Không áp dụng nếu đã tắt nhãn open-vocab ở trên.)",
    )

    top_k = st.slider("Số kết quả", min_value=5, max_value=50, value=12, step=1)

st.subheader("🔤 Chữ trên màn hình (OCR) — hard filter")
st.caption("Khớp chuỗi ký tự CHÍNH XÁC (không phải soft-boost) — xem "
           "share/tiers/tier1_filter.py::by_text")
ocr_text_raw = st.text_input("Nội dung chữ (không cần gõ dấu)", key="ocr_text_input")
st.caption("Khoanh vùng gần đúng (tuỳ chọn) — để trống nếu không quan tâm vị trí.")
# KHONG bo trong st.expander/tab: streamlit-drawable-canvas do chieu cao 1 LAN LUC MOUNT, neu
# component mount trong luc container dang AN (expander thu gon) no do duoc height=0 va KHONG
# BAO GIO do lai sau khi container hien ra - bug that phat hien 2026-08-10 (test Playwright:
# canvas chi hien dung 1/4 lan tai trang khi con nam trong expander).
_CANVAS_W, _CANVAS_H = 720, 405  # ti le 16:9, co dinh de chuan hoa toa do ve [0,1]
_canvas_result = st_canvas(
    fill_color="rgba(255,165,0,0.3)",
    stroke_width=2,
    stroke_color="red",
    background_color="#222",
    height=_CANVAS_H,
    width=_CANVAS_W,
    drawing_mode="rect",
    key="ocr_region_canvas",
)
ocr_region = None
if _canvas_result.json_data is not None:
    objs = _canvas_result.json_data.get("objects", [])
    if objs:
        o = objs[-1]  # chi lay hinh MOI NHAT ve duoc (1 vung duy nhat/lan tim)
        xmin = max(0.0, o["left"] / _CANVAS_W)
        ymin = max(0.0, o["top"] / _CANVAS_H)
        xmax = min(1.0, (o["left"] + o["width"] * o.get("scaleX", 1)) / _CANVAS_W)
        ymax = min(1.0, (o["top"] + o["height"] * o.get("scaleY", 1)) / _CANVAS_H)
        ocr_region = (ymin, xmin, ymax, xmax)
        st.caption(f"Vùng đã chọn: y=[{ymin:.2f},{ymax:.2f}] x=[{xmin:.2f},{xmax:.2f}]")
st.divider()

mode = st.radio(
    "Loại truy vấn",
    ["1 câu (Tầng 2)", "Chuỗi sự kiện theo thời gian (Tầng 3)", "Hỏi đáp (Q&A)",
     "Dense đa mô hình (SigLIP2/PE-Core/BEiT-3/RRF)"],
    horizontal=True,
)
is_temporal = mode == "Chuỗi sự kiện theo thời gian (Tầng 3)"
is_qa = mode == "Hỏi đáp (Q&A)"
# Dense (2026-08-14): tang tim kiem THU 2, doc lap voi Tang 1-3 - dung tren bo keyframe TU
# TRICH mat do cao hon (xem hoi thoai), CHI co CLIP-family embedding (khong co
# objects_index/OCR/ASR di kem cho bo nay) -> khong ap dung duoc filter Tang 1 o sidebar.
is_dense = mode == "Dense đa mô hình (SigLIP2/PE-Core/BEiT-3/RRF)"

use_planner = False

if is_temporal:
    query = ""
    st.caption("Mỗi ô = 1 khoảnh khắc, ĐÚNG thứ tự thời gian (vd: giậm nhảy / bay qua xà / tiếp đất)")
    # UI dong (2026-08-11) - thay the textarea tinh bang danh sach co the them/xoa/sap xep
    # tung moc rieng, dung st.session_state de giu trang thai giua cac lan rerun cua Streamlit.
    # moi moc gio la dict {"text","labels_raw","ocr_raw"} - khong con string tinh, de mang
    # theo filter RIENG cho tung khoanh khac (2026-08-11, xem tier3_temporal.py::_normalize_anchor)
    if "trake_anchors" not in st.session_state:
        st.session_state.trake_anchors = [
            {"text": "", "labels_raw": "", "ocr_raw": ""},
            {"text": "", "labels_raw": "", "ocr_raw": ""},
        ]

    for i in range(len(st.session_state.trake_anchors)):
        anc = st.session_state.trake_anchors[i]
        col_text, col_up, col_down, col_del = st.columns([10, 1, 1, 1])
        anc["text"] = col_text.text_input(
            f"Mốc {i + 1}", value=anc["text"], key=f"trake_anchor_{i}",
            label_visibility="collapsed", placeholder=f"Mốc {i + 1} (vd: giậm nhảy)",
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

        with st.expander(f"Lọc riêng cho mốc {i + 1} (tuỳ chọn)", expanded=False):
            anc["labels_raw"] = st.text_input(
                "Nhãn bắt buộc riêng cho mốc này (vd: người, xà)", value=anc["labels_raw"],
                key=f"trake_labels_{i}",
                help="Chỉ áp dụng cho khoảnh khắc này, KHÔNG áp dụng cho cả chuỗi.",
            )
            anc["ocr_raw"] = st.text_input(
                "Chữ trên màn hình riêng cho mốc này", value=anc["ocr_raw"], key=f"trake_ocr_{i}",
            )

    if st.button("+ Thêm mốc"):
        st.session_state.trake_anchors.append({"text": "", "labels_raw": "", "ocr_raw": ""})
        st.rerun()

    anchors_raw = ""  # khong con dung duong text-join cu - xem cho xay dung `anchors` ben duoi
    question = ""
elif is_qa:
    query = st.text_input("Mô tả sự kiện", placeholder="vd: một người đang nấu ăn với chảo và thìa")
    question = st.text_input("Câu hỏi", placeholder="vd: người trong ảnh đang cầm vật gì?")
    vqa_top_n = st.number_input(
        "Số ứng viên đầu gọi VQA thật (tốn phí/lần)", min_value=1, max_value=20, value=3, step=1
    )
    anchors_raw = ""
elif is_dense:
    query = st.text_input("Query", placeholder="vd: một diễn giả mặc áo đỏ phát biểu",
                           key="dense_query_input")
    dense_mode = st.radio(
        "Model", ["siglip", "pe_core", "beit3", "rrf"], horizontal=True, key="dense_model_radio",
        help="siglip/pe_core/beit3: dung DUNG 1 model. rrf: fusion Reciprocal Rank Fusion tren "
        "ca 3 (RRF_K=60) — thuong on dinh hon vi khong phu thuoc 1 model duy nhat.",
    )
    st.caption("⚠ Bo keyframe rieng (tu trich, mat do cao hon), KHONG dung chung filter Object/OCR "
               "o sidebar — chi vector search thuan.")
    anchors_raw = ""
    question = ""
else:
    query = st.text_input("Query", placeholder="vd: một diễn giả mặc áo đỏ phát biểu tại cuộc họp báo ngoài trời")
    anchors_raw = ""
    question = ""
    use_planner = st.checkbox(
        "Tự động phân rã câu ghép nhiều ràng buộc (LLM tách entity+số lượng cho Tầng 1, "
        "phần còn lại giao CLIP)",
        value=False,
        help='Vd: "diễn giả áo đỏ... phía sau có nhiều cây xanh" -> tự ép "cây">=2 qua Objects, '
        "không chỉ dựa vào CLIP đoán cả câu.",
    )
    use_region_clip = True
    if use_planner:
        use_region_clip = st.checkbox(
            "Dùng Region-CLIP rerank theo thuộc tính (màu/quần áo...)",
            value=True,
            help="Tắt để so sánh: chỉ dùng Tầng 1 (must_have_labels) + Tầng 2 (CLIP toàn khung "
            "hình), không crop+encode riêng từng vùng object theo thuộc tính.",
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

    must_have_labels = None
    terms = [t.strip() for t in labels_raw.split(",") if t.strip()]
    if terms:
        resolved, unresolved = [], []
        for t in terms:
            hits = resolve_label_vi(t)
            if hits:
                resolved.extend(hits)
            else:
                unresolved.append(t)
        if unresolved:
            st.warning(f"Không map được sang nhãn tiếng Anh (đã bỏ qua): {', '.join(unresolved)}")
            for t in unresolved:
                cands = suggest_label_vi(t, top_k=3)
                cand_str = " · ".join(f"{en} ({vi}, {sc:.2f})" for en, vi, sc in cands)
                st.caption(f"Gợi ý mờ cho '{t}' (chưa chắc đúng, tự gõ lại nếu khớp): {cand_str}")
        must_have_labels = resolved or None

    min_count = {"Person": int(min_person)} if min_person > 0 else None
    common_filters = dict(
        authors=authors or None,
        date_from=str(date_from) if date_from else None,
        date_to=str(date_to) if date_to else None,
        keywords_any=keywords_any,
        must_have_labels=must_have_labels,
        min_count=min_count,
        use_suppression=use_suppression,
        include_open_vocab=include_open_vocab,
        ocr_text=ocr_text_raw.strip() or None,
        ocr_region=ocr_region,
    )

    log = StepLog()
    t_start = time.perf_counter()
    plan = None
    with st.spinner("Đang tìm..." if not is_qa else "Đang tìm + hỏi VQA (có thể mất chút thời gian)..."):
        if is_dense:
            # KHONG dung StepLog/common_filters day du - tang doc lap - nhung TAI SU DUNG duoc
            # o OCR text_input o tren (ocr_text_raw) lam hard-filter, vi bo dense DA co OCR
            # rieng (2026-08-14, xem build_dense_ocr_index.py + dense_search.py::_ocr_candidates).
            results = search_dense(query, dense_mode, top_k=top_k, ocr_text=ocr_text_raw.strip() or None)
        elif is_qa:
            results = answer_qa(query, question, top_k=top_k, vqa_top_n=int(vqa_top_n), log=log, **common_filters)
        elif use_planner:
            results, plan = planned_search(
                query, top_k=top_k, log=log, use_region_clip=use_region_clip, **common_filters
            )
        else:
            results = search(query, top_k=top_k, anchors=anchors if is_temporal else None, log=log, **common_filters)
    elapsed = time.perf_counter() - t_start

    if plan is not None:
        with st.expander("Xem plan đã phân rã (LLM)", expanded=True):
            st.write(f"**Entity trích được:** {plan.get('entities')}")
            st.write(f"**Đã resolve sang nhãn:** {plan.get('resolved_must_have_labels')} "
                     f"(min_count={plan.get('resolved_min_count')})")
            if plan.get("unresolved"):
                st.warning(f"Không resolve được (bỏ qua, chỉ CLIP xử lý): {plan['unresolved']}")
            if plan.get("attributes"):
                st.write(f"**Thuộc tính (region-CLIP):** {plan.get('attributes')}")
            st.caption(f"CLIP text: {plan.get('clip_text')}")
            if "region_score" in results.columns:
                st.caption("Có cột `score_before_rerank`/`region_score` trong bảng log bước "
                           "region-CLIP — xem chi tiết ở log từng bước bên dưới.")

    st.caption(f"⏱ Tổng thời gian xử lý: {elapsed:.2f}s")
    with st.expander(f"📋 Log từng bước ({len(log.steps)} bước)", expanded=True):
        for i, s in enumerate(log.steps, 1):
            st.markdown(f"**{i}. {s['step']}** — `{s['elapsed_s']}s`  \n{s['detail']}")

    if results.empty:
        st.warning("Không tìm thấy kết quả nào khớp bộ lọc.")
    elif is_qa:
        st.caption(f"{len(results)} kết quả — chỉ {min(int(vqa_top_n), len(results))} ứng viên đầu được hỏi VQA thật, "
                   f"các ứng viên sau dùng lại câu trả lời của ứng viên đầu")
        cols = st.columns(4)
        for i, row in results.iterrows():
            col = cols[i % 4]
            with col:
                try:
                    img_bytes = read_keyframe_bytes(row["video_id"], int(row["local_idx"]))
                    st.image(img_bytes, width="stretch")
                except Exception as e:
                    st.error(f"Không đọc được ảnh ({e})")
                vqa_tag = "🔎 VQA thật" if i < vqa_top_n else "↪ dùng lại rank 1"
                st.markdown(
                    f"**{row['video_id']}** · frame `{row['frame_id']}` ({vqa_tag})  \n"
                    f"**Trả lời:** {row['answer']}"
                )
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
                    try:
                        img_bytes = read_keyframe_bytes(row["video_id"], int(row[f"anchor{i}_local_idx"]))
                        st.image(img_bytes, width="stretch")
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
                        f"frame `{int(row[f'anchor{i}_frame_idx'])}` · t={row[f'anchor{i}_pts_time']:.2f}s"
                    )
            st.divider()
    elif is_dense:
        st.caption(f"{len(results)} kết quả — model: {dense_mode} — bộ keyframe tự trích "
                   f"(không phải zip BTC gốc)")
        cols = st.columns(4)
        for i, row in results.iterrows():
            col = cols[i % 4]
            with col:
                try:
                    # KHAC read_keyframe_bytes() (doc tu Keyframes_*.zip cua BTC) - anh dense
                    # nam THANG tren dia local (xem manifest "path" trong dense_meta.parquet).
                    st.image(row["path"], width="stretch")
                except Exception as e:
                    st.error(f"Không đọc được ảnh ({e})")
                st.markdown(
                    f"**{row['video_id']}** · frame `{row['frame_id']}` · score={row['score']:.3f}"
                )
    else:
        st.caption(f"{len(results)} kết quả")
        cols = st.columns(4)
        for i, row in results.iterrows():
            col = cols[i % 4]
            with col:
                try:
                    img_bytes = read_keyframe_bytes(row["video_id"], int(row["local_idx"]))
                    st.image(img_bytes, width="stretch")
                except Exception as e:
                    st.error(f"Không đọc được ảnh ({e})")
                st.markdown(
                    f"**{row['video_id']}** · frame `{row['frame_idx']}`  \n"
                    f"t={row['pts_time']:.2f}s · score={row['score']:.3f}"
                )
