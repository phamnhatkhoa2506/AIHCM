"""Giao diện đơn giản: nhập query text -> hiển thị top-k frame khớp nhất (ảnh + info).

Chạy: streamlit run app.py
"""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/

import time
from datetime import datetime

import streamlit as st

from label_translate import resolve as resolve_label_vi
from app_flags import DISABLE_LLM_ENTITY_HARD_FILTER
from objects_canvas import objects_canvas
from query_distill import DEFAULT_DISTILL_MODEL, DISTILL_MODELS
from query_planner import extract_entities
from steplog import StepLog
from submission_pipeline import DEFAULT_VLM_OCR_MODEL, VLM_OCR_MODELS, answer_qa, vlm_read_text
from tiers import dense_temporal
from tiers.dense_search import DEFAULT_SCORE_ALGORITHM, SCORE_ALGORITHMS, apply_region_clip_rerank, search_dense
from tiers.tier1_filter import DEFAULT_OCR_ALGORITHM, OCR_MATCH_ALGORITHMS
from video_clip import get_shot_clip_bytes, get_fixed_window_clip_bytes
from vlm_corrections import save_approved_vlm_text

st.set_page_config(page_title="AIC 2026", layout="wide")

# Nhan hien thi (UI) -> gia tri that search_dense() can (DENSE_MODES trong dense_search.py) -
# radio doi sang nhan than thien ("All"/"SigLIP2"/...) nhung backend van nhan "rrf"/"siglip"/...
_DENSE_MODEL_DISPLAY_TO_VALUE = {"All": "rrf", "SigLIP2": "siglip", "PE-Core": "pe_core", "BEiT-3": "beit3"}


# 2026-08-16 (theo yeu cau nguoi dung: "thiet ke lai phan hien thi video ... giong popup nam 1
# goc, bam video khac thi REPLACE nhau chu khong hien dong dap") - CSS 1 LAN DUY NHAT, dinh vi
# container st.container(key="floating_video_player") thanh popup NOI CO DINH goc man hinh
# (Streamlit >=1.29 tu gan class "st-key-<key>" cho container co key= - dung class do de style,
# khong can component/JS rieng). CHI 1 popup DUY NHAT tren toan trang (khong phai 1 cai/dong ket
# qua nhu truoc) - bam "▶ Video" o BAT KY dong nao chi doi noi dung BEN TRONG popup do (REPLACE),
# khong tao them popup moi.
st.markdown(
    """<style>
    .st-key-floating_video_player {
        position: fixed; bottom: 20px; right: 20px; width: 360px; max-width: 90vw;
        background: #1c1c1e; border: 1px solid #3a3a3c; border-radius: 10px;
        padding: 10px 12px 4px 12px; z-index: 9999; box-shadow: 0 6px 24px rgba(0,0,0,0.5);
    }
    .st-key-floating_video_player video { border-radius: 6px; }
    /* 2026-08-18 (theo yeu cau nguoi dung: "vẫn còn bị lõm xuống không thẳng hàng" LAP LAI SAU
    2 lan thu CSS thu cong (position:absolute + aspect-ratio) - van khong het hoan toan) - BO
    HAN cach lam CSS/container-key thu cong o tren (fragile voi cau truc DOM long nhau cua
    Streamlit, kho kiem soat het truong hop) - CHUYEN sang st.popover() (xem
    _render_vlm_ocr_popover) - co che overlay GOC cua Streamlit, tu dam bao KHONG anh huong
    layout ben ngoai (khong can CSS position/aspect-ratio gi them cho VLM nua). */

    /* 2026-08-18 (theo yeu cau nguoi dung "cuộn ngang", sau 3 LAN THU CSS THU CONG DEU THAT BAI
    - "chưa được", "bị đè chồng lấn", "có cuộn ngang gì đâu") - DA TIM RA NGUYEN NHAN GOC bang
    cach doc THANG source bundle Streamlit da cai (static/js/index.*.js) thay vi doan tiep:
      1. st.columns() dat `flex: 1 1 calc(weight*100% - gap)` cho moi cot => cot LUON CO LAI
         theo % chieu rong cha, khong bao gio tran de sinh thanh cuon.
      2. Co san 1 MEDIA QUERY `@media (max-width: breakpoints.columns) { min-width: calc(100%
         - spacing.twoXL) }` => man hinh hep thi cot TU STACK DOC (dung hien tuong "thu nhỏ lại
         xong rồi như cũ" nguoi dung thay).
    => Ca 2 deu la CSS emotion cua Streamlit, ghi de bang selector doan mo ho la sai huong.
    GIAI PHAP DUNG: Streamlit 1.60 co API NATIVE `st.container(horizontal=True, width=<px>)` -
    render THANG horizontal block + chieu rong CO DINH theo pixel, KHONG dung st.columns() nua,
    nen KHONG dinh ca 2 co che tren. CSS duy nhat con lai o day chi la overflow-x cho container
    cha (selector "st-key-trake_row" RO RANG, 1 lan duy nhat, khong long nhau) de tran thi
    hien thanh cuon ngang thay vi bi cat.

    2026-08-18 (bo sung sau khi test that: van WRAP xuong hang chu chua cuon) - st.container(
    horizontal=True) cua Streamlit MAC DINH cho phep WRAP (flex-wrap mac dinh), va KHONG co
    tham so API nao tat wrap (signature chi co gap/width/height/alignment/autoscroll - da kiem
    tra) => phai ep `flex-wrap: nowrap` bang CSS. Dat cho CA 2 kha nang (class st-key gan
    THANG tren horizontal block, HOAC gan tren wrapper ngoai) de chac chan trung 1 trong 2 -
    van an toan vi "st-key-trake_row" chi ton tai DUNG 1 element tren trang, va selector con
    dung ">" (con TRUC TIEP) nen KHONG cham toi container ngang cua hang 3 nut (nam sau hon 1
    cap, ben trong tung the mốc). */
    div[class*="st-key-trake_row"] { overflow-x: auto; flex-wrap: nowrap !important; padding-bottom: 8px; }
    div[class*="st-key-trake_row"] > div[data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; }
    </style>""",
    unsafe_allow_html=True,
)


def _render_video_toggle(video_id: str, frame_id: int, widget_key: str, fixed_window: bool = False) -> None:
    """Nút "▶ Video" dưới mỗi keyframe — bấm CHỌN video này làm nội dung của popup nổi DUY
    NHẤT (xem _render_floating_video_player), KHÔNG tự phát tại chỗ nữa. Bấm nút khác sẽ
    THAY THẾ popup, không mở thêm cái mới.

    fixed_window=True (2026-08-18, chỉ TRAKE truyền vào — xem call site): phát popup cắt theo
    cửa sổ thời lượng CỐ ĐỊNH quanh frame_id thay vì theo đúng ranh giới shot (KIS/Q&A giữ
    nguyên mặc định False)."""
    if st.button("▶ Video", key=f"vidbtn_{widget_key}"):
        st.session_state.active_video = {
            "video_id": video_id, "frame_id": frame_id, "fixed_window": fixed_window,
        }


def _render_image_with_vlm_overlay(image_path: str, widget_key: str) -> None:
    """Vẽ ảnh keyframe — TÊN HÀM giữ nguyên để không phải sửa lại 3 nơi gọi (KIS/Q&A/TRAKE),
    nhưng từ 2026-08-18 KHÔNG còn tự vẽ overlay VLM ở đây nữa (đã chuyển hẳn sang st.popover(),
    xem _render_vlm_ocr_popover — Streamlit tự lo overlay, không cần container/CSS thủ công neo
    theo ảnh nữa, nên hàm này chỉ còn đúng việc vẽ ảnh)."""
    try:
        st.image(image_path, width="stretch")
    except Exception as e:
        st.error(f"Không đọc được ảnh ({e})")


def _render_vlm_ocr_verify(image_path: str, video_id: str, frame_id: int, widget_key: str) -> None:
    """Nút icon "🔍"/"👁" mở popover (2026-08-17, theo yêu cầu người dùng: "dùng 1 VLM nhỏ để
    đọc chữ thôi, bỏ qua box cho các trường hợp như thế này", sau khi phát hiện PaddleOCR bỏ
    sót/đọc sai 1 biển 3 dòng ở cả 4 frame trong shot — xem hội thoại).

    2026-08-18 (SỬA LẦN 3 sau 2 lần thử container/CSS thủ công đều làm lệch layout ở nhiều mức
    độ khác nhau — "Ý mình không phải như thế :))" rồi "Vẫn chưa được bạn ạ" rồi "Vẫn còn bị lõm
    xuống") — CHUYỂN HẲN sang st.popover(): cơ chế overlay GỐC của Streamlit, tự bảo đảm không
    ảnh hưởng layout xung quanh (không cần position/aspect-ratio/container-key thủ công nữa).

    on_change="rerun" + key -> BẮT BUỘC để dùng .open (mặc định "ignore" thì .open luôn None
    và code BÊN TRONG popover chạy MỌI LẦN rerun bất kể đóng/mở — sẽ phá vỡ tính "lazy, chỉ chạy
    khi bấm" đã yêu cầu từ đầu, xem help(st.popover)). Khi .open=True: nếu CHƯA cache thì gọi
    VLM NGAY BÊN TRONG popover (hiện "Đang xử lý..." rồi kết quả, đều nằm gọn trong popover đã
    mở sẵn — không cần rerun lại). Đã cache thì hiển thị lại ngay, không gọi lại API.

    2026-08-18 (theo yêu cầu người dùng: "thêm 1 cái... approved text để lưu lại vào hệ thống
    ... chỉ được chấp thuận khi có sự đồng ý của mình, có nút để approved chứ không để tự
    động") — thêm nút "✅ Duyệt" NGAY TRONG popover, dưới kết quả — CHỈ ghi vào
    vlm_corrections.VLM_APPROVED_OCR_PATH khi người dùng BẤM nút này, không có đường nào khác
    ghi tự động. Bấm lại (vd sau khi đổi model, đọc lại) vẫn ghi thêm 1 dòng mới — giữ lịch sử,
    không tự động coi 1 lần duyệt là áp dụng mãi mãi cho frame đó."""
    cache_key = f"vlm_ocr_{widget_key}"
    has_result = cache_key in st.session_state
    icon = "👁" if has_result else "🔍"
    help_text = "Xem lại chữ VLM đã đọc" if has_result else "Xác minh chữ trong ảnh bằng VLM"

    pop = st.popover(icon, help=help_text, on_change="rerun", key=f"vlmpop_{widget_key}")
    if pop.open:
        with pop:
            if cache_key not in st.session_state:
                model = st.session_state.get("vlm_ocr_model", DEFAULT_VLM_OCR_MODEL)
                with st.spinner(f"Đang hỏi {VLM_OCR_MODELS.get(model, model)} đọc chữ trong ảnh..."):
                    try:
                        text = vlm_read_text(image_path, model)
                    except Exception as e:
                        text = f"LỖI: {type(e).__name__} {str(e)[:150]}"
                    st.session_state[cache_key] = (text, model)
            text, used_model = st.session_state[cache_key]
            model_tag = VLM_OCR_MODELS.get(used_model, used_model)
            st.markdown(f"**VLM đọc chữ** _({model_tag})_")
            st.write(text or "_(không thấy chữ)_")

            approved_key = f"vlm_approved_{widget_key}"
            already_approved = st.session_state.get(approved_key)
            if text and not text.startswith("LỖI:"):
                if st.button("✅ Duyệt, lưu để cải thiện dữ liệu", key=f"vlmapprove_{widget_key}"):
                    save_approved_vlm_text(video_id, frame_id, text, used_model, image_path)
                    st.session_state[approved_key] = datetime.now().strftime("%H:%M:%S")
                if st.session_state.get(approved_key):
                    st.caption(f"✅ Đã duyệt lúc {st.session_state[approved_key]} — đã lưu vào hệ thống")
            elif already_approved:
                st.caption(f"✅ Đã duyệt lúc {already_approved} — đã lưu vào hệ thống")


def _render_floating_video_player() -> None:
    """Popup nổi DUY NHẤT, gọi 1 LẦN cho cả trang (vị trí gọi trong script không quan trọng vì
    CSS `position: fixed` tự đưa ra góc màn hình, không theo luồng layout bình thường). Đọc
    st.session_state.active_video (do _render_video_toggle set) — None thì không hiện gì cả."""
    active = st.session_state.get("active_video")
    if not active:
        return
    with st.container(key="floating_video_player"):
        col_title, col_close = st.columns([5, 1])
        col_title.markdown(f"**{active['video_id']}** · frame `{active['frame_id']}`")
        if col_close.button("✕", key="close_floating_video", help="Đóng video"):
            st.session_state.active_video = None
            st.rerun()
        else:
            # 2026-08-16 (theo yeu cau nguoi dung: "index/keyframe_meta_all.jsonl chinh la
            # gioi han shot ... chu khong lay co dinh nhu dang lam") - cat DUNG doan SHOT chua
            # frame nay (dense_meta.parquet, phu 873/873 video) thay vi pad co dinh +-3s -
            # xem video_clip.get_shot_clip_bytes.
            # 2026-08-18 (theo yeu cau nguoi dung: "TRAKE ... chuyen tu lay tu shot sang thoi
            # luong co dinh theo cua so 7 8s gi do", "khong lien quan toi KIS va QA") - TRAKE
            # (active["fixed_window"]=True, xem _render_video_toggle call site) dung cua so co
            # dinh TRAKE_FIXED_WINDOW_SECONDS; KIS/Q&A khong doi, van cat theo dung shot.
            if active.get("fixed_window"):
                spinner_msg = "Đang cắt đoạn video (cửa sổ cố định)..."
                clip_fn = get_fixed_window_clip_bytes
            else:
                spinner_msg = "Đang cắt đoạn video theo đúng shot..."
                clip_fn = get_shot_clip_bytes
            with st.spinner(spinner_msg):
                try:
                    st.video(clip_fn(active["video_id"], active["frame_id"]))
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
        help="All: fusion Reciprocal Rank Fusion trên SigLIP2 + PE-Core (2026-08-18: BỎ BEiT-3 "
        "khỏi fusion theo yêu cầu người dùng sau nhiều lần test — BEiT-3 không đủ mạnh so với "
        "2 model kia, RRF cộng theo rank nên 1 model yếu vẫn có 'phiếu' ngang hàng, có thể kéo "
        "xuống kết quả đúng của model mạnh). SigLIP2/PE-Core/BEiT-3: dùng đúng 1 model riêng "
        "(SigLIP2 thắng benchmark rõ rệt; BEiT-3 vẫn giữ làm lựa chọn riêng, chỉ không còn "
        "trong fusion).",
    )
    dense_model = _DENSE_MODEL_DISPLAY_TO_VALUE[dense_model_display]

    # 2026-08-17 (theo yeu cau nguoi dung: "trên giao diện sẽ để các nút để người dùng chọn
    # thuật toán tự chọn đó bạn, mặc định là cái dễ nhất") - chon thuat toan khop chu OCR
    # (dung chung cho ca hard-filter ocr_text/khung ve LAN soft-boost do gon khop, xem
    # tier1_filter.OCR_MATCH_ALGORITHMS + dense_search.search_dense(ocr_algorithm=...)).
    _ocr_algo_keys = list(OCR_MATCH_ALGORITHMS)
    ocr_algorithm = st.radio(
        "Thuật toán khớp OCR",
        _ocr_algo_keys,
        format_func=lambda k: OCR_MATCH_ALGORITHMS[k][0],
        index=_ocr_algo_keys.index(DEFAULT_OCR_ALGORITHM),
        horizontal=True,
        help="Đơn giản: khớp từng từ chính xác (nhanh nhất, mặc định). RapidFuzz: chịu được "
        "lỗi ký tự OCR nhưng không giữ chặt thứ tự — chậm hơn vì phải quét toàn bộ frame có "
        "OCR (không dùng được bộ lọc sơ bộ nhanh). Sequence Alignment: quy hoạch động, có cơ "
        "sở lý thuyết rõ ràng hơn, chịu lỗi ký tự nhẹ — chậm nhất trong 3 lựa chọn.",
    )

    # 2026-08-18 (theo yeu cau nguoi dung: "mình muốn thử với nhiều LLM hơn... model từ bé đến
    # lớn" - sau khi phat hien bug that: LLaMA 3.1 8B XOA MAT "nhiều người" khoi cau chung cat,
    # LLaMA 70B GIU DUNG khi test that) - chon model LLM cho buoc chung cat query (dich + rut
    # gon truoc khi encode) - xem query_distill.py::DISTILL_MODELS.
    _distill_model_keys = list(DISTILL_MODELS)
    distill_model = st.selectbox(
        "Model chưng cất query (dịch + rút gọn)",
        _distill_model_keys,
        format_func=lambda k: DISTILL_MODELS[k],
        index=_distill_model_keys.index(DEFAULT_DISTILL_MODEL),
        help="Model LLM dịch + rút gọn câu query tiếng Việt sang tiếng Anh trước khi đưa vào "
        "CLIP encode. Model 8B (mặc định) đôi khi BỎ SÓT chi tiết (test thật: xoá mất \"nhiều "
        "người\" khỏi câu, chỉ giữ lại vật thể) — model 70B lớn hơn giữ đúng chi tiết hơn "
        "nhưng chậm hơn 1 chút.",
    )

    # 2026-08-17 (theo yeu cau nguoi dung: "có thuật toán nào đáng tin cậy hơn cosine similarity
    # không" -> chon huong 1: sigmoid hieu chinh THEO DUNG cong thuc SigLIP2 duoc train (sigmoid
    # loss, khong phai contrastive softmax nhu CLIP) - xem dense_search.py::_apply_score_algorithm.
    # KHONG xoa cosine (van la mac dinh), CHI THEM lua chon - giong het pattern OCR_MATCH_ALGORITHMS.
    _score_algo_keys = list(SCORE_ALGORITHMS)
    score_algorithm = st.radio(
        "Thuật toán tính điểm xếp hạng", _score_algo_keys,
        format_func=lambda k: SCORE_ALGORITHMS[k],
        index=_score_algo_keys.index(DEFAULT_SCORE_ALGORITHM),
        horizontal=True,
        help="Cosine: điểm thô từ FAISS (mặc định, đang dùng từ trước). Sigmoid hiệu chỉnh: "
        "dùng ĐÚNG công thức SigLIP2 được train (sigmoid loss, không phải contrastive softmax "
        "như CLIP) — đưa điểm về khoảng [0,1] có ý nghĩa xác suất, giúp các soft-boost (vị trí "
        "khung vẽ/độ gọn OCR/ASR) cộng vào công bằng hơn. CHỈ có hiệu lực thật với model SigLIP2 "
        "(và fusion RRF, vì RRF luôn dùng SigLIP2) — PE-Core/BEiT-3 không có công thức hiệu "
        "chỉnh riêng nên vẫn giữ nguyên cosine dù chọn Sigmoid.",
    )

    # 2026-08-18 (theo yeu cau nguoi dung, sau khi TU CHUNG MINH bang so lieu that voi query
    # "khu chợ...phụ nữ đội nón lá..." - xem hoi thoai): cau ghep NHIEU mo ta encode chung 1
    # vector khien 1 menh de LAN AT menh de kia (GT rot tu hang #0 rieng menh de xuong #159 khi
    # ghep). Bat co nay -> tach cau theo dau cham (./!/?), encode RIENG tung menh de, gop diem
    # bang MAX (khong phai trung binh - xem dense_search.py::_rank_multi_clause) - GT len lai
    # #1 khi test that. Mac dinh TAT (khong doi hanh vi cu) vi query 1-cau van encode y het
    # truoc, CHI anh huong query CO nhieu cau/menh de that su.
    multi_clause = st.checkbox(
        "Tách câu thành từng mệnh đề, xếp hạng riêng rồi lấy điểm cao nhất",
        value=False,
        help="Dành cho câu MÔ TẢ GHÉP NHIỀU Ý (vd 'Cảnh khu chợ... Giữa khung hình một phụ nữ "
        "đội nón lá...') — mặc định CLIP encode CẢ CÂU thành 1 vector, dễ bị 1 mệnh đề (thường "
        "là chủ thể người) LẤN ÁT mệnh đề còn lại (bối cảnh/vật thể) trong biểu diễn chung. Bật "
        "lên: tách câu theo dấu chấm câu (./!/?), CHƯNG CẤT + XẾP HẠNG RIÊNG từng mệnh đề, rồi "
        "lấy ĐIỂM CAO NHẤT trong các mệnh đề cho mỗi frame — không cần khớp đều tay cả câu, chỉ "
        "cần khớp MẠNH ít nhất 1 khía cạnh. Câu chỉ có 1 mệnh đề (không có dấu chấm giữa câu) "
        "thì không đổi gì. Tốn thời gian hơn theo đúng SỐ MỆNH ĐỀ (mỗi mệnh đề chạy full "
        "pipeline riêng).",
    )

    # 2026-08-17 (theo yeu cau nguoi dung: "thêm 1 số mô hình to hơn xem thử") - chon model VLM
    # cho nut "🔍 VLM đọc chữ" (xem _render_vlm_ocr_verify + submission_pipeline.VLM_OCR_MODELS
    # cho ket qua test truc tiep tren case that dan toi thu tu/lua chon nay). Luu vao
    # session_state["vlm_ocr_model"] (khong phai bien local) vi doc lai TRONG _render_vlm_ocr_
    # verify - ham do dinh nghia o DAU file, TRUOC KHI sidebar nay chay lan dau trong 1 rerun.
    _vlm_model_keys = list(VLM_OCR_MODELS)
    st.selectbox(
        "Model VLM đọc chữ (xác minh OCR)",
        _vlm_model_keys,
        format_func=lambda k: VLM_OCR_MODELS[k],
        index=_vlm_model_keys.index(DEFAULT_VLM_OCR_MODEL),
        key="vlm_ocr_model",
        help="Dùng cho nút '🔍 VLM đọc chữ' dưới mỗi kết quả — CHỈ chạy khi bấm tay (không tự "
        "động theo mỗi query). Nemotron Nano VL 8B đọc đúng nhất khi test trên case PaddleOCR "
        "đọc sai/bỏ sót (xem hội thoại) — cả 3 model đều KHÔNG đọc được chữ quá nhỏ (giới hạn "
        "chung, không phải do chọn sai model).",
    )

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


def _render_filter_canvas(key_suffix: str, after_list=None, full_width: bool = False) -> list[dict]:
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
        chọn màu/loại trước khi vẽ nên không có khả năng nhận nhầm loại khung.

    full_width (2026-08-18, theo yêu cầu người dùng "muốn biểu diễn các mốc theo chiều ngang...
    align các input, lọc chữ với chiều ngang bằng với khung vẽ") - TRAKE giờ gọi VỚI
    full_width=True: mỗi mốc đã nằm trong 1 cột NGANG hẹp riêng (xem vòng lặp trake_anchors),
    không cần chia thêm 50/50 canvas/danh-sách-tóm-tắt nữa (danh sách đó cũng đã bỏ, xem
    2026-08-18 khác) - canvas dùng THẲNG full chiều rộng cột (tự giới hạn 480px qua CSS riêng
    của component, xem _CANVAS_W) - khớp đúng max-width CSS của thẻ mốc (.st-key-trake_card_)."""
    if full_width:
        raw_boxes = objects_canvas(key=f"objects_canvas_{key_suffix}", width=_CANVAS_W, height=_CANVAS_H)
        col_list = None
    else:
        # 2026-08-15 (theo yeu cau nguoi dung "ben phai bi du nhieu qua") - component canvas co
        # chieu rong CO DINH (_CANVAS_W=480px, khong gian ra theo container) nen neu de full-
        # width se thua rat nhieu khoang trong ben phai trong the (container) rong. Dat canvas
        # vao cot trai HEP VUA DU, danh sach tom tat khung ve sang cot phai TAN DUNG khoang do.
        col_canvas, col_list = st.columns([1, 1])
        with col_canvas:
            raw_boxes = objects_canvas(key=f"objects_canvas_{key_suffix}", width=_CANVAS_W, height=_CANVAS_H)

    # 2026-08-18 (theo yeu cau nguoi dung: "muốn thiết kế gọn hơn, bỏ phần [danh sách tóm tắt
    # khung 'N khung đã vẽ:'/'Khung #i: ...'] cho mọi mode câu hỏi") - BO HAN doan liet ke text
    # nay (trung lap thong tin voi chinh canvas - nhan/chu da BAKE TRUC TIEP len khung ve trong
    # objects_canvas, xem docstring dau ham) - CHI GIU logic build spatial_boxes (van can cho
    # search), khong hien thi lai duoi dang danh sach rieng nua.
    spatial_boxes: list[dict] = []
    for b in raw_boxes:
        xmin = max(0.0, min(b["x0"], b["x1"]) / _CANVAS_W)
        ymin = max(0.0, min(b["y0"], b["y1"]) / _CANVAS_H)
        xmax = min(1.0, max(b["x0"], b["x1"]) / _CANVAS_W)
        ymax = min(1.0, max(b["y0"], b["y1"]) / _CANVAS_H)
        region = (ymin, xmin, ymax, xmax)
        if b["kind"] == "ocr":
            text = b["text"].strip()
            if not text:
                continue
            spatial_boxes.append({"type": "ocr", "text": text, "region": region})
        else:
            label = b["label"].strip()
            if not label:
                continue
            # dropdown chi cho chon dung 514 nhan co san -> resolve LUON thanh cong, khong can
            # canh bao "khong map duoc" nhu duong text tu do truoc day.
            hits = resolve_label_vi(label)
            min_count = int(b.get("minCount") or 1)
            spatial_boxes.append({"type": "object", "labels": hits, "region": region, "min_count": min_count})

    if after_list is not None:
        if col_list is not None:
            with col_list:
                after_list()
        else:
            after_list()
    return spatial_boxes


_REGION_CLIP_CHECKBOX_HELP = (
    "Ép ưu tiên frame khớp thuộc tính LLM trích được (vd 'áo dài màu tím'), không chỉ dựa vào "
    "CLIP đoán cả câu. Chạy qua server Modal riêng (aic2026-region-rerank, luôn giữ ấm), ~2s/lần. "
    "Tự động BỎ QUA (không xáo kết quả) nếu SigLIP2 không đủ phân biệt thuộc tính đó trên các "
    "ứng viên (dynamic range điểm quá thấp — xem log chi tiết)."
)


def _render_kis_query_inline() -> None:
    """2026-08-17 (theo yeu cau nguoi dung: "sửa input query nằm ngang hàng với phần box cho
    đỡ tốn chỗ", sau do "cho cái này cùng cột với input query luôn bạn" cho checkbox Region-
    CLIP) - Query + checkbox rerank cua KIS render NGAY DUOI danh sach khung ve (trong col_list
    cua _render_filter_canvas, xem after_list) thay vi 1 khoi RIENG, FULL-WIDTH ben duoi ca
    canvas (ton them nhieu hang cao du canvas van con). Dung key co dinh + doc lai qua
    session_state (thay vi return value) vi ham nay duoc TRUYEN LAM CALLBACK vao trong
    _render_filter_canvas, KHONG goi truc tiep tai noi can gia tri."""
    st.text_input(
        "Query", key="kis_query_input",
        placeholder="VD: Một diễn giả mặc áo đỏ phát biểu tại cuộc họp báo ngoài trời",
    )
    st.checkbox(
        "Region-CLIP rerank theo thuộc tính (màu/quần áo..., SigLIP2)",
        value=True, key="kis_region_clip_rerank", help=_REGION_CLIP_CHECKBOX_HELP,
    )


def _render_qa_query_inline() -> None:
    """2026-08-17 (theo yeu cau nguoi dung "sửa luôn cả Q&A luôn bạn" - cung tinh than voi
    _render_kis_query_inline o tren) - 2 o "Mô tả sự kiện"/"Câu hỏi" + checkbox rerank cua Q&A
    CUNG render ngang hang voi canvas (trong col_list), khong con 1 khoi rieng FULL-WIDTH ben
    duoi.

    2026-08-18 (theo yeu cau nguoi dung: "Cho cái số ứng viên cùng cột với input query luôn
    bạn") - vqa_top_n GIO CUNG render o day (truoc rieng ben duoi, full-width) - doc lai qua
    session_state["qa_vqa_top_n"] o nhanh is_qa (giong pattern cac o khac trong callback nay)."""
    st.text_input("Mô tả sự kiện", key="qa_query_input", placeholder="vd: một người đang nấu ăn với chảo và thìa")
    st.text_input("Câu hỏi", key="qa_question_input", placeholder="vd: người trong ảnh đang cầm vật gì?")
    st.number_input(
        "Số ứng viên đầu gọi VQA thật (tốn phí/lần)", min_value=1, max_value=20, value=3, step=1,
        key="qa_vqa_top_n",
    )
    st.checkbox(
        "Region-CLIP rerank theo thuộc tính (màu/quần áo..., SigLIP2)",
        value=True, key="qa_region_clip_rerank", help=_REGION_CLIP_CHECKBOX_HELP,
    )


if not is_temporal:
    _inline_query_fn = _render_qa_query_inline if is_qa else _render_kis_query_inline
    spatial_boxes = _render_filter_canvas("global", after_list=_inline_query_fn)
else:
    spatial_boxes = None  # TRAKE: khong con canvas chung - moi moc co canvas RIENG (xem duoi)
st.divider()

if is_temporal:
    query = ""
    # UI dong (2026-08-11) - thay the textarea tinh bang danh sach co the them/xoa/sap xep
    # tung moc rieng, dung st.session_state de giu trang thai giua cac lan rerun cua Streamlit.
    # moi moc gio la dict {"text","spatial_boxes"} - khong con string tinh, de mang theo filter
    # RIENG cho tung khoanh khac (2026-08-11, xem tier3_temporal.py::_normalize_anchor).
    # 2026-08-18: bo "labels_raw"/"ocr_raw" (o go tu do trung + kem an toan hon canvas Object/
    # OCR - xem docstring _render_filter_canvas o vong lap ben duoi).
    if "trake_anchors" not in st.session_state:
        st.session_state.trake_anchors = [
            {"text": "", "spatial_boxes": []},
            {"text": "", "spatial_boxes": []},
        ]

    # 2026-08-18 (theo yeu cau nguoi dung: "muốn biểu diễn các mốc theo chiều ngang... Mốc 1 ->
    # Mốc 2 -> ... theo chiều ngang", sau do "Có cuộn ngang gì đâu bạn... thấy thu nhỏ lại xong
    # rồi như cũ" khi thu wrap-xuong-hang) - LAN THU 3: BO HAN st.columns() cho hang mốc (nguon
    # goc phai doan cau truc stColumn/stHorizontalBlock, da sai 2 lan CSS truoc) - moi mốc gio
    # la 1 st.container() RIENG goi LIEN TIEP (khong qua columns) BEN TRONG 1 container cha DUY
    # NHAT (key="trake_row") - CSS dau file chi can ep DUNG 1 CHO (khoi doc mac dinh cua
    # container cha chuyen tu xep DOC sang NGANG + cho cuon), it tang doan hon han 2 lan truoc.
    n_anchors = len(st.session_state.trake_anchors)
    with st.container(key="trake_row", horizontal=True, gap="small"):
        for i in range(n_anchors):
            anc = st.session_state.trake_anchors[i]
            # 2026-08-15 (theo yeu cau nguoi dung "nhieu o roi rac qua, dinh kem input/nhan/khung
            # ve lai voi nhau") - BOC CA MOC (text mo ta + nut sap xep + bo loc rieng + canvas vao
            # 1 st.container(border=True) DUY NHAT -> doc thanh 1 THE lien khoi thay vi cac dong
            # thanh phan roi rac tach biet nhu truoc. Canvas van nam TRUC TIEP trong container nay
            # (KHONG long trong expander) de khong dinh bug mount height=0 (xem docstring
            # _render_filter_canvas) - container thuong (khong collapse duoc) khong dinh bug do.
            # width=_CANVAS_W (480px, NATIVE - khong phai CSS): moi the mốc CO DINH dung bang
            # be rong canvas ben trong => input/expander/canvas TU thang hang, va tong be rong
            # cac the co the TRAN ra ngoai container cha (khong bi ep co lai nhu st.columns),
            # nen thanh cuon ngang moi thuc su xuat hien - xem CSS "st-key-trake_row" dau file.
            with st.container(border=True, key=f"trake_card_{i}", width=_CANVAS_W):
                # 2026-08-18 (theo yeu cau nguoi dung: "xóa cái hàng này [hàng nút ↑/↓/🗑],
                # Chuyển phần xóa mốc thành dấu x ở trên cho dễ thấy, không cần ký hiệu thùng
                # rác") - BO HAN hang nut rieng (mat luon chuc nang sap xep ↑/↓ theo yeu cau
                # "xóa cái hàng này"), CHI GIU nut xoá, chuyen thanh dau "✕" nho dat NGAY CANH
                # tieu de "Mốc N" (de thay hon, quen thuoc kieu nut dong cua so/tab).
                col_title, col_close = st.columns([6, 1])
                col_title.markdown(f"**Mốc {i + 1}**")
                if col_close.button("✕", key=f"trake_del_{i}", disabled=(len(st.session_state.trake_anchors) <= 2), help="Xoá mốc này"):
                    st.session_state.trake_anchors.pop(i)
                    st.rerun()

                anc["text"] = st.text_input(
                    f"Mốc {i + 1}", value=anc["text"], key=f"trake_anchor_{i}",
                    label_visibility="collapsed", placeholder=f"Mô tả mốc {i + 1} (vd: giậm nhảy)",
                )

                # 2026-08-18 (theo yeu cau nguoi dung: "Trong các phần này có chỗ nào dư không
                # bạn (như chỗ màn hình)?") - DA XOA expander "Lọc chữ/nhãn phụ" (2 o gõ tự do
                # "Nhãn bắt buộc riêng"/"Chữ trên màn hình riêng") - TRUNG CHUC NANG voi canvas
                # Object/OCR ngay duoi day VA KEM AN TOAN HON: labels_raw go tu do phai qua
                # resolve_label_vi() (co the KHONG resolve duoc, dung lop bug "rổ xoài xanh" da
                # gap ca buoi hom nay), trong khi Object box duoi canvas la DROPDOWN chon thang
                # tu 514 nhan hop le (khong bao gio fail); ocr_raw trung voi OCR box nhung MAT
                # luon toa do (khong co soft-boost vi tri). Day la UI CU tu truoc khi doi sang
                # canvas 2 che do ro rang (2026-08-15) - bi bo sot, chua don theo thiet ke moi.
                anc["spatial_boxes"] = _render_filter_canvas(f"trake_{i}", full_width=True)

    if st.button("+ Thêm mốc"):
        st.session_state.trake_anchors.append({"text": "", "spatial_boxes": []})
        st.rerun()

    anchors_raw = ""  # khong con dung duong text-join cu - xem cho xay dung `anchors` ben duoi
    question = ""
elif is_qa:
    # 2026-08-17: 2 o nay rendered NGANG HANG voi danh sach khung ve (xem _render_qa_query_
    # inline, goi tu ben trong _render_filter_canvas o tren) - CHI doc lai qua session_state.
    query = st.session_state.get("qa_query_input", "")
    question = st.session_state.get("qa_question_input", "")
    # 2026-08-18: o nay CUNG rendered NGANG HANG voi canvas (xem _render_qa_query_inline) - CHI
    # doc lai qua session_state, KHONG goi st.number_input() lan 2.
    vqa_top_n = st.session_state.get("qa_vqa_top_n", 3)
    # 2026-08-17: checkbox nay rendered NGANG HANG voi canvas (xem _render_qa_query_inline) -
    # CHI doc lai qua session_state, KHONG goi st.checkbox() lan 2.
    use_region_clip_rerank = st.session_state.get("qa_region_clip_rerank", True)
    anchors_raw = ""
else:
    # 2026-08-17: o + checkbox rendered NGANG HANG voi danh sach khung ve (xem
    # _render_kis_query_inline, goi tu ben trong _render_filter_canvas o tren) - CHI doc lai
    # gia tri qua session_state o day, KHONG goi widget lan 2 (se tao trung key/nhan doi).
    query = st.session_state.get("kis_query_input", "")
    anchors_raw = ""
    question = ""
    use_region_clip_rerank = st.session_state.get("kis_region_clip_rerank", True)

run = st.button("Tìm kiếm", type="primary")

if is_temporal:
    # xay dung anchors dang dict THAT (khong qua text-join/split nua) - tu session_state,
    # bo qua moc rong (text trong) giong het logic cu (a.strip() if a.strip()).
    anchors = []
    for anc in st.session_state.trake_anchors:
        text = anc["text"].strip()
        if not text:
            continue
        entry: dict = {"text": text}
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
    # spatial_boxes = None khi TRAKE (moi moc co canvas RIENG, khong con canvas chung o day -
    # xem _render_filter_canvas/vong lap trake_anchors) - phong than voi (spatial_boxes or []).
    _obj_boxes = [b for b in (spatial_boxes or []) if b["type"] == "object"]
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
                                 ocr_algorithm=ocr_algorithm, score_algorithm=score_algorithm,
                                 multi_clause=multi_clause, distill_model=distill_model,
                                 log=log, **common_filters)
        elif is_temporal:
            # 2026-08-15: TRAKE gio co 1 canvas RIENG cho MOI moc (xem vong lap trake_anchors o
            # tren) - spatial_boxes o day la None (khong con canvas CHUNG cho ca chuoi), tung
            # anchor dict trong `anchors` da tu mang theo "spatial_boxes" cua rieng no (xem
            # dense_temporal.py::_normalize_anchor/search - merge per-anchor, khong con global).
            results = dense_temporal.search(
                anchors, top_k=top_k, dense_model=dense_model,
                spatial_boxes=spatial_boxes, spatial_op=spatial_op, ocr_algorithm=ocr_algorithm,
                score_algorithm=score_algorithm, distill_model=distill_model, log=log,
            )
        else:
            # 2026-08-15 (theo yeu cau nguoi dung: bo checkbox, LUON BAT LLM tach entity) -
            # extract_entities() (query_planner.py) LUON chay truoc, gop must_have_labels/
            # min_count trich duoc VOI cai tu khung Object tren canvas (UNION nhan, MAX
            # min_count) - roi giao het cho search_dense (bo dense, KHONG qua planned_search/
            # search() BTC cu nua).
            plan = extract_entities(query, log=log)
            # 2026-08-17 (GAC LAI de TEST - xem share/app_flags.py cho ly do day du: LLM
            # plan_query() bia them entity khong co that, resolve nham thanh hard-filter SAI,
            # bop hep sai corpus). Dung CHUNG 1 co voi submission_pipeline.answer_qa() qua
            # app_flags.py (khong tach rieng 2 bien, tranh KIS/Q&A lech hanh vi nhau).
            if DISABLE_LLM_ENTITY_HARD_FILTER:
                merged_must_have = must_have_labels
                merged_min_count = min_count
            else:
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
                audio_mentions=plan.get("audio_mentions") or None, ocr_algorithm=ocr_algorithm,
                score_algorithm=score_algorithm, multi_clause=multi_clause,
                distill_model=distill_model, log=log,
            )
            if attributes and not results.empty:
                results = apply_region_clip_rerank(results, attributes, top_k, log=log)
            elif log and plan.get("attributes") and not use_region_clip_rerank:
                log.add("Region-CLIP", "TẮT (checkbox người dùng) — bỏ qua bước rerank theo thuộc tính", 0.0)
    elapsed = time.perf_counter() - t_start

    # 2026-08-16 (BUG THAT nguoi dung phat hien: "bấm nút Video thì reset lại màn hình ban
    # đầu") - bat ky nut nao khac (vd "▶ Video") cung kich hoat Streamlit RERUN TOAN BO script,
    # luc do `run` (nut "Tìm kiếm") lai la False -> khoi "if run and ready" nay bi BO QUA HOAN
    # TOAN, `results` khong duoc tinh lai -> trang nhu bi reset. Fix: luu SNAPSHOT ket qua vao
    # st.session_state NGAY SAU KHI tinh xong - phan RENDER ben duoi doc tu day (khong phu
    # thuoc `run`), nen cac lan rerun sau (do bam nut Video/Hoan tac/...) van giu nguyen ket
    # qua cu tren man hinh.
    st.session_state.last_search = {
        "results": results, "elapsed": elapsed, "plan": plan, "log_steps": log.steps,
        "is_qa": is_qa, "is_temporal": is_temporal, "anchors": anchors,
        "vqa_top_n": vqa_top_n if is_qa else None, "dense_model": dense_model,
    }

# 2026-08-16: render TU SNAPSHOT (khong phu thuoc `run`) - xem giai thich o tren. Dung tien to
# `_r_` cho cac bien doc lai tu snapshot de KHONG lam nham voi is_qa/is_temporal/dense_model...
# cua sidebar HIEN TAI (nguoi dung co the da doi mode sau khi search xong, snapshot van phai
# giu dung boi canh LUC search do).
if "last_search" in st.session_state:
    _snap = st.session_state.last_search
    results = _snap["results"]
    elapsed = _snap["elapsed"]
    plan = _snap["plan"]
    _r_log_steps = _snap["log_steps"]
    _r_is_qa = _snap["is_qa"]
    _r_is_temporal = _snap["is_temporal"]
    _r_anchors = _snap["anchors"]
    _r_vqa_top_n = _snap["vqa_top_n"]
    _r_dense_model = _snap["dense_model"]

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
    # 2026-08-15 (theo yeu cau nguoi dung: "dong chu do nam trong bang step log luon, nam duoi
    # day thi kho") - giu THAM CHIEU expander (bien `_log_expander`) de GHI TIEP vao no o cuoi
    # script (sau khi render anh xong, luc do moi tinh duoc _render_elapsed) - Streamlit cho
    # phep ghi vao 1 container da tao TU TRUOC bat ky luc nao trong cung 1 lan chay script.
    _log_expander = st.expander(f"📋 Log từng bước ({len(_r_log_steps)} bước)", expanded=True)
    with _log_expander:
        for i, s in enumerate(_r_log_steps, 1):
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

    # 2026-08-15 (theo yeu cau nguoi dung: "log khong dua ra het van de, tong 20s nhung log chi
    # 5-6s") - `elapsed` o tren CHI do toi luc search_dense()/rerank xong (het spinner), KHONG
    # tinh phan RENDER ket qua (st.image tung dong) o duoi day - day chinh la nghi van "thoi
    # gian mat tich". Do rieng doan nay de XAC NHAN THAT thay vi doan, hien canh "Tong thoi gian
    # xu ly" cho de doi chieu.
    _t_render_start = time.perf_counter()

    if results.empty:
        st.warning("Không tìm thấy kết quả nào khớp bộ lọc.")
    elif _r_is_qa:
        # 2026-08-15 (cutover): Q&A gio chay tren bo dense - anh doc THANG tu row["path"]
        # (dia local), khong qua Keyframes_*.zip cua BTC nua (xem submission_pipeline.answer_qa).
        st.caption(f"{len(results)} kết quả — chỉ {min(int(_r_vqa_top_n), len(results))} ứng viên đầu được hỏi VQA thật, "
                   f"các ứng viên sau dùng lại câu trả lời của ứng viên đầu")
        cols = st.columns(4)
        for i, row in results.iterrows():
            col = cols[i % 4]
            with col:
                _render_image_with_vlm_overlay(row["path"], f"qa_{i}")
                vqa_tag = "🔎 VQA thật" if i < _r_vqa_top_n else "↪ dùng lại rank 1"
                backfill_tag = " 🔓" if row.get("is_backfill") else ""
                st.markdown(
                    f"**{row['video_id']}**{backfill_tag} · frame `{row['frame_id']}` ({vqa_tag})  \n"
                    f"**Trả lời:** {row['answer']}"
                )
                _col_vid, _col_vlm = st.columns(2, vertical_alignment="bottom")
                with _col_vid:
                    _render_video_toggle(row["video_id"], int(row["frame_id"]), f"qa_{i}")
                with _col_vlm:
                    _render_vlm_ocr_verify(row["path"], row["video_id"], int(row["frame_id"]), f"qa_{i}")
    elif _r_is_temporal:
        st.caption(f"{len(results)} video khớp chuỗi {len(_r_anchors)} anchor (đúng thứ tự thời gian)")
        for _, row in results.iterrows():
            st.markdown(f"**{row['video_id']}** · score={row['score']:.3f}")

            # Timeline truc quan (2026-08-11) - cham diem tung moc theo dung vi tri thoi gian
            # trong video, thay vi chi liet ke anh theo hang - de soat NHANH thu tu co dung
            # khong (dac biet khi cac moc gan nhau ve thoi gian, kho thay qua text).
            times = [float(row[f"anchor{i}_pts_time"]) for i in range(len(_r_anchors))]
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

            cols = st.columns(len(_r_anchors))
            for i, anchor in enumerate(_r_anchors):
                with cols[i]:
                    _trake_widget_key = f"trake_{row['video_id']}_{i}_{int(row[f'anchor{i}_frame_id'])}"
                    # 2026-08-15 (migrate sang bo dense): doc anh THANG tu row[f"anchor{i}_path"]
                    # (dia local), khong qua Keyframes_*.zip cua BTC nua (xem tiers/dense_temporal.py).
                    _render_image_with_vlm_overlay(row[f"anchor{i}_path"], _trake_widget_key)
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
                    _col_vid, _col_vlm = st.columns(2, vertical_alignment="bottom")
                    with _col_vid:
                        _render_video_toggle(row["video_id"], int(row[f"anchor{i}_frame_id"]), _trake_widget_key, fixed_window=True)
                    with _col_vlm:
                        _render_vlm_ocr_verify(
                            row[f"anchor{i}_path"], row["video_id"], int(row[f"anchor{i}_frame_id"]), _trake_widget_key
                        )
            st.divider()
    else:
        # duong mac dinh (2026-08-15, cutover): search_dense() - cot path/frame_id (xem
        # tiers/dense_search.py), anh nam THANG tren dia local, khong qua Keyframes_*.zip.
        st.caption(f"{len(results)} kết quả — model: {_r_dense_model}")
        cols = st.columns(4)
        for i, row in results.iterrows():
            col = cols[i % 4]
            with col:
                _render_image_with_vlm_overlay(row["path"], f"kis_{i}")
                # 🔓 = ket qua BU THEM (khong khop bo loc cung, xem is_backfill o dense_search.py)
                backfill_tag = " 🔓" if row.get("is_backfill") else ""
                st.markdown(
                    f"**{row['video_id']}**{backfill_tag} · frame `{row['frame_id']}` · score={row['score']:.3f}"
                )
                _col_vid, _col_vlm = st.columns(2, vertical_alignment="bottom")
                with _col_vid:
                    _render_video_toggle(row["video_id"], int(row["frame_id"]), f"kis_{i}")
                with _col_vlm:
                    _render_vlm_ocr_verify(row["path"], row["video_id"], int(row["frame_id"]), f"kis_{i}")

    _render_elapsed = time.perf_counter() - _t_render_start
    # 2026-08-15 (theo yeu cau nguoi dung: "dong chu do nam trong bang step log luon" + "phan
    # ngoai thuc thi query thi de mau khac cho de nhin") - GHI TIEP vao _log_expander da tao o
    # tren (thay vi st.markdown roi ben duoi trang) - to mau CAM de phan biet voi cac step tinh
    # toan (LLM/encode/rerank, mau mac dinh) vi day la thoi gian VE UI, khong phai tim kiem.
    with _log_expander:
        st.markdown(
            f'<span style="color:#f59e0b;">⏱ Render kết quả (ảnh + widget) — NGOÀI phần thực '
            f'thi query: {_render_elapsed:.2f}s</span>  \n'
            f'<span style="color:#f59e0b;">Tổng cộng thực tế (thực thi + render) ≈ '
            f'{elapsed + _render_elapsed:.2f}s</span>',
            unsafe_allow_html=True,
        )

# Popup video noi - goi 1 LAN o day, KHONG phu thuoc "last_search" co ton tai hay khong (CSS
# position:fixed tu dat dung goc man hinh, khong theo vi tri goi trong code) - _render_video_
# toggle() o cac dong ket qua chi SET session_state.active_video, popup nay moi thuc su ve.
_render_floating_video_player()
# (2026-08-17: khong con popup VLM rieng o day - overlay ket qua VLM gio DE LEN TREN chinh
# anh cua no, xem _render_image_with_vlm_overlay - render TAI CHO, khong phai 1 popup toan trang.)
