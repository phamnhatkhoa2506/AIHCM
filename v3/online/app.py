"""Giao diện đơn giản: nhập query text -> hiển thị top-k frame khớp nhất (ảnh + info).

Chạy: streamlit run app.py
"""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/

import time
from datetime import datetime

import statistics

import pandas as pd
import streamlit as st

from label_translate import resolve as resolve_label_vi
from app_flags import DISABLE_LLM_ENTITY_HARD_FILTER, ModalTimeoutError, NIMTimeoutError
from objects_canvas import objects_canvas
from query_distill import DEFAULT_DISTILL_MODEL, DISTILL_MODELS
from query_planner import extract_entities
from steplog import StepLog
from submission_pipeline import DEFAULT_VLM_OCR_MODEL, VLM_OCR_MODELS, answer_qa, vlm_read_text
from tiers import dense_temporal
from tiers.dense_search import DEFAULT_SCORE_ALGORITHM, SCORE_ALGORITHMS, apply_region_clip_rerank, search_dense, _fps_by_video, _frame_idx_by_video
from tiers.tier1_filter import DEFAULT_OCR_ALGORITHM, OCR_MATCH_ALGORITHMS
from video_clip import get_shot_clip_bytes, get_fixed_window_clip_bytes, extract_single_frame
from video_audio import read_video_bytes
from vlm_corrections import save_approved_vlm_text
from trake_corrections import save_trake_correction

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


# 2026-08-20 (theo yeu cau nguoi dung, sau khi lam ro y: "bấm vào thì sẽ scroll tới vị trí của
# frame hiển thị trên UI đó" - KHONG phai mo popup video nhu _render_video_toggle, ma la CUON
# TRANG toi dung the ket qua dang hien trong luoi ket qua) - dung anchor HTML thuan (`<div
# id=...>` + `<a href="#...">`, trinh duyet TU xu ly cuon muot, KHONG can JS/component rieng).
# 1 anchor id = 1 widget_key (da la duy nhat cho tung the ket qua san co trong toan app).
def _anchor_id(widget_key: str) -> str:
    return f"jump-{widget_key}"


def _render_scroll_anchor(widget_key: str) -> None:
    """Dat 1 anchor VO HINH ngay DAU the ket qua (KIS/Q&A/TRAKE) - muc tieu de link trong bang
    nop bai (sidebar) cuon toi. Goi 1 LAN duy nhat/the, cang SOM cang tot trong the do."""
    st.markdown(f'<div id="{_anchor_id(widget_key)}"></div>', unsafe_allow_html=True)


def _render_jump_link(widget_key: str | None) -> None:
    """Link "🔖" cuon toi anchor cua the ket qua tuong ung (xem _render_scroll_anchor) - dat o
    hang nut trong bang nop bai. widget_key=None (vd dong tu nut tu dong dien, chua gan duoc
    the ket qua cu the nao) -> KHONG ve gi (khong co noi de cuon toi)."""
    if not widget_key:
        st.caption("—")
        return
    st.markdown(
        f'<a href="#{_anchor_id(widget_key)}" target="_self" '
        f'style="text-decoration:none;font-size:1.1em;" title="Cuộn tới kết quả này">🔖</a>',
        unsafe_allow_html=True,
    )


def _render_video_toggle(video_id: str, frame_id: int, widget_key: str, fixed_window: bool = False) -> None:
    """Nút "▶ Video" dưới mỗi keyframe — bấm CHỌN video này làm nội dung của popup nổi DUY
    NHẤT (xem _render_floating_video_player), KHÔNG tự phát tại chỗ nữa. Bấm nút khác sẽ
    THAY THẾ popup, không mở thêm cái mới.

    fixed_window=True (2026-08-18, chỉ TRAKE truyền vào — xem call site): phát popup cắt theo
    cửa sổ thời lượng CỐ ĐỊNH quanh frame_id thay vì theo đúng ranh giới shot (KIS/Q&A giữ
    nguyên mặc định False)."""
    if st.button("▶", key=f"vidbtn_{widget_key}", help="Xem đoạn video quanh frame này"):
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


def _render_trake_frame_tune(
    video_id: str, anchor_index: int, anchor_text: str,
    cur_frame_id: int, cur_pts_time: float, widget_key: str,
) -> None:
    """Nút "🎯 Tinh chỉnh" dưới mỗi mốc TRAKE (2026-08-20, theo yêu cầu người dùng: "chức năng
    xem video của kết quả temporal để tinh chỉnh lại frame sau khi đã có kết quả... có chức
    năng approve frame mong muốn khi đã kiểm duyệt qua video. Video phải dễ dàng thao tác").

    Popover (cùng pattern lazy st.popover(on_change="rerun") với _render_vlm_ocr_verify — xem
    docstring hàm đó cho lý do chọn popover thay vì CSS/container thủ công) chứa 2 lớp thao tác:
      1. VIDEO xem tự do quanh mốc (st.video + start_time, kéo-thả bằng control gốc trình
         duyệt — "dễ dàng thao tác" theo đúng yêu cầu) để ĐỊNH VỊ đúng khoảnh khắc.
      2. Sau khi định vị được, dùng st.slider (bước 0.2s, mịn hơn nhiều so với mật độ mẫu dense
         gốc ~0.55-2.65s/frame) + ẢNH XEM TRƯỚC LIVE (extract_single_frame — trích ĐÚNG frame
         tại giây đó, không giới hạn theo các frame đã có sẵn trong dense_meta.parquet) để
         CHỐT chính xác 1 frame.

    "✅ Duyệt frame này": (a) ghi vào trake_corrections.TRAKE_APPROVED_PATH (lưu tham khảo, xem
    docstring file đó), (b) override HIỂN THỊ ngay trong phiên hiện tại qua
    st.session_state.trake_frame_overrides[(video_id, anchor_index)] — áp dụng NGAY cho card
    đang xem mà KHÔNG cần chạy lại search (kết quả gốc trên đĩa/ranking không đổi, chỉ đổi
    những gì đang HIỂN THỊ cho người dùng)."""
    pop = st.popover("🎯", help="Tinh chỉnh lại frame của mốc này (xem video, chọn đúng khoảnh khắc)",
                      on_change="rerun", key=f"tunepop_{widget_key}")
    if not pop.open:
        return
    with pop:
        st.markdown("**Xem video quanh mốc này** _(kéo-thả để tìm đúng khoảnh khắc)_")
        window_start = max(0.0, cur_pts_time - 10.0)
        try:
            st.video(get_fixed_window_clip_bytes(video_id, cur_frame_id, window_seconds=20.0),
                     start_time=int(window_start))
        except Exception as e:
            st.error(f"Không phát được video ({e})")

        slider_key = f"tuneslider_{widget_key}"
        if slider_key not in st.session_state:
            st.session_state[slider_key] = cur_pts_time
        t_pick = st.slider(
            "Thời điểm chính xác (giây)", min_value=max(0.0, cur_pts_time - 10.0),
            max_value=cur_pts_time + 10.0, step=0.2, key=slider_key,
        )

        try:
            preview_path = extract_single_frame(video_id, t_pick)
            st.image(str(preview_path), caption=f"t={t_pick:.1f}s")
        except Exception as e:
            st.error(f"Không trích được frame xem trước ({e})")
            preview_path = None

        if preview_path is not None and st.button("✅ Duyệt frame này", key=f"tuneapprove_{widget_key}"):
            fps = _fps_by_video().get(video_id, 25.0)
            new_frame_id = round(t_pick * fps)
            save_trake_correction(video_id, anchor_index, anchor_text,
                                   cur_frame_id, new_frame_id, t_pick)
            st.session_state.setdefault("trake_frame_overrides", {})[(video_id, anchor_index)] = {
                "frame_id": new_frame_id, "pts_time": t_pick, "path": str(preview_path),
            }
            st.success(f"✅ Đã duyệt frame `{new_frame_id}` (t={t_pick:.1f}s) cho mốc này.")


# ============================================================ Nộp bài (2026-08-20)
# 2026-08-20 (theo yeu cau nguoi dung: "Với mỗi frame thêm nút Submit trực tiếp. Có 1 nút để
# tự động thêm toàn bộ frame được xếp từ rank cap nhất -> thấp (tổng frame được submit thủ
# công + tự động = 100)") - danh sach nop bai giu trong session_state["submissions_by_query"]
# (dict[str, list[dict]], 1 bucket/cau hoi - xem _active_submission_list ben duoi; MOI dict
# trong 1 bucket co "mode" + cac truong theo dung dinh dang BTC - xem submission_pipeline.py
# docstring dau file: KIS=[video_id,frame_id], Q&A=[video_id,frame_id,answer], TRAKE=[video_id,
# frame_id_1..n]). Toi da SUBMISSION_MAX=100 dong/CAU HOI (khop dung SUBMISSION_TOP_K cua BTC) -
# nut nop tay VA nut tu dong DEU cong chung vao 1 bucket, KHONG tach rieng 2 quota.
SUBMISSION_MAX = 100

# 2026-08-20 (theo yeu cau nguoi dung: "BTC yêu cầu submission cho mỗi câu phải 100 câu trả
# lời, nhưng thuật toán của chúng ta đang làm mất đi cái logic đó" - phat hien qua screenshot
# "Danh sách nộp bài (45/100)" dung CHUNG 1 pool 100 cho CA PHIEN, bat ke dang lam CAU HOI nao -
# neu doi qua cau khac giua chung, 2 cau se bi TRON LAN vao CHUNG 1 file 100 dong, thay vi MOI
# CAU rieng 100 dong nhu BTC doi hoi thuc su) - BO HAN 1 danh sach phang session_state[
# "submission_list"], thay bang session_state["submissions_by_query"]: dict[str, list[dict]] -
# MOI "cau hoi" (dinh danh boi CHINH o "Name" da them truoc do, vd "query-p1-1-kis") co 1 bucket
# 100-dong RIENG. Doi "Name" (vd tu "...-1-..." sang "...-2-...") = TU DONG chuyen sang cau hoi
# khac, bucket rong moi neu chua tung dung ten do, hoac phuc hoi dung bucket cu neu go LAI ten
# da dung truoc do (khong mat du lieu, chi chuyen "khung nhin" dang xem).
def _active_submission_key() -> str:
    """Tra ve dinh danh CAU HOI HIEN TAI - dung 1 ID ON DINH RIENG (session_state[f"submission_
    current_id_{suffix}"]), KHONG con dung truc tiep noi dung o "Name" nua.

    2026-08-20 (theo yeu cau nguoi dung: "bỗng nhiên kết quả mất luôn và tên file không thấy gì
    nữa" - BUG THAT nghiem trong o ban truoc: dung CHINH text cua o "Name" lam khoa luu tru ->
    MOI LAN go/sua 1 KY TU trong o do (vd sua "x" thanh "3" cho dung so cau) la 1 CHUOI KHOA
    HOAN TOAN MOI chua tung ton tai -> danh sach da nop (o khoa CU) bi "treo" lai, man hinh hien
    bucket MOI toanh RONG - dung y ("mất luôn"). Fix: tach RIENG "dinh danh luu tru" (ID on
    dinh, CHI doi khi bam nut "🆕 Câu hỏi mới" ro rang - xem _render_submission_panel) khoi
    "Name" (gio THUAN COSMETIC, chi anh huong TEN FILE tai ve, sua thoai mai KHONG lam mat du
    lieu nua)."""
    suffix = _FILENAME_MODE_SUFFIX.get(mode, "kis")
    id_key = f"submission_current_id_{suffix}"
    if id_key not in st.session_state:
        st.session_state[id_key] = f"{suffix}_default"
    return st.session_state[id_key]


def _active_submission_list() -> list[dict]:
    """Danh sach nop bai (list[dict]) CUA DUNG cau hoi hien tai (xem _active_submission_key) -
    dung ham nay (KHONG doc thang session_state["submission_list"] nua, bien do KHONG con dung)
    o MOI noi can doc/ghi danh sach nop bai."""
    return st.session_state.setdefault("submissions_by_query", {}).setdefault(_active_submission_key(), [])

# 2026-08-20: dat SOM (truoc sidebar - block sidebar chay TRUOC, can bien nay ngay luc do) - xem
# checkbox Region-CLIP rerank o sidebar.
_REGION_CLIP_CHECKBOX_HELP = (
    "Ép ưu tiên frame khớp thuộc tính LLM trích được (vd 'áo dài màu tím'), không chỉ dựa vào "
    "CLIP đoán cả câu. Chạy qua server Modal riêng (aic2026-region-rerank, luôn giữ ấm), ~2s/lần. "
    "Tự động BỎ QUA (không xáo kết quả) nếu SigLIP2 không đủ phân biệt thuộc tính đó trên các "
    "ứng viên (dynamic range điểm quá thấp — xem log chi tiết)."
)


def _submission_row_key(row: dict) -> tuple:
    """Key chong trung 1 dong nop bai - theo dung "mode" (kis/qa/trake co so cot khac nhau)."""
    if row["mode"] == "trake":
        return (row["mode"], row["video_id"], tuple(row["frame_ids"]))
    return (row["mode"], row["video_id"], row["frame_id"])


def _submit_row(mode: str, row: dict) -> bool:
    """Them 1 dong vao danh sach nop bai - False neu da du 100 hoac da co san (trung), True neu
    them thanh cong. KHONG rerun o day (de nguoi goi tu quyet dinh - vd nut tu dong goi lap
    nhieu lan lien tiep, chi rerun 1 lan cuoi)."""
    lst = _active_submission_list()
    full_row = {"mode": mode, **row}
    key = _submission_row_key(full_row)
    if any(_submission_row_key(r) == key for r in lst):
        return False
    if len(lst) >= SUBMISSION_MAX:
        return False
    lst.append(full_row)
    return True


def _unsubmit_row(mode: str, row: dict) -> bool:
    """Hoàn tác 1 dòng ĐÃ nộp — xoá KHỎI danh sách nộp bài theo đúng key (2026-08-20, theo yêu
    cầu người dùng: "thêm thao tác Hoàn tác ghi đã nộp frame đó"). True nếu tìm thấy và xoá
    được, False nếu không có trong danh sách (không nên xảy ra khi gọi từ nút "✅ Đã nộp")."""
    lst = _active_submission_list()
    key = _submission_row_key({"mode": mode, **row})
    for idx, r in enumerate(lst):
        if _submission_row_key(r) == key:
            lst.pop(idx)
            return True
    return False


def _render_submit_button(mode: str, row: dict, widget_key: str, label: str | None = None) -> None:
    """Nút "📤 Nộp" cho 1 frame/chuỗi cụ thể - đặt cạnh các nút khác (Video/VLM) trong mỗi thẻ
    kết quả. Đã nộp rồi -> nút "✅" CHUYỂN THÀNH nút Hoàn tác (bấm để xoá khỏi danh sách, không
    còn bị khoá cứng như trước — 2026-08-20, theo yêu cầu người dùng).

    label (2026-08-20, theo yêu cầu người dùng: "Có 2 nút Submit riêng biệt: Submit Temporal /
    Submit TRAKE") - None (mặc định, dùng ở mọi nơi khác) -> nút CHỈ ICON, gọn cho hàng nút dày
    đặc. Có label -> hiện CHỮ kèm icon (dùng khi cần PHÂN BIỆT rõ 2 nút cạnh nhau, như trong
    Playback đa mốc — chỉ icon thì không phân biệt được 2 nút giống hệt nhau)."""
    icon_submit = "📤" if label is None else f"📤 {label}"
    icon_done = "✅" if label is None else f"✅ {label} (đã nộp)"
    icon_full = "📤" if label is None else f"📤 {label} (đã đủ {SUBMISSION_MAX})"
    lst = _active_submission_list()
    full_row = {"mode": mode, **row}
    already = any(_submission_row_key(r) == _submission_row_key(full_row) for r in lst)
    if already:
        if st.button(icon_done, key=f"submitted_{widget_key}", help="Đã nộp — bấm để hoàn tác (xoá khỏi danh sách nộp bài)"):
            _unsubmit_row(mode, row)
            st.rerun()
    elif len(lst) >= SUBMISSION_MAX:
        st.button(icon_full, key=f"submitfull_{widget_key}", disabled=True,
                   help=f"Danh sách nộp bài đã đủ {SUBMISSION_MAX} dòng — xoá bớt trước khi nộp thêm.")
    else:
        if st.button(icon_submit, key=f"submitbtn_{widget_key}", help="Thêm dòng này vào danh sách nộp bài"):
            _submit_row(mode, row)
            st.rerun()


# 2026-08-20 (theo yeu cau nguoi dung: "Thêm cho mình Mode VQA — Khi nhấn Submit tại từng
# Frame: Hiển thị ô nhập Answer... Nhấn Confirm để submit") - KHAC _render_submit_button thuong
# (nop NGAY khi bam 1 nut duy nhat): nut "📤" o day CHI MO 1 o nhap answer + nut Confirm rieng,
# PHAI bam Confirm moi thuc su them vao danh sach nop bai - tranh nop nham cau tra loi VQA tu
# dong/rong ma nguoi dung chua kip doc/sua. Dinh dang dong CSV KHONG doi (van
# `video_id,frame_id,"answer"`, xem _submission_to_csv) - chi khac quy trinh nhap lieu. Tach 2
# ham (_render_qa_submit_toggle + _render_qa_submit_panel, xem docstring tung ham) vi o nhap
# Answer can TOAN CHIEU RONG the, khong vua hang icon ngang hep nhu cac nut khac.
def _render_qa_submit_toggle(video_id: str, frame_id: int, widget_key: str) -> None:
    """Nút icon (📤/✅) đặt TRONG hàng nút ngang cùng Video/Playback/VLM - CHỈ mở/đóng bảng
    nhập Answer (xem _render_qa_submit_panel), KHÔNG tự nộp ngay khi bấm."""
    lst = _active_submission_list()
    key_tuple = _submission_row_key({"mode": "qa", "video_id": video_id, "frame_id": frame_id})
    already = any(_submission_row_key(r) == key_tuple for r in lst)
    open_key = f"qa_submit_open_{widget_key}"

    if already:
        if st.button("✅", key=f"qa_submitted_{widget_key}", help="Đã nộp — bấm để hoàn tác"):
            _unsubmit_row("qa", {"video_id": video_id, "frame_id": frame_id})
            st.rerun()
    elif len(lst) >= SUBMISSION_MAX:
        st.button("📤", key=f"qa_submitfull_{widget_key}", disabled=True,
                   help=f"Danh sách nộp bài đã đủ {SUBMISSION_MAX} dòng — xoá bớt trước khi nộp thêm.")
    else:
        if st.button("📤", key=f"qa_submitbtn_{widget_key}", help="Nhập câu trả lời rồi bấm Confirm để nộp"):
            st.session_state[open_key] = True
            st.rerun()


def _render_qa_submit_panel(video_id: str, frame_id: int, default_answer: str, widget_key: str) -> None:
    """Bảng nhập Answer + Confirm/Huỷ - gọi SAU KHI hàng nút icon (_render_qa_submit_toggle)
    đã đóng lại, vì ô nhập cần TOÀN CHIỀU RỘNG thẻ, không vừa hàng icon ngang hẹp như các nút
    khác. Chỉ vẽ gì khi đang "mở" (đã bấm 📤) và CHƯA nộp - PHẢI bấm Confirm mới thực sự thêm
    vào danh sách nộp bài (row["mode"]=="qa" nên đã tự nộp trước đây khi bấm nút duy nhất -
    tách 2 bước để tránh nộp nhầm câu trả lời VQA tự động/rỗng mà người dùng chưa kịp sửa)."""
    lst = _active_submission_list()
    key_tuple = _submission_row_key({"mode": "qa", "video_id": video_id, "frame_id": frame_id})
    already = any(_submission_row_key(r) == key_tuple for r in lst)
    open_key = f"qa_submit_open_{widget_key}"
    if already or not st.session_state.get(open_key):
        return

    answer_key = f"qa_submit_answer_{widget_key}"
    if answer_key not in st.session_state:
        st.session_state[answer_key] = default_answer
    st.text_input("Answer", key=answer_key, placeholder="Nhập câu trả lời...")
    col_confirm, col_cancel = st.columns(2)
    if col_confirm.button("✅ Confirm", key=f"qa_confirm_{widget_key}"):
        _submit_row("qa", {
            "video_id": video_id, "frame_id": frame_id, "answer": st.session_state[answer_key],
            "_anchor": _anchor_id(widget_key),
        })
        st.session_state[open_key] = False
        st.rerun()
    if col_cancel.button("✕ Huỷ", key=f"qa_cancelbtn_{widget_key}"):
        st.session_state[open_key] = False
        st.rerun()


def _result_row_to_submission(mode: str, row, n_anchors: int = 0) -> dict:
    """Chuyển 1 dòng results (pandas Series) thành dict đúng schema nộp bài theo mode - dùng
    CHUNG cho cả nút Submit lẻ và nút tự động điền (tránh lệch định dạng giữa 2 đường)."""
    if mode == "kis":
        return {"video_id": row["video_id"], "frame_id": int(row["frame_id"])}
    if mode == "qa":
        return {"video_id": row["video_id"], "frame_id": int(row["frame_id"]), "answer": row["answer"]}
    if mode == "trake":
        return {
            "video_id": row["video_id"],
            "frame_ids": [int(row[f"anchor{i}_frame_id"]) for i in range(n_anchors)],
        }
    if mode == "temporal":
        # xem docstring _render_playback cho dinh nghia "frame giua" (median, khong phai TB cong)
        frame_ids = [int(row[f"anchor{i}_frame_id"]) for i in range(n_anchors)]
        return {"video_id": row["video_id"], "frame_id": int(statistics.median(frame_ids))}
    raise ValueError(f"mode không hợp lệ: {mode}")


def _render_autofill_button(mode: str, results: pd.DataFrame, n_anchors: int = 0) -> None:
    """Nút tự động điền — thêm các dòng CHƯA CÓ trong danh sách nộp bài, theo ĐÚNG THỨ TỰ rank
    hiện có (cao -> thấp), cho tới khi đủ SUBMISSION_MAX (không tách quota riêng với nộp tay —
    "tổng frame được submit thủ công + tự động = 100")."""
    n_current = len(_active_submission_list())
    remaining = SUBMISSION_MAX - n_current
    label = (f"⬇️ Tự động điền đủ {SUBMISSION_MAX} (còn thiếu {remaining})" if remaining > 0
              else f"⬇️ Đã đủ {SUBMISSION_MAX}")
    if st.button(label, disabled=remaining <= 0, key=f"autofill_{mode}"):
        added = 0
        for _, row in results.iterrows():
            if added >= remaining:
                break
            if _submit_row(mode, _result_row_to_submission(mode, row, n_anchors)):
                added += 1
        st.success(f"Đã tự động thêm {added} dòng.")
        st.rerun()


def _submission_to_csv(lst: list[dict]) -> str:
    lines = []
    for r in lst:
        if r["mode"] == "trake":
            lines.append(",".join([r["video_id"]] + [str(f) for f in r["frame_ids"]]))
        elif r["mode"] == "qa":
            # 2026-08-20 (theo yeu cau nguoi dung: format 'L02_V011,1200,"Năm người"') - BOC
            # NGOAC KEP cau tra loi (chuan CSV - cho phep dau phay/xuong dong BEN TRONG cau tra
            # loi ma khong pha vo cot), tu ngoac kep BEN TRONG nhan doi thanh "" (escape CSV
            # chuan) thay vi loai bo dau phay/xuong dong nhu ban truoc.
            answer = str(r["answer"]).replace('"', '""')
            lines.append(f'{r["video_id"]},{r["frame_id"]},"{answer}"')
        else:
            lines.append(f"{r['video_id']},{r['frame_id']}")
    return "\n".join(lines)


# 2026-08-20 (theo yeu cau nguoi dung: "Thêm ô Name gần nút Download CSV") - dat ten file CSV
# nop bai theo dung quy uoc BTC (query-p1-x-A): x nguoi dung tu go (vd so thu tu cau hoi), A tu
# dong theo Mode dang chon. Chi anh huong TEN FILE tai xuong, KHONG doi noi dung/dinh dang CSV.
_FILENAME_MODE_SUFFIX = {"KIS": "kis", "Temporal": "kis", "TRAKE": "trake", "Q&A": "qa"}


def _render_submission_panel() -> None:
    """Bảng điều khiển danh sách nộp bài — đặt ở sidebar, LUÔN hiện (không phụ thuộc mode hiện
    tại) để người dùng theo dõi tiến độ xuyên suốt phiên làm việc, kể cả khi đổi qua lại giữa
    KIS/Q&A/TRAKE."""
    lst = _active_submission_list()
    with st.sidebar:
        st.divider()
        st.markdown(f"### 📤 Danh sách nộp bài ({len(lst)}/{SUBMISSION_MAX})")

        # `mode` la bien global cua sidebar (dinh nghia truoc _render_submission_panel() duoc
        # goi - xem cuoi file) - suffix "A" TU DONG doi theo Mode dang chon.
        suffix = _FILENAME_MODE_SUFFIX.get(mode, "kis")
        default_name = f"query-p1-x-{suffix}"
        # 2026-08-20 (theo yeu cau nguoi dung: "bỗng nhiên kết quả mất luôn và tên file không
        # thấy gì nữa" - xem ghi chu day du o _active_submission_key) - o Name gio LUON hien
        # (khong con nam trong `if lst:` - do CHINH la ly do "tên file không thấy gì nữa" khi
        # bucket rong) va THUAN COSMETIC (chi anh huong ten file tai ve), sua thoai mai KHONG
        # con lam mat du lieu nua.
        file_name_input = st.text_input(
            "Name", value=default_name, key=f"submission_file_name_{suffix}",
            help="Tên file CSV tải xuống — thay 'x' bằng số thứ tự câu hỏi của bạn "
            "(vd query-p1-3-kis). Phần cuối tự đổi theo Mode đang chọn. Sửa ô này KHÔNG ảnh "
            "hưởng danh sách đã nộp — chỉ đổi tên file khi tải về.",
        )
        csv_file_name = (file_name_input or default_name).strip() or default_name
        if not csv_file_name.lower().endswith(".csv"):
            csv_file_name += ".csv"

        # 2026-08-20 (theo yeu cau nguoi dung, tiep tuc) - nut RIENG, RO RANG de chuyen sang
        # cau hoi MOI (bucket 100-dong rong khac) - THAY THE cho co che cu (ngam dinh theo noi
        # dung o Name, da gay bug mat du lieu). Khong xoa bucket cu (van con trong session_state
        # "submissions_by_query", co the quay lai neu can - hien tai chua co UI chon lai bucket
        # cu, chi co the reset ve "{suffix}_default" bang cach xoa cache trinh duyet/restart).
        if st.button("🆕 Câu hỏi mới (bắt đầu 100 dòng mới)", key=f"submission_newq_{suffix}",
                     help="Bắt đầu 1 danh sách nộp bài MỚI, RIÊNG cho câu hỏi tiếp theo (không "
                     "xoá câu hiện tại) — dùng khi đã xong 1 câu, chuyển sang câu khác."):
            counter_key = f"submission_id_counter_{suffix}"
            n = st.session_state.get(counter_key, 0) + 1
            st.session_state[counter_key] = n
            st.session_state[f"submission_current_id_{suffix}"] = f"{suffix}_{n}"
            st.rerun()

        if lst:
            col_clear, col_dl = st.columns(2)
            if col_clear.button("🗑 Xoá hết"):
                # CHI xoa bucket cua CAU HOI dang xem - KHONG dung session_state.submission_
                # list = [] nua (bien do khong con ton tai, moi cau hoi co bucket rieng - xem
                # _active_submission_list).
                st.session_state.setdefault("submissions_by_query", {})[_active_submission_key()] = []
                st.rerun()
            col_dl.download_button(
                "⬇️ CSV", _submission_to_csv(lst), file_name=csv_file_name,
                mime="text/csv", key="submission_csv_dl",
            )
            with st.expander(f"Xem {len(lst)} dòng đã chọn"):
                # 2026-08-20 (theo yeu cau nguoi dung: "muốn có chức năng xóa trực tiếp ở đây +
                # di chuyển lên xuống") - moi dong giờ co 3 nut nho (↑/↓/✕) canh caption, sua
                # THANG vao `lst` (= _active_submission_list(), bucket cua CAU HOI dang xem) roi
                # st.rerun() - giong pattern nut ✕ xoa mốc TRAKE da lam (xem trake_del_{i}).
                for i, r in enumerate(lst, 1):
                    # 2026-08-20 (theo yeu cau nguoi dung, sua lai y sau khi lam ro: "ý mình là
                    # bấm vào thì sẽ scroll tới vị trí của frame hiển thị trên UI đó" - KHONG
                    # phai mo popup video (ban truoc dung _render_video_toggle, hieu SAI y) - ma
                    # la CUON TRANG toi dung the ket qua dang hien trong luoi ket qua ben duoi,
                    # xem _render_jump_link/_render_scroll_anchor. r.get("_anchor") duoc gan luc
                    # nop (xem cac call site _submit_row) - None neu nop qua nut TU DONG DIEN
                    # (khong gan voi 1 the cu the nao dang ve tren man hinh).
                    if r["mode"] == "trake":
                        label = f"{i}. [TRAKE] {r['video_id']} · {r['frame_ids']}"
                    elif r["mode"] == "qa":
                        label = f"{i}. [Q&A] {r['video_id']} · frame {r['frame_id']} · \"{r['answer']}\""
                    elif r["mode"] == "temporal":
                        label = f"{i}. [Temporal] {r['video_id']} · frame {r['frame_id']}"
                    else:
                        label = f"{i}. [KIS] {r['video_id']} · frame {r['frame_id']}"

                    col_jump, col_label, col_up, col_down, col_del = st.columns([1, 5, 1, 1, 1])
                    idx = i - 1
                    with col_jump:
                        _render_jump_link(r.get("_anchor"))
                    col_label.caption(label)
                    if col_up.button("↑", key=f"submission_up_{idx}", disabled=(idx == 0), help="Di chuyển lên"):
                        lst[idx - 1], lst[idx] = lst[idx], lst[idx - 1]
                        st.rerun()
                    if col_down.button("↓", key=f"submission_down_{idx}", disabled=(idx == len(lst) - 1), help="Di chuyển xuống"):
                        lst[idx + 1], lst[idx] = lst[idx], lst[idx + 1]
                        st.rerun()
                    if col_del.button("✕", key=f"submission_del_{idx}", help="Xoá dòng này"):
                        lst.pop(idx)
                        st.rerun()
        else:
            st.caption("Chưa có dòng nào — bấm \"📤 Nộp\" ở từng kết quả, hoặc nút tự động điền.")


PLAYBACK_NUDGE_FRAMES = 5  # buoc nhay lui/tien moi lan bam nut </> (theo yeu cau nguoi dung)


# 2026-08-20 (theo yeu cau nguoi dung: "frame được bấm chức năng playback sẽ được thay đổi bên
# ngoài luôn á bạn" - xac nhan trong Playback TRUOC DAY chi doi state RIENG BEN TRONG popup, cac
# noi hien thi BEN NGOAI (anh/so frame/nut "📤 Nộp" nhanh) van dung frame GOC he thong de xuat,
# khong doc lai gia tri DA XAC NHAN) - 2 ham duoi day dung CHUNG 1 nguon voi _render_playback
# (session_state[f"playback_state_{widget_key}"]) de moi noi GOI voi CUNG widget_key deu thay
# DUNG 1 gia tri, khong con lech nhau.
def _playback_confirmed_frame_id(widget_key: str, default_frame_id: int) -> int:
    """Doc frame DA XAC NHAN trong Playback (neu da tung mo + xac nhan) cho 1 the KIS/Q&A (1
    frame duy nhat) - tra ve default_frame_id (frame goc he thong de xuat) neu chua tung
    xac nhan gi khac."""
    state = st.session_state.get(f"playback_state_{widget_key}")
    return int(state[0]) if state else default_frame_id


def _frame_preview_path(video_id: str, frame_id: int, default_frame_id: int, default_path):
    """Path ảnh để hiển thị cho 1 frame - dùng THẲNG path có sẵn (dense-sampled, rẻ) nếu frame
    KHÔNG đổi so với đề xuất gốc, ngược lại trích MỚI qua extract_single_frame (frame tuỳ ý do
    người dùng chọn trong Playback, thường KHÔNG trùng đúng 1 frame dense-sampled có sẵn)."""
    if frame_id == default_frame_id:
        return default_path
    fps = _fps_by_video().get(video_id, 25.0)
    return str(extract_single_frame(video_id, frame_id / fps))


def _render_playback(
    video_id: str,
    frame_ids: list[int],
    labels: list[str],
    widget_key: str,
    mode: str,
    submit_extra: dict | None = None,
) -> None:
    """"🎬 Playback" (2026-08-20, theo yêu cầu người dùng — thay thế hoàn toàn bản cũ
    `_render_full_video_playback` chỉ phát video tĩnh):
      1. Timeline đánh dấu vị trí frame ĐANG chọn (đổi màu marker đang active).
      2. Nút "◀ -5"/"+5 ▶" di chuyển lùi/tiến PLAYBACK_NUDGE_FRAMES frame.
      3. Nút "📤 Nộp" NGAY TRONG playback — dùng đúng frame(s) đã chỉnh (không phải frame gốc
         hệ thống đề xuất, nếu người dùng đã nudge).
      4. Nhiều frame (TRAKE): TẤT CẢ mốc hiện trên CÙNG timeline, chọn mốc đang muốn chỉnh qua
         hàng nút "#i", nudge/nộp áp dụng cho mốc đang chọn / CẢ chuỗi tương ứng.

    frame_ids/labels: 1 phần tử (KIS/Q&A) hoặc N phần tử theo thứ tự (TRAKE, "Mốc 1".."Mốc n").
    State luu tai session_state[f"playback_state_{widget_key}"] — KHOI TAO 1 LAN tu frame_ids
    truyen vao, cac lan nudge sau CHI doi state (khong doi lai tu frame_ids goc, giu duoc dieu
    chinh cua nguoi dung qua nhieu lan mo lai dialog).

    2026-08-20 (theo yeu cau nguoi dung: "phần playback này khó thao tác... có thể làm cái này
    bằng cách cho 1 cửa sổ khác không") - DOI tu st.popover() (neo theo vi tri nut, kich thuoc
    NHO CO DINH, phai cuon rat nhieu ben trong 1 khung chat hep) sang st.dialog() (modal THAT
    SU, hien GIUA man hinh, rong toi 1280px voi width="large") - _render_playback() gio CHI ve
    1 nut bam mo dialog, phan noi dung THAT chuyen het sang _playback_dialog() (ham rieng,
    @st.dialog decorator - goi ham nay MOI THUC SU mo modal, xem Streamlit docs ve dialog)."""
    submit_extra = submit_extra or {}
    if st.button("🎬", key=f"playbackbtn_{widget_key}",
                 help="Phát toàn bộ video — xem/chỉnh vị trí frame rồi nộp trực tiếp (cửa sổ riêng)"):
        _playback_dialog(video_id, frame_ids, labels, widget_key, mode, submit_extra)


@st.dialog("🎬 Playback — xem/chỉnh frame", width="large")
def _playback_dialog(
    video_id: str,
    frame_ids: list[int],
    labels: list[str],
    widget_key: str,
    mode: str,
    submit_extra: dict,
) -> None:
    """Nội dung THẬT của Playback - xem docstring _render_playback (nơi gọi hàm này) cho lý do
    chuyển từ popover sang dialog. Streamlit tự động re-run ĐÚNG hàm này (không phải toàn bộ
    script) khi tương tác với widget bên trong (st.dialog kế thừa hành vi st.fragment) - dialog
    tự giữ mở qua các lần nudge/kéo slider, không cần tự quản lý trạng thái "đang mở"."""
    state_key = f"playback_state_{widget_key}"
    active_key = f"playback_active_{widget_key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = list(frame_ids)
    if active_key not in st.session_state:
        st.session_state[active_key] = 0
    cur_frames: list[int] = st.session_state[state_key]
    active = st.session_state[active_key]

    fps = _fps_by_video().get(video_id, 25.0)
    max_frame_arr = _frame_idx_by_video().get(video_id)
    max_frame = int(max_frame_arr.max()) if max_frame_arr is not None and len(max_frame_arr) else max(cur_frames)

    # Timeline (2026-08-20) - GIONG timeline TRAKE da co (dot + nhan theo % thoi gian), o
    # day THEM highlight cho marker DANG active (vien do) de phan biet khi co nhieu moc.
    t_axis_max = max(max_frame / fps, 1.0)
    timeline_html = '<div style="position:relative;height:40px;background:#eee;border-radius:4px;margin:8px 0;">'
    for i, fid in enumerate(cur_frames):
        pct = 100 * (fid / fps) / t_axis_max
        is_active = i == active
        dot_style = (
            "width:14px;height:14px;border:3px solid #16a34a;" if is_active
            else "width:10px;height:10px;border:none;"
        )
        timeline_html += (
            f'<div style="position:absolute;left:{pct:.1f}%;top:0;transform:translateX(-50%);text-align:center;">'
            f'<div style="{dot_style}border-radius:50%;background:#d33;margin:0 auto;"></div>'
            f'<div style="font-size:11px;white-space:nowrap;">{labels[i]}</div></div>'
        )
    timeline_html += "</div>"
    st.markdown(timeline_html, unsafe_allow_html=True)

    # Chon moc dang chinh (chi hien khi >1 frame, tuc TRAKE) - hang nut "#i".
    if len(cur_frames) > 1:
        with st.container(horizontal=True, gap="small"):
            for i in range(len(cur_frames)):
                btn_label = f"● {labels[i]}" if i == active else labels[i]
                if st.button(btn_label, key=f"playbackpick_{widget_key}_{i}"):
                    st.session_state[active_key] = i
                    st.rerun()

    active_frame_confirmed = cur_frames[active]
    # 2026-08-20 (theo yeu cau nguoi dung: "Playback Video – Đồng bộ Frame" - 3 thanh phan
    # (vi tri thanh keo, so Frame, anh Frame) LUON phai dong nhat, moi cach doi Frame deu
    # phai cap nhat CA 3). st.video() cua Streamlit KHONG co co che bao lai vi tri scrub ve
    # Python (gioi han nen tang - HTML5 <video> khong co callback JS-vao-Streamlit san co,
    # xem docstring PLAYBACK_NUDGE_FRAMES) nen KHONG THE dung thanh keo NATIVE cua video lam
    # nguon dong bo 2 chieu duoc. Thay bang 1 st.slider (buoc 1 frame) LAM "thanh Video" that
    # su dieu khien duoc 2 chieu: ca nut </> LAN keo slider DEU ghi vao CUNG 1 session_state
    # key (pending_key) - anh Frame duoi day trich THANG tu gia tri do (extract_single_frame)
    # nen 3 thanh phan (slider/so Frame/anh) khong bao gio lech nhau, vi CUNG doc 1 nguon.
    # video gốc (st.video) o cuoi CHI de xem TU DO quanh do (context), khong phai nguon dong
    # bo chinh.
    pending_key = f"playback_pending_{widget_key}_{active}"
    if pending_key not in st.session_state:
        st.session_state[pending_key] = active_frame_confirmed

    # 2026-08-20 (theo yeu cau nguoi dung: "mình muốn phần kéo frame và phần video bên dưới
    # nằm cùng hàng lại... đỡ kéo lên kéo xuống") - 2 COT CANH NHAU (trai: nudge/slider/anh
    # preview/xac nhan, phai: video goc) thay vi xep DOC nhu truoc - dialog width="large" (toi
    # 1280px) du rong cho ca 2 cot khong bi chat.
    col_frame, col_video = st.columns(2)
    with col_frame:
        col_prev, col_next = st.columns(2)
        if col_prev.button("◀ -5", key=f"playbackprev_{widget_key}"):
            st.session_state[pending_key] = max(0, st.session_state[pending_key] - PLAYBACK_NUDGE_FRAMES)
            st.rerun()
        if col_next.button("+5 ▶", key=f"playbacknext_{widget_key}"):
            st.session_state[pending_key] = min(max_frame, st.session_state[pending_key] + PLAYBACK_NUDGE_FRAMES)
            st.rerun()

        pending_frame = st.slider(
            f"{labels[active]} — Frame (kéo để tìm khoảnh khắc)",
            min_value=0, max_value=max_frame, step=1, key=pending_key,
        )
        t_pending = pending_frame / fps

        try:
            preview_path = extract_single_frame(video_id, t_pending)
            st.image(str(preview_path), caption=f"Frame {pending_frame} · t≈{t_pending:.1f}s")
        except Exception as e:
            st.error(f"Không trích được ảnh frame xem trước ({e})")

        # Nut xac nhan (2026-08-20, theo yeu cau nguoi dung: "Frame cũ chỉ được thay thế bằng
        # Frame hiện tại sau khi người dùng xác nhận") - MOI thay doi (slider LAN nut </>) chi
        # doi gia tri XEM THU (pending) - cur_frames[active] (gia tri THAT SU dung de nop bai,
        # hien marker tren timeline) CHI doi khi bam nut nay, khong tu dong theo pending.
        if pending_frame != active_frame_confirmed:
            st.caption(f"⏳ Đang xem thử frame `{pending_frame}` — frame ĐÃ CHỐT (dùng khi nộp) "
                       f"vẫn là `{active_frame_confirmed}`.")
            if st.button("✅ Xác nhận frame này", key=f"playbackconfirm_{widget_key}"):
                cur_frames[active] = pending_frame
                st.rerun()
        else:
            st.caption(f"✅ Frame đã chốt: `{active_frame_confirmed}`")

    with col_video:
        with st.spinner("Đang tải video gốc (lần đầu có thể chậm)..."):
            try:
                video_bytes = read_video_bytes(video_id)
            except Exception as e:
                st.error(f"Không tải được video gốc ({e})")
                video_bytes = None
        if video_bytes is not None:
            st.video(video_bytes, start_time=int(t_pending))

    if len(cur_frames) > 1:
        # 2026-08-20 (theo yeu cau nguoi dung: "Quy tắc quan trọng — Format Submit phải phụ
        # thuộc vào loại task đang chọn... người dùng không phải tự nhập format thủ công") -
        # `mode` truyen vao ("temporal" hoac "trake", xem sidebar "Loại truy vấn") QUYET
        # DINH DUY NHAT 1 dinh dang duoc sinh - KHONG con hien ca 2 nut de nguoi dung tu
        # chon nua (khac ban truoc "Có 2 nút Submit riêng biệt" — da THAY DOI theo yeu cau
        # moi nay, chinh xac hon: dinh dang phai TU DONG theo mode dang chon, khong phai
        # nguoi dung tu chon giua 2 nut).
        #   - mode="temporal": format nhu KIS (video_id, 1 frame_id DUY NHAT) - frame = frame
        #     GIUA cua cac frame tuong ung (median, khong phai trung binh cong - "lấy frame
        #     giữa" dung nghia la phan tu O GIUA khi sap xep, khong phai noi suy).
        #   - mode="trake": format day du (video_id, frame_id_1, ..., frame_id_n) - vd
        #     "L10_V001,1200,1850,2100,2450".
        changed = cur_frames != list(frame_ids)
        if changed:
            st.caption("✏️ Đã chỉnh so với đề xuất gốc — nộp sẽ dùng giá trị ĐÃ CHỈNH.")
        if mode == "temporal":
            median_frame = int(statistics.median(cur_frames))
            st.caption(f"Loại truy vấn hiện tại: **Temporal** — frame giữa = `{median_frame}`")
            _render_submit_button(
                "temporal", {"video_id": video_id, "frame_id": median_frame, "_anchor": _anchor_id(widget_key)},
                f"playbacksubmit_{widget_key}", label="Temporal",
            )
        else:
            # 2026-08-20 (theo yeu cau nguoi dung: "Mode TRAKE... Không có nút Submit bên
            # trong Playback Video... chỉ có nút Submit bên ngoài Playback") - BO nut Submit
            # o day cho rieng TRAKE (Temporal van giu nguyen o nhanh "if mode == temporal"
            # phia tren, KHONG bi anh huong - nguoi dung chi noi ro "Mode TRAKE"). Nop TRAKE
            # gio CHI qua nut "📤 Nộp" ben ngoai Playback (cap video, xem vong lap ket qua) -
            # van doc DUNG frame DA XAC NHAN trong Playback nay (qua playback_state, xem
            # _playback_confirmed_frame_id/_trake_chain_key o noi goi), khong mat gi ca, chi
            # bot 1 nut trung lap trong popup.
            st.caption("Loại truy vấn hiện tại: **TRAKE** — nộp đủ cả chuỗi bằng nút "
                       "\"📤 Nộp\" BÊN NGOÀI Playback (đã tự đọc đúng frame vừa xác nhận ở đây).")
    else:
        # 2026-08-20 (theo yeu cau nguoi dung: "VQA... Có thêm ô nhập câu trả lời") - CHI
        # hien voi Q&A (mode="qa") - o nhap CO THE CHINH lai cau tra loi (mac dinh = answer
        # co san trong submit_extra, RONG neu LVLM dang tat - xem submission_pipeline.
        # answer_qa use_lvlm) TRUOC khi nop, khong bat buoc dung nguyen VQA tu dong.
        if mode == "qa":
            answer_key = f"playback_answer_{widget_key}"
            if answer_key not in st.session_state:
                st.session_state[answer_key] = submit_extra.get("answer", "")
            answer = st.text_input("Câu trả lời", key=answer_key)
            submit_row = {"video_id": video_id, "frame_id": cur_frames[0], "answer": answer,
                          "_anchor": _anchor_id(widget_key)}
        else:
            submit_row = {"video_id": video_id, "frame_id": cur_frames[0], "_anchor": _anchor_id(widget_key), **submit_extra}
        _render_submit_button(mode, submit_row, f"playbacksubmit_{widget_key}")


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
    # 2026-08-20 (theo yeu cau nguoi dung: "Quy tắc quan trọng - Format Submit phải phụ thuộc
    # vào loại task đang chọn ... người dùng không phải tự nhập format thủ công") - THEM
    # "Temporal" thanh 1 LOAI RIENG (truoc day chi co "TRAKE") - Temporal va TRAKE dung CHUNG
    # 1 luong tim kiem chuoi-nhieu-moc (dense_temporal.search, giao dien anchors giong het nhau)
    # nhung SINH RA dinh dang nop bai KHAC NHAU - xem _temporal_submit_mode duoi day. Nut Submit
    # (trong Playback lan nut nhanh ben ngoai) gio CHI hien DUNG 1 nut khop voi mode DANG CHON,
    # khong con hien ca 2 nut "Temporal"/"TRAKE" song song bat nguoi dung tu chon nua.
    mode = st.radio(
        "Loại truy vấn",
        ["KIS", "Temporal", "TRAKE", "Q&A"],
    )
    is_temporal = mode in ("Temporal", "TRAKE")
    is_qa = mode == "Q&A"
    # "temporal" (format nhu KIS, 1 frame GIUA) hoac "trake" (format day du N frame) - dung de
    # nut Submit trong Playback/nut nhanh TU DONG chon DUNG 1 dinh dang, khong hoi lai.
    _temporal_submit_mode = "temporal" if mode == "Temporal" else "trake"

    # 2026-08-20 (theo yeu cau nguoi dung, phat hien qua case that GT L26_V194 [4700,5125,5450,
    # 5850] - 2/3 khoang cach that (17s/16s) VUOT nguong cu 15s, khien anchor-chain LOAI HAN cap
    # moc dung, chon nham cum frame sai gan nhau hon - xem dense_temporal.py::MAX_ANCHOR_GAP_
    # SECONDS) - THAY VI sua code moi lan gap case moi, cho nguoi dung TU CHINH ngay tren UI khi
    # dang chay Temporal/TRAKE. Chi hien khi is_temporal (khong lien quan KIS/Q&A).
    max_gap_seconds = dense_temporal.MAX_ANCHOR_GAP_SECONDS
    if is_temporal:
        max_gap_seconds = st.slider(
            "Khoảng cách tối đa giữa 2 mốc liên tiếp (giây)",
            min_value=1.0, max_value=60.0, value=dense_temporal.MAX_ANCHOR_GAP_SECONDS, step=1.0,
            help="RÀNG BUỘC CỨNG (không phải phạt điểm) — mốc sau PHẢI cách mốc trước không quá "
            "N giây, nếu không sẽ bị loại hẳn khỏi xét, dù điểm khớp cao. Đặt QUÁ THẤP có thể "
            "loại oan chuỗi mốc đúng (case thật: 2 mốc cách nhau 17s bị loại khi ngưỡng=15s, hệ "
            "thống chọn nhầm 1 cụm khác sai nhưng gần nhau). Đặt QUÁ CAO dễ ghép nhầm các đoạn "
            "không liên quan cách xa nhau (case thật: ghép nhầm 2 cảnh cách nhau 65s). "
            f"Mặc định {dense_temporal.MAX_ANCHOR_GAP_SECONDS:.0f}s.",
        )

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

    # 2026-08-20 (theo yeu cau nguoi dung: "tích hợp search theo ASR... đảm bảo những câu như
    # ... nặng đến 211kg... mà embedding model thường không thấy") - LOC CUNG theo loi noi
    # (khac audio_mentions tu LLM - do la SOFT boost tu dong, cai nay nguoi dung TU DIEN, giong
    # OCR). Khong co khai niem "vi tri" nhu OCR nen dung 1 o text don gian, KHONG qua canvas.
    asr_text = st.text_input(
        "Lọc theo lời nói (ASR)",
        help="Lọc CỨNG các frame nằm trong đoạn video có lời nói CHỨA cụm từ này (khớp không "
        "phân biệt dấu, không cần đủ câu — vd chỉ cần gõ '211kg' hoặc 'cá nhám' cũng đủ). Dùng "
        "cho các câu hỏi nhắc tới số liệu/tên riêng/chi tiết CHỈ được NÓI RA chứ không hiện chữ "
        "hay nhìn thấy được — mô hình embedding hình ảnh thường không 'thấy' được các chi tiết "
        "này.",
    )

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

    # 2026-08-20 (theo yeu cau nguoi dung: "trước đó mình muốn thêm option cho [LLM phân rã
    # câu]... mặc định là tắt") - dao nguoc lai quyet dinh 2026-08-15 ("bỏ checkbox, LUÔN BẬT")
    # - LLM phan ra cau (extract_entities/plan_query, ~3s/lan goi NIM) gio la TUY CHON, TAT mac
    # dinh. Dung CHUNG 1 checkbox cho ca KIS lan Q&A (khong tach rieng, tranh lech hanh vi) -
    # "mọi option sau đó ràng buộc theo cái này": Region-CLIP rerank (can plan["attributes"]) +
    # ASR audio_mentions soft-boost (can plan["audio_mentions"]) DEU tu dong khong co gi de dung
    # khi tat (plan la STUB rong, xem "_EMPTY_PLAN" duoi day) - khong can sua rieng 2 cho do.
    use_llm_entity_extraction = st.checkbox(
        "Dùng LLM phân rã câu (trích entity/audio mentions/thuộc tính — ~3s/lần gọi NIM)",
        value=False,
        help="TẮT (mặc định): bỏ qua bước gọi LLM này hoàn toàn, nhanh hơn — hard-filter theo "
        "Object/OCR/ASR chỉ còn dựa vào khung bạn TỰ VẼ trên canvas. BẬT: gọi LLM trích entity/ "
        "thuộc tính/lời nhắc âm thanh từ câu query — Region-CLIP rerank và ASR soft-boost tự "
        "động PHỤ THUỘC vào cờ này (tắt LLM thì 2 cái đó cũng không có gì để chạy).",
    )
    # 2026-08-20 (theo yeu cau nguoi dung: "Cho cái này vào sidebar luôn") - CHUYEN checkbox
    # Region-CLIP tu 2 cho rieng (KIS/Q&A, 2 key/gia tri mac dinh khac nhau) VE 1 checkbox DUY
    # NHAT dung chung o sidebar - dat NGAY DUOI checkbox LLM vi PHU THUOC vao no (can
    # plan["attributes"]). disabled khi LLM tat, giu dung y nghia da lam truoc do.
    use_region_clip_rerank = st.checkbox(
        "Region-CLIP rerank theo thuộc tính (màu/quần áo..., SigLIP2)",
        value=False, help=_REGION_CLIP_CHECKBOX_HELP, disabled=not use_llm_entity_extraction,
    )

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
        # 2026-08-20 (theo yeu cau nguoi dung: "Chuyển default frames render thành 100") - khop
        # dung SUBMISSION_TOP_K (online/submission_pipeline.py) - BTC nhan toi da 100 cau tra
        # loi/truy van (R@{1,5,20,50,100}), nen render du 100 mac dinh de "tu dong dien toi da"
        # (xem nut submit hang loat duoi day) co du nguon ma khong can nguoi dung tu keo slider.
        st.session_state.top_k_slider = 100
        st.session_state.top_k_input = 100

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

# Bang nop bai (2026-08-20) - LUON hien o sidebar, khong phu thuoc mode/co vua chay search hay
# khong, de nguoi dung theo doi tien do xuyen suot phien (xem docstring _render_submission_panel).
_render_submission_panel()

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
    # 2026-08-20 (theo yeu cau nguoi dung: "Cho cái này vào sidebar luôn") - checkbox Region-CLIP
    # da CHUYEN LEN sidebar (1 checkbox DUY NHAT dung chung KIS/Q&A, xem gan checkbox "Dùng LLM
    # phân rã câu"), khong con rieng o day nua.


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
    # 2026-08-20 (theo yeu cau nguoi dung: "Dùng LVLM để trả lời giờ là 1 option và để mặc định
    # là không dùng") - TAT mac dinh (value=False) - khong goi VQA that (ton phi/cham) tru khi
    # nguoi dung CHU DONG bat. Khi tat, "answer" tra ve RONG cho MOI dong (xem submission_
    # pipeline.answer_qa) - nguoi dung tu go cau tra loi trong Playback truoc khi nop.
    st.checkbox(
        "Dùng LVLM tự động trả lời (tốn phí/lần gọi API — mặc định TẮT, tự gõ đáp án trong Playback)",
        value=False, key="qa_use_lvlm",
    )
    st.number_input(
        "Số ứng viên đầu gọi VQA thật (tốn phí/lần)", min_value=1, max_value=20, value=3, step=1,
        key="qa_vqa_top_n", disabled=not st.session_state.get("qa_use_lvlm", False),
    )
    # 2026-08-20: checkbox Region-CLIP da CHUYEN LEN sidebar (dung chung KIS/Q&A) - xem
    # _render_kis_query_inline cho ghi chu day du.


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
    # 2026-08-20: checkbox Region-CLIP CHUYEN LEN sidebar (dung chung KIS/Q&A) - `use_region_
    # clip_rerank` gio la bien module-level tu sidebar, KHONG con doc rieng qua session_state
    # nua (khong gan de lai o day nua, tranh SHADOW nham bien sidebar).
    use_lvlm = st.session_state.get("qa_use_lvlm", False)
    anchors_raw = ""
else:
    # 2026-08-17: o + checkbox rendered NGANG HANG voi danh sach khung ve (xem
    # _render_kis_query_inline, goi tu ben trong _render_filter_canvas o tren) - CHI doc lai
    # gia tri qua session_state o day, KHONG goi widget lan 2 (se tao trung key/nhan doi).
    query = st.session_state.get("kis_query_input", "")
    anchors_raw = ""
    question = ""

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
    _asr_text = asr_text.strip() or None  # "" -> None, giong ocr_text (khong loc)

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
        asr_text=_asr_text,
    )

    log = StepLog()
    t_start = time.perf_counter()
    plan = None
    with st.spinner("Đang tìm..." if not is_qa else "Đang tìm + hỏi VQA (có thể mất chút thời gian)..."):
      # 2026-08-20 (theo yeu cau nguoi dung: "Bên NVIDIA NIM đã xóa đi hiển thị credit... viết
      # exception trả về lỗi trên UI khi LLM quá 60s mà không trả về kết quả" - phat hien qua
      # case that distill_query() TREO VO HAN vi OpenAI client khong dat timeout) - BOC CA 3
      # nhanh (Q&A/TRAKE/KIS) trong 1 try/except DUY NHAT: ca 3 DEU co the goi NIM (distill_
      # query() luon chay ben trong search_dense/_encode_query bat ke nhanh nao, extract_
      # entities/VQA rieng cho Q&A/KIS) - NIMTimeoutError (xem app_flags.py, cac client NIM gio
      # DA dat timeout=60s) duoc bat RIENG, hien st.error() RO RANG + dung script (st.stop())
      # thay vi de Streamlit tu roi vao spinner treo vo han hoac in traceback tho kho hieu.
      try:
        if is_qa:
            # 2026-08-15: truyen THEM spatial_boxes/spatial_op (khung ve OCR/Object da nang,
            # AND/OR, soft-boost vi tri) - truoc do BI THIEU (common_filters chi mang ocr_text
            # tu khung OCR DAU TIEN kieu cu, khong mang het cac khung tren canvas) - phat hien
            # qua cau hoi nguoi dung "Q&A co dung duoc khung ve khong".
            results = answer_qa(query, question, top_k=top_k, vqa_top_n=int(vqa_top_n),
                                 dense_model=dense_model, use_region_clip_rerank=use_region_clip_rerank,
                                 use_lvlm=use_lvlm, use_llm_entity=use_llm_entity_extraction,
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
                asr_text=_asr_text, score_algorithm=score_algorithm, distill_model=distill_model,
                max_gap_seconds=max_gap_seconds, log=log,
            )
        else:
            # 2026-08-20 (theo yeu cau nguoi dung: "trước đó mình muốn thêm option cho [LLM
            # phân rã câu]... mặc định là tắt" - DAO NGUOC quyet dinh 2026-08-15 "bỏ checkbox,
            # LUÔN BẬT") - CHI goi extract_entities() (LLM that, ~3s + phi API) khi checkbox
            # "Dùng LLM phân rã câu" o sidebar DUOC BAT. TAT (mac dinh) -> dung STUB rong, cac
            # buoc phu thuoc (Region-CLIP rerank/ASR audio_mentions) tu nhien khong co gi de
            # chay - xem ghi chu tai noi dinh nghia checkbox.
            if use_llm_entity_extraction:
                plan = extract_entities(query, log=log)
            else:
                plan = {
                    "entities": [], "secondary_entities": [], "attributes": [],
                    "audio_mentions": [], "clip_text": query, "unresolved": [],
                    "resolved_must_have_labels": [], "resolved_min_count": {},
                }
                if log:
                    log.add("LLM phân rã câu (NIM)", "TẮT (checkbox người dùng) — bỏ qua hoàn toàn", 0.0)
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
                asr_text=_asr_text, spatial_boxes=spatial_boxes or None, spatial_op=spatial_op,
                audio_mentions=plan.get("audio_mentions") or None, ocr_algorithm=ocr_algorithm,
                score_algorithm=score_algorithm, multi_clause=multi_clause,
                distill_model=distill_model, log=log,
            )
            if attributes and not results.empty:
                results = apply_region_clip_rerank(results, attributes, top_k, log=log)
            elif log and plan.get("attributes") and not use_region_clip_rerank:
                log.add("Region-CLIP", "TẮT (checkbox người dùng) — bỏ qua bước rerank theo thuộc tính", 0.0)
      except (NIMTimeoutError, ModalTimeoutError) as e:
        st.error(f"⏱ {e}")
        st.stop()
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
        "use_lvlm": use_lvlm if is_qa else None,
        "temporal_submit_mode": _temporal_submit_mode if is_temporal else None,
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
    _r_use_lvlm = _snap.get("use_lvlm")
    _r_temporal_submit_mode = _snap.get("temporal_submit_mode", "trake")

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
        # 2026-08-20 (theo yeu cau nguoi dung, phat hien qua screenshot that: cong tay cac dong
        # log ra ~6.5s nhung "Tổng thời gian xử lý" tren dau lai la 4.65s) - BUG THAT: 1 so buoc
        # (vd "Encode query + xếp hạng" goi TU BEN TRONG buoc "Bộ lọc cứng — bù thêm...") vua
        # duoc ghi la 1 dong RIENG, vua nam TRONG khoang thoi gian cua buoc cha -> cong tay se
        # dem 2 LAN. Fix: StepLog.timed() gio danh dau "nested" (xem steplog.py) - o day THUT
        # VAO + ghi chu ro cac dong nested, va hien THEM 1 dong TONG (chi cong buoc top-level,
        # KHOP voi "Tổng thời gian xử lý" o tren) de nguoi dung khong con nham khi tu cong tay.
        for i, s in enumerate(_r_log_steps, 1):
            if s.get("nested"):
                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;↳ **{i}. {s['step']}** — `{s['elapsed_s']}s` "
                            f"_(đã tính trong bước cha ở trên, KHÔNG cộng thêm vào tổng)_  \n{s['detail']}")
            else:
                st.markdown(f"**{i}. {s['step']}** — `{s['elapsed_s']}s`  \n{s['detail']}")
        _top_level_sum = sum(s["elapsed_s"] for s in _r_log_steps if not s.get("nested"))
        st.caption(f"Σ Tổng các bước KHÔNG lồng nhau: {_top_level_sum:.2f}s "
                   f"(khớp với \"⏱ Tổng thời gian xử lý\" ở trên, sai số nhỏ do làm tròn/phần "
                   f"render ngoài log)")

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
        if _r_use_lvlm:
            st.caption(f"{len(results)} kết quả — chỉ {min(int(_r_vqa_top_n), len(results))} ứng viên đầu được hỏi VQA thật, "
                       f"các ứng viên sau dùng lại câu trả lời của ứng viên đầu")
        else:
            st.caption(f"{len(results)} kết quả — LVLM đang TẮT (mặc định), chưa có câu trả lời tự động — "
                       f"mở 🎬 Playback ở từng frame để tự gõ câu trả lời trước khi nộp.")
        _render_autofill_button("qa", results)
        cols = st.columns(4)
        for i, row in results.iterrows():
            col = cols[i % 4]
            with col:
                _qa_key = f"qa_{i}"
                _render_scroll_anchor(_qa_key)
                _default_frame_id = int(row["frame_id"])
                # 2026-08-20 (theo yeu cau nguoi dung: "frame được bấm chức năng playback sẽ
                # được thay đổi bên ngoài luôn") - xem ghi chu day du o nhanh KIS phia tren.
                _eff_frame_id = _playback_confirmed_frame_id(_qa_key, _default_frame_id)
                _eff_path = _frame_preview_path(row["video_id"], _eff_frame_id, _default_frame_id, row["path"])
                _render_image_with_vlm_overlay(_eff_path, _qa_key)
                backfill_tag = " 🔓" if row.get("is_backfill") else ""
                edited_tag = " ✏️" if _eff_frame_id != _default_frame_id else ""
                if _r_use_lvlm:
                    vqa_tag = "🔎 VQA thật" if i < _r_vqa_top_n else "↪ dùng lại rank 1"
                    st.markdown(
                        f"**{row['video_id']}**{backfill_tag}{edited_tag} · frame `{_eff_frame_id}` ({vqa_tag})  \n"
                        f"**Trả lời:** {row['answer']}"
                    )
                else:
                    st.markdown(f"**{row['video_id']}**{backfill_tag}{edited_tag} · frame `{_eff_frame_id}`")
                # 2026-08-20 (theo yeu cau nguoi dung: "cho mấy cái nút này nằm trên 1 hàng...
                # chỉ cần có các icon") - st.container(horizontal=True) (KHONG phai st.columns,
                # xem ghi chu TRAKE anchor row cu ve ly do chon API nay) giu CA 4 nut TREN 1
                # HANG, khong bi wrap xuong dong nhu columns hep.
                with st.container(horizontal=True, gap="small"):
                    _render_video_toggle(row["video_id"], _eff_frame_id, _qa_key)
                    _render_playback(
                        row["video_id"], [_default_frame_id], ["Frame"], _qa_key, "qa",
                        submit_extra={"answer": row["answer"]},
                    )
                    _render_vlm_ocr_verify(_eff_path, row["video_id"], _eff_frame_id, _qa_key)
                    _render_qa_submit_toggle(row["video_id"], _eff_frame_id, _qa_key)
                _render_qa_submit_panel(row["video_id"], _eff_frame_id, row["answer"], _qa_key)
    elif _r_is_temporal:
        _r_mode_label_top = "Temporal" if _r_temporal_submit_mode == "temporal" else "TRAKE"
        st.caption(f"{len(results)} video khớp chuỗi {len(_r_anchors)} anchor (đúng thứ tự thời gian) — "
                   f"loại truy vấn: **{_r_mode_label_top}** (định dạng nộp bài tự động theo loại này)")
        _render_autofill_button(_r_temporal_submit_mode, results, n_anchors=len(_r_anchors))
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
            # 2026-08-20 (theo yeu cau nguoi dung: "frame được bấm chức năng playback sẽ được
            # thay đổi bên ngoài luôn") - tinh _trake_chain_key TU FRAME GOC (on dinh, khong doi
            # theo chinh sua) TRUOC vong lap moc, doc lai chuoi DA XAC NHAN qua playback_state
            # (CUNG nguon voi _render_playback ben duoi - KHONG con dung trake_frame_overrides,
            # xem _overrides ben duoi CHI con la fallback tuong thich cho _render_trake_frame_
            # tune cu, hien khong con noi nao goi toi nhung giu de khong gay loi neu bat lai sau).
            _default_chain_frame_ids = [int(row[f"anchor{i}_frame_id"]) for i in range(len(_r_anchors))]
            _trake_chain_key = f"trakechain_{row['video_id']}_{'_'.join(map(str, _default_chain_frame_ids))}"
            _render_scroll_anchor(_trake_chain_key)
            _confirmed_chain = st.session_state.get(f"playback_state_{_trake_chain_key}", _default_chain_frame_ids)
            _overrides = st.session_state.get("trake_frame_overrides", {})
            _chain_frame_ids = []  # gom lai frame_id (DA tinh chinh neu co) cua CA chuoi, dung
            # cho nut "Nộp" cap VIDEO ben duoi (1 dong nop bai = 1 video + N frame_id, khong
            # phai 1 dong/moc).
            for i, anchor in enumerate(_r_anchors):
                with cols[i]:
                    _default_frame_id = _default_chain_frame_ids[i]
                    _ov = _overrides.get((row["video_id"], i))
                    if _confirmed_chain[i] != _default_frame_id:
                        # DA xac nhan frame moi trong Playback (uu tien cao nhat - moi nhat).
                        _frame_id = int(_confirmed_chain[i])
                        _pts_time = _frame_id / _fps_by_video().get(row["video_id"], 25.0)
                        _img_path = _frame_preview_path(
                            row["video_id"], _frame_id, _default_frame_id, row[f"anchor{i}_path"]
                        )
                        _edited = True
                    elif _ov:
                        # fallback cu (_render_trake_frame_tune) - xem ghi chu tren.
                        _frame_id, _pts_time, _img_path, _edited = _ov["frame_id"], _ov["pts_time"], _ov["path"], True
                    else:
                        _frame_id = _default_frame_id
                        _pts_time = float(row[f"anchor{i}_pts_time"])
                        _img_path = row[f"anchor{i}_path"]
                        _edited = False
                    _chain_frame_ids.append(_frame_id)

                    _trake_widget_key = f"trake_{row['video_id']}_{i}_{_default_frame_id}"
                    # 2026-08-15 (migrate sang bo dense): doc anh THANG tu row[f"anchor{i}_path"]
                    # (dia local), khong qua Keyframes_*.zip cua BTC nua (xem tiers/dense_temporal.py).
                    _render_image_with_vlm_overlay(_img_path, _trake_widget_key)
                    extra = []
                    if anchor.get("must_have_labels"):
                        extra.append(f"nhãn: {anchor['must_have_labels']}")
                    if anchor.get("ocr_text"):
                        extra.append(f"OCR: \"{anchor['ocr_text']}\"")
                    if _edited:
                        extra.append("✅ đã tinh chỉnh thủ công")
                    extra_str = f"  \n_{' · '.join(extra)}_" if extra else ""
                    st.caption(
                        f"{anchor['text']}{extra_str}  \n"
                        f"frame `{_frame_id}` · t={_pts_time:.2f}s"
                    )
                    with st.container(horizontal=True, gap="small"):
                        _render_video_toggle(row["video_id"], _frame_id, _trake_widget_key, fixed_window=True)
                        _render_vlm_ocr_verify(_img_path, row["video_id"], _frame_id, _trake_widget_key)

            # 2026-08-20 (theo yeu cau nguoi dung: "task cần nhiều frame như TRAKE... Playback
            # hiển thị các mốc frame do hệ thống đề xuất và cho phép người dùng chỉnh lại chính
            # xác trước khi submit") - 1 nut "🎬 Playback" DUY NHAT o CAP VIDEO (khong phai tung
            # moc rieng) - hien CA CHUOI tren 1 timeline, chinh + nop trong CUNG 1 cho (thay the
            # 2 nut rieng 🎬/🎯 tung moc truoc day). Nut nop nhanh ben ngoai (frame he thong de
            # xuat, KHONG qua chinh) van giu song song cho truong hop khong can chinh gi.
            # (_trake_chain_key da tinh TRUOC vong lap moc o tren - dung LAI, KHONG tinh lai tu
            # _chain_frame_ids da chinh, tranh key TU DOI theo gia tri no dang dung de doc).
            with st.container(horizontal=True, gap="small"):
                # 2026-08-20 (theo yeu cau nguoi dung: "Format Submit phải phụ thuộc vào loại
                # task đang chọn") - `_r_temporal_submit_mode` ("temporal"/"trake") lay TU MODE
                # DANG CHON o sidebar (khong con hoi lai/hien 2 nut de nguoi dung tu chon).
                _render_playback(
                    row["video_id"], _chain_frame_ids, [f"Mốc {i + 1}" for i in range(len(_r_anchors))],
                    _trake_chain_key, _r_temporal_submit_mode,
                )
                _r_mode_label = "Temporal" if _r_temporal_submit_mode == "temporal" else "TRAKE"
                st.caption(f"👆 Xem/chỉnh trong Playback, hoặc nộp nhanh frame đề xuất (loại **{_r_mode_label}**) →")
                if _r_temporal_submit_mode == "temporal":
                    _render_submit_button(
                        "temporal", {"video_id": row["video_id"], "frame_id": int(statistics.median(_chain_frame_ids)),
                                     "_anchor": _anchor_id(_trake_chain_key)},
                        _trake_chain_key, label="Temporal",
                    )
                else:
                    _render_submit_button(
                        "trake", {"video_id": row["video_id"], "frame_ids": _chain_frame_ids,
                                  "_anchor": _anchor_id(_trake_chain_key)},
                        _trake_chain_key, label="TRAKE",
                    )
            st.divider()
    else:
        # duong mac dinh (2026-08-15, cutover): search_dense() - cot path/frame_id (xem
        # tiers/dense_search.py), anh nam THANG tren dia local, khong qua Keyframes_*.zip.
        st.caption(f"{len(results)} kết quả — model: {_r_dense_model}")
        _render_autofill_button("kis", results)
        cols = st.columns(4)
        for i, row in results.iterrows():
            col = cols[i % 4]
            with col:
                _kis_key = f"kis_{i}"
                _render_scroll_anchor(_kis_key)
                _default_frame_id = int(row["frame_id"])
                # 2026-08-20 (theo yeu cau nguoi dung: "frame được bấm chức năng playback sẽ
                # được thay đổi bên ngoài luôn") - doc lai frame DA XAC NHAN (neu co) thay vi
                # luon dung frame goc he thong de xuat - anh/caption/nut Nộp nhanh BEN NGOAI
                # Playback gio DONG BO theo dung gia tri da chot trong popup.
                _eff_frame_id = _playback_confirmed_frame_id(_kis_key, _default_frame_id)
                _eff_path = _frame_preview_path(row["video_id"], _eff_frame_id, _default_frame_id, row["path"])
                _render_image_with_vlm_overlay(_eff_path, _kis_key)
                # 🔓 = ket qua BU THEM (khong khop bo loc cung, xem is_backfill o dense_search.py)
                backfill_tag = " 🔓" if row.get("is_backfill") else ""
                edited_tag = " ✏️" if _eff_frame_id != _default_frame_id else ""
                st.markdown(
                    f"**{row['video_id']}**{backfill_tag}{edited_tag} · frame `{_eff_frame_id}` · score={row['score']:.3f}"
                )
                with st.container(horizontal=True, gap="small"):
                    _render_video_toggle(row["video_id"], _eff_frame_id, _kis_key)
                    _render_playback(row["video_id"], [_default_frame_id], ["Frame"], _kis_key, "kis")
                    _render_vlm_ocr_verify(_eff_path, row["video_id"], _eff_frame_id, _kis_key)
                    _render_submit_button(
                        "kis", {"video_id": row["video_id"], "frame_id": _eff_frame_id, "_anchor": _anchor_id(_kis_key)},
                        _kis_key,
                    )

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
