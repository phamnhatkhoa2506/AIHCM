"""Cac co (flag) tam thoi dung CHUNG giua online/app.py va online/submission_pipeline.py - tach
rieng file nay de 2 noi luon doc CUNG 1 gia tri (tranh sua 1 cho quen sua cho kia, lech hanh vi
giua KIS/Q&A)."""
from __future__ import annotations

# 2026-08-17 (GAC LAI de TEST, theo yeu cau nguoi dung sau bug that): LLM query_planner.plan_
# query() phan ra entity KHONG ON DINH giua cac lan goi CUNG 1 cau (khong sua duoc bang cache -
# de thi that moi cau chi hoi DUNG 1 LAN, hen xui van fail dung lan do) - co lan TU BIA THEM
# entity khong co trong cau goc (vd cau ve "chiếc xuồng gỗ..." khong nhac gi toi thung/vai, LLM
# van bia ra 1 entity resolve nham thanh nhan "Barrel"), bien thanh HARD FILTER SAI, bop hep sai
# corpus (vd con 4675/369589 frame khong lien quan) - lam sai toan bo ket qua ma nguoi dung
# khong he thay dau hieu gi bat thuong (query/UI y het lan dung).
#
# True = TAT hard-filter tu LLM entity (chi con giu filter tu khung Object VE TAY tren canvas -
# dang tin cay hon nhieu vi la y dinh THAT cua nguoi dung, khong phai LLM doan/bia). Doi ve
# False de bat lai NHU CU khi da co huong sua tan goc (vd tang do nghiem ngat SYSTEM_PROMPT cua
# query_planner.py, hoac co benchmark that de kiem chung truoc/sau) va muon so sanh lai.
DISABLE_LLM_ENTITY_HARD_FILTER = True

# 2026-08-20 (theo yeu cau nguoi dung: "Bên NVIDIA NIM đã xóa đi hiển thị credit nên không biết
# là có hết credit hay không. Bạn hãy viết exception trả về lỗi trên UI khi LLM quá 60s mà
# không trả về kết quả" - phat hien qua case that: distill_query() TREO VO HAN, khong loi
# khong ket qua, vi OpenAI client KHONG dat timeout - request cu the o do "cho mai" tren httpx
# mac dinh) - dung CHUNG 1 nguong + 1 loai exception cho CA 3 noi goi NIM (query_distill.py,
# query_planner.py, submission_pipeline.py) de UI (app.py) bat DUOC 1 loai duy nhat, hien loi
# RO RANG thay vi Streamlit tu roi vao spinner treo vo han hoac in traceback tho.
NIM_TIMEOUT_SECONDS = 60.0


class NIMTimeoutError(RuntimeError):
    """NVIDIA NIM khong phan hoi trong NIM_TIMEOUT_SECONDS giay - co the het credit (tai khoan
    NIM da AN thong tin credit, khong con thay truoc duoc), rate-limit, hoac dich vu dang gap
    su co. Cac noi goi NIM (.chat.completions.create) PHAI bat openai.APITimeoutError va raise
    lai thanh loai nay, thay vi de httpx tu treo vo han hoac nuot loi im lang."""

    def __init__(self, context: str) -> None:
        super().__init__(
            f"NVIDIA NIM không phản hồi sau {NIM_TIMEOUT_SECONDS:.0f}s ({context}) — có thể đã "
            f"hết credit (tài khoản NIM hiện ẩn thông tin credit, không kiểm tra trước được) "
            f"hoặc dịch vụ đang gặp sự cố. Kiểm tra tại https://build.nvidia.com hoặc thử lại sau."
        )


# 2026-08-20 (theo yeu cau nguoi dung, tiep tuc SAU KHI da them timeout cho NIM: "hơn 1 phút
# nhưng chưa hết exception nào được raise" - xac nhan nguoi dung DA restart Streamlit han hoi,
# nghia la diem treo THAT SU KHONG PHAI NIM (da co timeout 60s) ma la cac cuoc goi MODAL
# (.remote() toi query-encoder/dense-index/region-rerank) - CHUA TUNG dat timeout nao ca. Them
# 1 nguong + exception TUONG TU cho Modal, dung CHUNG bang helper _call_modal_with_timeout
# (xem dense_search.py/local_text_encoders.py).
MODAL_TIMEOUT_SECONDS = 60.0


class ModalTimeoutError(RuntimeError):
    """Modal (.remote()) khong phan hoi trong MODAL_TIMEOUT_SECONDS giay - container co the
    dang cold-start qua lau, mang loi, hoac app chua deploy/da bi dung. Cac noi goi Modal PHAI
    dung _call_modal_with_timeout() thay vi goi .remote() truc tiep, de KHONG treo vo han."""

    def __init__(self, context: str) -> None:
        super().__init__(
            f"Modal không phản hồi sau {MODAL_TIMEOUT_SECONDS:.0f}s ({context}) — container có "
            f"thể đang cold-start quá lâu, mạng lỗi, hoặc app chưa deploy/đã bị dừng. Kiểm tra "
            f"bằng `modal container list` / `modal app list`, hoặc thử lại sau."
        )


# 2026-08-20 (theo yeu cau nguoi dung, phat hien qua bug that: "chạy nhiều lần cho hết cold-
# start nhưng pe_core vẫn 6s" - da them @modal.concurrent cho 2 app (query_encoders_app.py/
# dense_index_app.py) NHUNG van con cham khi 2 model goi DONG THOI qua ThreadPoolExecutor
# (_rank_rrf) - co lap lai bang isolated test: fn.spawn()+call.get(timeout=...) (cach CU o day)
# CHAM HAN so voi fn.remote() truc tiep, ĐẶC BIỆT khi nhieu thread goi cung luc (spawn tao them
# 1 FunctionCall object + 1 vong poll rieng, ganh chi phi gap ~2-4x remote() thang - do that:
# pe_core 5.9s qua spawn+get vs 1.4s qua remote() thang, CUNG dieu kien concurrent). Doi sang
# fn.remote() THAT (nhanh, dung API sync goc) chay trong 1 thread rieng (ThreadPoolExecutor 1
# worker) + .result(timeout=...) o phia CLIENT - dat duoc CUNG hieu ung "khong treo qua
# `timeout` giay" ma KHONG cham nhu spawn+get (do that: gan bang remote() truc tiep). Luu y:
# cach nay KHONG huy duoc execution phia Modal khi het han (fn.remote() van chay tiep tren
# server, chi client thoi cho) - chap nhan duoc vi muc tieu la UI KHONG treo, khong phai huy
# tac vu.
def call_modal_with_timeout(fn, /, *args, context: str, timeout: float = MODAL_TIMEOUT_SECONDS, **kwargs):
    """Goi 1 Modal Function/Method (`fn`, vd `enc.encode_siglip_text`) VOI TIMEOUT o phia CLIENT
    - dung fn.remote() (dong bo, nhanh) trong 1 thread rieng, .result(timeout=...) tren Future
    de KHONG cho qua `timeout` giay. Nem ModalTimeoutError neu qua han."""
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as _FutureTimeoutError

    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(fn.remote, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except _FutureTimeoutError as e:
            raise ModalTimeoutError(context) from e
