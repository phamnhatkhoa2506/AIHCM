"""Cac co (flag) tam thoi dung CHUNG giua online/app.py va online/submission_pipeline.py - tach
rieng file nay de 2 noi luon doc CUNG 1 gia tri (tranh sua 1 cho quen sua cho kia, lech hanh vi
giua KIS/Q&A)."""
from __future__ import annotations


DISABLE_LLM_ENTITY_HARD_FILTER = True
NIM_TIMEOUT_SECONDS = 60.0
MODAL_TIMEOUT_SECONDS = 60.0


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


class ModalUnavailableError(RuntimeError):
    """Modal tra loi LOI (KHAC voi treo qua han - xem ModalTimeoutError) - vd chua dang nhap/
    token sai (AuthError), app chua duoc deploy, workspace het spend limit, mat mang... Nem ra
    THONG BAO RO RANG kem huong dan 2 lua chon: sua phia Modal, HOAC bat bien moi truong local
    tuong ung (neu co, xem local_env_hint) de chay HOAN TOAN khong can Modal."""

    def __init__(self, context: str, original: Exception, local_env_hint: str | None = None) -> None:
        hint = (
            f" Hoặc đặt biến môi trường {local_env_hint}=1 trong .env để chạy HOÀN TOÀN KHÔNG "
            f"CẦN Modal cho phần này (xem README.md mục 'Chạy model local')."
            if local_env_hint else ""
        )
        super().__init__(
            f"Không gọi được Modal ({context}) — {type(original).__name__}: {original}. "
            f"Kiểm tra: đã `modal deploy` app tương ứng chưa, `modal token` còn hợp lệ không "
            f"(`modal profile current`), workspace còn hạn mức chi tiêu không.{hint}"
        )


def call_modal_with_timeout(
    fn, /, *args, context: str, timeout: float = MODAL_TIMEOUT_SECONDS,
    local_env_hint: str | None = None, **kwargs,
):
    """Goi 1 Modal Function/Method (`fn`, vd `enc.encode_siglip_text`) VOI TIMEOUT o phia CLIENT
    - dung fn.remote() (dong bo, nhanh) trong 1 thread rieng, .result(timeout=...) tren Future
    de KHONG cho qua `timeout` giay. Nem ModalTimeoutError neu qua han, ModalUnavailableError
    neu Modal loi vi ly do KHAC (auth/chua deploy/het spend limit/mat mang...) - xem ghi chu o
    tren. local_env_hint (optional): ten bien AIC_LOCAL_* tuong ung de goi y trong thong bao
    loi (vd "AIC_LOCAL_QUERY_ENCODER_SIGLIP") - None neu khong co lua chon local (hiem)."""
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as _FutureTimeoutError

    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(fn.remote, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except _FutureTimeoutError as e:
            raise ModalTimeoutError(context) from e
        except Exception as e:
            raise ModalUnavailableError(context, e, local_env_hint) from e
