"""Encode TEXT (cau truy van) cho 3 model dense (SigLIP2/PE-Core/BEiT-3) - qua Modal app
NHE luon giu am (aic2026-query-encoders, min_containers=1, xem
offline/modal_infra/query_encoders_app.py).

SUA 2026-08-15 (theo yeu cau nguoi dung: "host nhe 3 mo hinh embedding do do chay tren may
minh cho do lag") - TRUOC DAY module nay load ca 3 model TRUC TIEP tren may local (torch CPU +
clone repo GitHub PE-Core/BEiT-3 luc dau) - nang may, gay lag ro ret khi encode query. GIO
GOI .remote() toi 1 Modal app CPU-only nho, LUON GIU AM (min_containers=1 - khong co van de
cold-start ma tung tranh khi thiet ke ban local truoc do, vi container khong bao gio tat).

Giu NGUYEN interface cu (ENCODERS dict, tra ve np.ndarray shape (1, dim)) - dense_search.py
KHONG can sua gi (van goi ENCODERS[model](query) y het truoc)."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

MODAL_APP_NAME = "aic2026-query-encoders"
MODAL_CLASS_NAME = "QueryEncoder"


@lru_cache(maxsize=1)
def _encoder():
    import modal

    Encoder = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)
    return Encoder()


def encode_text_siglip(text: str) -> np.ndarray:
    vec = _encoder().encode_siglip_text.remote(text)
    return np.asarray([vec], dtype=np.float32)


def encode_text_pe_core(text: str) -> np.ndarray:
    vec = _encoder().encode_pe_core_text.remote(text)
    return np.asarray([vec], dtype=np.float32)


def encode_text_beit3(text: str) -> np.ndarray:
    vec = _encoder().encode_beit3_text.remote(text)
    return np.asarray([vec], dtype=np.float32)


ENCODERS = {
    "siglip": encode_text_siglip,
    "pe_core": encode_text_pe_core,
    "beit3": encode_text_beit3,
}
