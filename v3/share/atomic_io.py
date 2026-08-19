"""Ghi file atomic - tranh hong file khi tien trinh bi kill DUNG LUC dang ghi.

Bai hoc 2026-08-14: build_dense_objects_index.py ghi thang df.to_parquet(OUT_PATH) trong vong
lap checkpoint - tien trinh bi kill giua chung ghi lam file MAT HAN footer (Parquet doc tu cuoi
len, can footer de biet vi tri row group), khong the phuc hoi bang bat ky cong cu nao, xoa sach
~122k anh (hang gio GPU) da xu ly du checkpoint/resume o muc log/done-file van tuong la xong.

Sua bang pattern chuan: ghi vao file .tmp CUNG THU MUC roi os.replace() - la thao tac atomic
cua he dieu hanh (khong bao gio co trang thai "nua ghi"), ap dung cho MOI script co checkpoint
dinh ky trong vong lap dai (ASR/OCR/DINO/objects) - noi rui ro bi kill giua chung la cao nhat.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def atomic_write_parquet(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """Thay the truc tiep cho df.to_parquet(path, ...) - an toan khi bi kill giua chung."""
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp_path, **kwargs)
    os.replace(tmp_path, path)
