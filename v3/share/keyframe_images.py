"""Đọc ảnh keyframe trực tiếp từ file zip (không giải nén toàn bộ ~30GB ra đĩa).

Quy ước đã kiểm chứng trên data thật:
  - Zip: Keyframes_<prefix>.zip, riêng L26 tách 5 file con theo khoảng V (a: V001-099,
    b: V100-199, c: V200-299, d: V300-399, e: V400-499).
  - Trong zip: keyframes/<video_id>/<n:03d>.jpg, với n = local_idx + 1 (n khớp 1-1 thứ tự
    hàng trong map-keyframes/<video_id>.csv, đã verify không lệch).
"""
from __future__ import annotations

import zipfile

from config import DATA_ROOT

_zip_cache: dict[str, zipfile.ZipFile] = {}


def _zip_filename_for(video_id: str) -> str:
    prefix = video_id.split("_")[0]  # vd "L21", "L26"
    if prefix == "L26":
        vnum = int(video_id.split("_V")[1])
        if vnum <= 99:
            suf = "a"
        elif vnum <= 199:
            suf = "b"
        elif vnum <= 299:
            suf = "c"
        elif vnum <= 399:
            suf = "d"
        else:
            suf = "e"
        return f"Keyframes_L26_{suf}.zip"
    return f"Keyframes_{prefix}.zip"


def _get_zip(video_id: str) -> zipfile.ZipFile:
    fn = _zip_filename_for(video_id)
    if fn not in _zip_cache:
        _zip_cache[fn] = zipfile.ZipFile(DATA_ROOT / fn)
    return _zip_cache[fn]


def read_keyframe_bytes(video_id: str, local_idx: int) -> bytes:
    """local_idx: 0-based, khớp cột local_idx trong meta.parquet (n = local_idx + 1)."""
    z = _get_zip(video_id)
    n = local_idx + 1
    inner_path = f"keyframes/{video_id}/{n:03d}.jpg"
    return z.read(inner_path)
