"""Đọc video gốc từ zip BTC cấp (Videos_L*.zip — KHÁC hẳn Keyframes_*.zip, chỉ dùng cho ASR,
chưa từng đụng tới trước đây vì mọi tầng khác chỉ cần keyframe ảnh) + trích audio bằng ffmpeg
local TRƯỚC khi gửi Modal (audio nhẹ hơn video gốc rất nhiều — gửi audio thay vì cả file mp4
tiết kiệm băng thông đáng kể khi chạy hàng trăm video).

Quy ước zip đã kiểm chứng trên data thật (khác Keyframes_*.zip ở chỗ MỌI prefix đều có hậu tố
_a, không chỉ riêng L26 mới tách):
  - Videos_<prefix>_a.zip, riêng L26 tách 5 file con giống Keyframes (a..e theo khoảng V).
  - Trong zip: video/<video_id>.mp4
"""
from __future__ import annotations

import subprocess
import tempfile
import zipfile
from pathlib import Path

from config import DATA_ROOT

_zip_cache: dict[str, zipfile.ZipFile] = {}


def _video_zip_filename_for(video_id: str) -> str:
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
        return f"Videos_L26_{suf}.zip"
    return f"Videos_{prefix}_a.zip"


def _get_zip(video_id: str) -> zipfile.ZipFile:
    fn = _video_zip_filename_for(video_id)
    if fn not in _zip_cache:
        _zip_cache[fn] = zipfile.ZipFile(DATA_ROOT / fn)
    return _zip_cache[fn]


def read_video_bytes(video_id: str) -> bytes:
    z = _get_zip(video_id)
    return z.read(f"video/{video_id}.mp4")


def extract_audio_bytes(video_id: str, sample_rate: int = 16000, fmt: str = "mp3") -> bytes:
    """Trich audio mono 16kHz bang ffmpeg LOCAL - tranh gui nguyen video (nang hon audio rat
    nhieu) len Modal. ffmpeg khong doc duoc tu memory truc tiep cho input MP4 (container
    format can seek) -> phai ghi tam ra file, KHONG the pipe stdin thang cho input nay (khac
    output co the pipe stdout duoc).

    fmt="mp3" (2026-08-11, doi tu "wav" mac dinh - do that: nen MP3 nhe hon WAV ~5.7x (37MB ->
    6.6MB cho 1 video ~19 phut) nhung wall-clock chi nhanh hon ~5% (169.8s -> 160.8s) - KHONG
    phai nut that chinh (do that: bien thien tu nhien cua GPU compute lon hon nhieu, ~20-25%
    qua nhieu lan goi cung 1 audio) nhung van la loi ich MIEN PHI, khong rui ro (transformers
    ASR pipeline tu giai ma MP3 qua ffmpeg noi bo, khong can sua ben Modal)."""
    video_bytes = read_video_bytes(video_id)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
        tmp_in.write(video_bytes)
        tmp_in_path = tmp_in.name

    try:
        if fmt == "mp3":
            cmd = [
                "ffmpeg", "-y", "-i", tmp_in_path,
                "-vn", "-ac", "1", "-ar", str(sample_rate),
                "-codec:a", "libmp3lame", "-qscale:a", "4",
                "-f", "mp3", "pipe:1",
            ]
        else:
            cmd = [
                "ffmpeg", "-y", "-i", tmp_in_path,
                "-vn", "-ac", "1", "-ar", str(sample_rate),
                "-f", "wav", "pipe:1",
            ]
        result = subprocess.run(cmd, capture_output=True, check=True)
        return result.stdout
    finally:
        Path(tmp_in_path).unlink(missing_ok=True)
