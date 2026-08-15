"""Trích 1 đoạn video NGẮN quanh 1 keyframe để hiển thị PHÁT ĐƯỢC trên UI (2026-08-15, theo
yêu cầu người dùng "keyframe hiển thị có thể chạy video" — BTC chấm điểm UI cao hơn nếu kết
quả không chỉ là ảnh tĩnh).

KHÔNG load nguyên video vào RAM mỗi lần xem — ghi nguyên file mp4 ra CACHE ĐĨA cục bộ 1 lần/
video (dùng lại y hệt pattern extract_audio_bytes ở video_audio.py, chỉ khác đích là file cache
lâu dài chứ không phải tempfile xoá ngay), rồi dùng `ffmpeg -ss <t> -i <file> -t <dur> -c copy`
— input-seeking (KHÔNG decode từ đầu file) nên rất nhanh dù video gốc dài ~15-20 phút.
`-c copy` (không re-encode) nên đoạn cắt không chính xác tuyệt đối tới khung hình (căn theo
keyframe H.264 gần nhất) nhưng đủ tốt cho mục đích xem trước, và NHANH hơn hẳn re-encode.

CHỦ ĐỘNG (lazy): chỉ gọi hàm này khi người dùng thật sự bấm nút xem — KHÔNG gọi tự động cho
mọi dòng kết quả (top_k có thể tới 500 dòng, extract clip cho tất cả sẽ rất chậm/tốn đĩa)."""
from __future__ import annotations

import subprocess
from pathlib import Path

from config import _V3_ROOT
from video_audio import read_video_bytes

VIDEO_CACHE_DIR = _V3_ROOT / ".cache" / "video_full"  # video gốc trích 1 lần/video_id, giữ lại
CLIP_CACHE_DIR = _V3_ROOT / ".cache" / "video_clips"  # đoạn ngắn đã cắt, giữ lại theo (video_id,t)
VIDEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CLIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PAD_SECONDS = 3.0  # +-3s quanh frame -> đoạn ~6s


def _cached_full_video_path(video_id: str) -> Path:
    """Trả path file mp4 gốc trên đĩa cục bộ — trích từ zip BTC 1 LẦN, các lần xem sau (kể cả
    xem nhiều mốc khác nhau trong CÙNG video, hay xem lại) đọc thẳng từ cache, không đọc zip
    lại (zip đọc nguyên video vào RAM mỗi lần, đắt nếu gọi lặp lại nhiều lần)."""
    path = VIDEO_CACHE_DIR / f"{video_id}.mp4"
    if not path.exists():
        video_bytes = read_video_bytes(video_id)
        tmp_path = path.with_suffix(".mp4.tmp")
        tmp_path.write_bytes(video_bytes)
        tmp_path.replace(path)  # ghi nguyên tử — tránh file dở dang neu bi ngat giua chung
    return path


def get_clip_bytes(video_id: str, center_seconds: float, pad_seconds: float = DEFAULT_PAD_SECONDS) -> bytes:
    """Trả bytes mp4 đoạn [center_seconds - pad, center_seconds + pad] (kẹp >=0) của video_id.
    Cache theo (video_id, làm tròn giây của center) — bấm xem lại cùng 1 kết quả không cắt lại."""
    start = max(0.0, center_seconds - pad_seconds)
    duration = 2 * pad_seconds
    cache_key = f"{video_id}_{round(start, 1)}_{round(duration, 1)}.mp4"
    clip_path = CLIP_CACHE_DIR / cache_key
    if clip_path.exists():
        return clip_path.read_bytes()

    full_path = _cached_full_video_path(video_id)
    tmp_clip = clip_path.with_suffix(".mp4.tmp")
    cmd = [
        "ffmpeg", "-y", "-ss", str(start), "-i", str(full_path), "-t", str(duration),
        "-c", "copy", "-avoid_negative_ts", "make_zero", str(tmp_clip),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not tmp_clip.exists() or tmp_clip.stat().st_size == 0:
        # -c copy that bai (vd cat dung luc khong co keyframe H.264 gan do) -> fallback re-encode
        # (cham hon nhung LUON ra duoc file, dung 1 lan/clip nen chap nhan duoc).
        cmd_reencode = [
            "ffmpeg", "-y", "-ss", str(start), "-i", str(full_path), "-t", str(duration),
            "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(tmp_clip),
        ]
        subprocess.run(cmd_reencode, capture_output=True, check=True)
    tmp_clip.replace(clip_path)
    return clip_path.read_bytes()
