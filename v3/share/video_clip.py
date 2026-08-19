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
from video_audio import local_video_path, read_video_bytes

VIDEO_CACHE_DIR = _V3_ROOT / ".cache" / "video_full"  # video gốc trích 1 lần/video_id, giữ lại
CLIP_CACHE_DIR = _V3_ROOT / ".cache" / "video_clips"  # đoạn ngắn đã cắt, giữ lại theo (video_id,t)
# 2026-08-20 (theo yeu cau nguoi dung: "chức năng xem video của kết quả temporal để tinh chỉnh
# ... approve frame mong muốn khi đã kiểm duyệt qua video") - anh xem-truoc khi nguoi dung
# tinh chinh 1 moc TRAKE bang tay (KHONG gioi han theo cac frame dense_meta.parquet DA co san -
# cho phep "scrub" TU DO tai BAT KY thoi diem nao, xem extract_single_frame duoi day).
CORRECTED_FRAME_CACHE_DIR = _V3_ROOT / ".cache" / "trake_corrected_frames"
VIDEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CLIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CORRECTED_FRAME_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PAD_SECONDS = 3.0  # +-3s quanh frame -> đoạn ~6s (CHỈ dùng khi fallback, xem
# get_shot_clip_bytes - đường CHÍNH giờ cắt theo đúng RANH GIỚI SHOT thay vì pad cố định).
MIN_SHOT_CLIP_SECONDS = 2.0  # shot quá ngắn (vd 1-2 keyframe) -> nới ra cho dễ xem, căn giữa
# frame kết quả thay vì play 1 đoạn <1s giật cục.
MAX_SHOT_CLIP_SECONDS = 20.0  # shot quá dài (cảnh tĩnh/máy quay cố định lâu) -> KHÔNG phát cả
# shot (có thể vài phút) - cắt 1 đoạn 20s CĂN GIỮA frame kết quả trong shot đó.


def _cached_full_video_path(video_id: str) -> Path:
    """Trả path file mp4 gốc trên đĩa cục bộ, sẵn sàng cho ffmpeg seek.

    2026-08-15 (theo yêu cầu người dùng: "mới unzip Videos_L21_a ra rồi, đã tích hợp chưa?") —
    nếu video đã được GIẢI NÉN THỦ CÔNG sẵn ra folder (xem video_audio.local_video_path), dùng
    THẲNG file đó — KHÔNG copy lại vào .cache/video_full/ nữa (video gốc đã là file thật, seek
    được ngay, copy thêm 1 bản chỉ tốn đĩa vô ích). Chỉ khi CÒN là .zip mới cần trích ra cache
    (zip không seek trực tiếp được, xem docstring đầu file) — làm 1 LẦN duy nhất, các lần xem
    sau (kể cả mốc khác trong cùng video) dùng lại cache, không đọc zip lại."""
    local_path = local_video_path(video_id)
    if local_path is not None:
        return local_path

    path = VIDEO_CACHE_DIR / f"{video_id}.mp4"
    if not path.exists():
        video_bytes = read_video_bytes(video_id)
        # ".tmp" o GIUA ten (khong o cuoi) - dong bo voi fix o get_clip_bytes(), tranh bay
        # tuong tu neu sau nay co code khac doc lai duoi file nay.
        tmp_path = path.with_name(path.stem + "_tmp.mp4")
        tmp_path.write_bytes(video_bytes)
        tmp_path.replace(path)  # ghi nguyên tử — tránh file dở dang neu bi ngat giua chung
    return path


def _cut_clip(video_id: str, start: float, duration: float) -> bytes:
    """Cat that bang ffmpeg [start, start+duration) cua video_id, cache theo (video_id, start,
    duration) - dung CHUNG cho ca get_clip_bytes (fallback, pad co dinh) va get_shot_clip_bytes
    (duong CHINH, cat theo ranh gioi shot that)."""
    start = max(0.0, start)
    cache_key = f"{video_id}_{round(start, 1)}_{round(duration, 1)}.mp4"
    clip_path = CLIP_CACHE_DIR / cache_key
    if clip_path.exists():
        return clip_path.read_bytes()

    full_path = _cached_full_video_path(video_id)
    # BUG THAT (2026-08-16, phat hien qua test that voi video vua giai nen): ffmpeg chon dinh
    # dang output THEO DUOI FILE CUOI CUNG - "*.mp4.tmp" bi doc la duoi ".tmp" (khong nhan
    # dang duoc), loi "Unable to choose an output format" NGAY CA VOI -c copy. Cung 1 lop bug
    # da gap voi np.save() truoc day (duoi .npy tu dong them). Fix: dat ".tmp" o GIUA ten file
    # (truoc ".mp4"), khong phai o CUOI - "*_tmp.mp4" van la duoi .mp4 hop le.
    tmp_clip = clip_path.with_name(clip_path.stem + "_tmp.mp4")
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


def get_clip_bytes(video_id: str, center_seconds: float, pad_seconds: float = DEFAULT_PAD_SECONDS) -> bytes:
    """Trả bytes mp4 đoạn [center_seconds - pad, center_seconds + pad] (kẹp >=0) của video_id -
    CẮT THEO PAD CỐ ĐỊNH, dùng làm FALLBACK khi không xác định được shot (xem
    get_shot_clip_bytes — đường CHÍNH nên dùng, cắt đúng ranh giới shot thật thay vì đoạn cố
    định không liên quan tới nội dung cảnh)."""
    start = max(0.0, center_seconds - pad_seconds)
    return _cut_clip(video_id, start, 2 * pad_seconds)


TRAKE_FIXED_WINDOW_SECONDS = 7.5  # 2026-08-18 (theo yeu cau nguoi dung: "chuyen tu lay tu shot
# sang thoi luong co dinh theo cua so 7 8s" - CHI cho TRAKE, KIS/QA van giu nguyen cat theo shot
# o get_shot_clip_bytes ben tren). Can giua frame_id ket qua, KHONG phu thuoc ranh gioi shot -
# hop ly hon cho TRAKE vi cac moc trong 1 chuoi co the nam gan ranh gioi shot that.


def get_fixed_window_clip_bytes(
    video_id: str, frame_id: int, window_seconds: float = TRAKE_FIXED_WINDOW_SECONDS,
) -> bytes:
    """Trả bytes mp4 đoạn [frame_id/fps - w/2, frame_id/fps + w/2] CĂN GIỮA frame_id — cửa sổ
    thời lượng CỐ ĐỊNH, không phụ thuộc ranh giới shot (khác get_shot_clip_bytes). Dùng riêng
    cho TRAKE (xem app.py::_render_floating_video_player, gọi khi active_video["fixed_window"]
    truthy) — KIS/Q&A không đổi, vẫn cắt theo đúng shot."""
    import sys as _sys
    from pathlib import Path as _P
    _share_dir = str(_P(__file__).resolve().parent)
    if _share_dir not in _sys.path:
        _sys.path.insert(0, _share_dir)
    from tiers.dense_search import _fps_by_video

    fps = _fps_by_video().get(video_id, 25.0)
    center_s = frame_id / fps
    start_s = max(0.0, center_s - window_seconds / 2)
    return _cut_clip(video_id, start_s, window_seconds)


def extract_single_frame(video_id: str, time_seconds: float) -> Path:
    """Trích 1 frame DUY NHẤT tại đúng thời điểm time_seconds (2026-08-20, tính năng "tinh
    chỉnh TRAKE" — xem tune_trake_anchor / _render_trake_frame_tune trong app.py) — KHÁC mọi
    hàm khác trong file này (chỉ cắt CLIP), và KHÔNG giới hạn theo các frame đã có sẵn trong
    dense_meta.parquet (mật độ mẫu thưa, ~0.55-2.65s/frame) — cho phép người dùng "scrub" tới
    BẤT KỲ giây nào rồi lấy đúng frame tại đó, để chỉnh sửa kết quả TRAKE ban đầu nếu chưa đúng.

    Trả về PATH file JPEG (không phải bytes) — cache theo (video_id, giây làm tròn 0.1s) để
    dùng LẠI ĐƯỢC như 1 image_path bình thường (canvas vẽ Object/OCR, VLM đọc chữ...), giống hệt
    các keyframe khác trong hệ thống, không cần xử lý đặc biệt ở nơi gọi."""
    t = max(0.0, round(time_seconds, 1))
    cache_path = CORRECTED_FRAME_CACHE_DIR / f"{video_id}_t{t}.jpg"
    if cache_path.exists():
        return cache_path

    full_path = _cached_full_video_path(video_id)
    tmp_path = cache_path.with_name(cache_path.stem + "_tmp.jpg")  # ".tmp" o GIUA (khong o cuoi)
    # - dong bo voi _cut_clip(), tranh bug dinh dang output doan theo duoi file (xem _cut_clip).
    cmd = [
        "ffmpeg", "-y", "-ss", str(t), "-i", str(full_path),
        "-frames:v", "1", "-q:v", "2", str(tmp_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not tmp_path.exists() or tmp_path.stat().st_size == 0:
        raise RuntimeError(
            f"ffmpeg không trích được frame tại t={t}s cho {video_id}: "
            f"{result.stderr.decode(errors='ignore')[-300:]}"
        )
    tmp_path.replace(cache_path)
    return cache_path


def get_shot_clip_bytes(video_id: str, frame_id: int) -> bytes:
    """Trả bytes mp4 đúng ĐOẠN SHOT chứa frame_id (2026-08-16, theo yêu cầu người dùng —
    index/keyframe_meta_all.jsonl chỉ ra ranh giới shot thật, KHÔNG nên cắt cố định +-3s như
    trước vì có thể cắt ngang cảnh hoặc bỏ lỡ phần đầu/cuối shot). Ranh giới shot lấy từ
    dense_meta.parquet (đã khớp 100% quy ước với keyframe_meta_all.jsonl nhưng phủ ĐỦ 873/873
    video — xem tiers.dense_search.get_shot_frame_range).

    Kẹp độ dài trong [MIN_SHOT_CLIP_SECONDS, MAX_SHOT_CLIP_SECONDS] — shot quá ngắn (vài
    keyframe) được nới ra cho dễ xem, shot quá dài (cảnh tĩnh) bị cắt bớt, CĂN GIỮA đúng
    frame_id (không lệch về đầu/cuối shot)."""
    import sys as _sys
    from pathlib import Path as _P
    _share_dir = str(_P(__file__).resolve().parent)  # video_clip.py nam ngay trong share/
    if _share_dir not in _sys.path:
        _sys.path.insert(0, _share_dir)  # phong than neu goi module nay ngoai app.py (app.py
        # da tu insert share/ vao sys.path roi nen binh thuong khong can, nhung goi truc tiep
        # tu noi khac - vd test CLI - can dong nay de "from tiers.dense_search import" chay duoc.
    from tiers.dense_search import _fps_by_video, get_shot_frame_range

    fps = _fps_by_video().get(video_id, 25.0)
    bounds = get_shot_frame_range(video_id, frame_id)
    if bounds is None:
        return get_clip_bytes(video_id, frame_id / fps)  # video khong co trong dense corpus

    lo, hi = bounds
    start_s, end_s = lo / fps, hi / fps
    duration = end_s - start_s
    center_s = frame_id / fps

    if duration < MIN_SHOT_CLIP_SECONDS:
        start_s = max(0.0, center_s - MIN_SHOT_CLIP_SECONDS / 2)
        duration = MIN_SHOT_CLIP_SECONDS
    elif duration > MAX_SHOT_CLIP_SECONDS:
        start_s = max(start_s, center_s - MAX_SHOT_CLIP_SECONDS / 2)
        start_s = min(start_s, end_s - MAX_SHOT_CLIP_SECONDS)
        duration = MAX_SHOT_CLIP_SECONDS

    return _cut_clip(video_id, start_s, duration)
