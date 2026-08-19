"""Modal app chạy pipeline trích keyframe dày (repo riêng D:\\Programming\\AIHCM\\keyframe,
AutoShot shot-detection + action-based frame sampling, đã tune đạt ~4-5x mật độ BTC — xem
keyframe/config.yaml, docs/METHODS.md) trên GPU Modal, CHỈ cho 3 bộ L26_c/d/e chưa ai chạy
(các bộ khác bạn đã chạy sẵn local, xem hội thoại 2026-08-11).

KHÔNG viết lại logic pipeline — chạy nguyên `scripts/run_keyframes_only.py <folder>` (đã có
sẵn resume + dừng trước bước caption/embedding) như 1 subprocess trong container GPU, y hệt
cách chạy local theo README, chỉ khác là input video và output nằm trên Modal Volume thay vì
đĩa local.

Volume `aic2026-dense-keyframe-data` mount tại /root/keyframe/data — driver
(run_dense_keyframes_l26_cde.py) upload video vào data/video/<folder>/ trước khi gọi hàm này,
rồi tải data/output/ về sau khi xong.

Deploy: modal deploy dense_keyframe_app.py
"""
import modal

REPO_LOCAL_PATH = r"D:\Programming\AIHCM\keyframe"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    # BUG THAT (2026-08-11): du da pip_install opencv-python-headless, mot dependency khac
    # (scenedetect/transnetv2-pytorch) keo them ban opencv-python THUONG can libGL.so.1 -
    # container debian_slim khong co san (khac may local co GUI libs) -> ImportError luc
    # import cv2. Them libgl1+libglib2.0-0 la fix chuan cho opencv trong container headless.
    .pip_install(
        "numpy>=1.24", "opencv-python-headless>=4.9", "Pillow>=10.0", "tqdm>=4.66", "pyyaml>=6.0",
        "scenedetect>=0.6.4", "transnetv2-pytorch>=1.0.5", "einops>=0.7", "imageio-ffmpeg>=0.5",
        "torch", "open_clip_torch>=2.24", "huggingface_hub",
        # KHONG cai transformers/accelerate/bitsandbytes/qwen-vl-utils/pymilvus/elasticsearch -
        # run_keyframes_only.py dung MockCaptioner (khong goi VLM) + tat embedding, khong can.
    )
    .add_local_dir(REPO_LOCAL_PATH, remote_path="/root/keyframe", ignore=["data", ".git", "logs", ".venv"])
)

data_vol = modal.Volume.from_name("aic2026-dense-keyframe-data", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("aic2026-hf-cache", create_if_missing=True)  # AutoShot ckpt tu tai tu HF

app = modal.App("aic2026-dense-keyframe")


@app.function(
    image=image,
    gpu="T4",  # AutoShot (shot detector) + CLIP ViT-B-32 dedup - nhe, khong can GPU manh
    timeout=6 * 3600,  # 1 bo ~100 video, du thoi gian rong
    volumes={"/root/keyframe/data": data_vol, "/root/.cache/huggingface": hf_cache_vol},
)
def run_folder(folder_name: str, force: bool = False) -> str:
    """folder_name vd 'Videos_L26_c' - phai da co san data/video/<folder_name>/*.mp4 trong
    Volume (driver upload truoc khi goi ham nay)."""
    import os
    import subprocess

    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    os.chdir("/root/keyframe")

    cmd = ["python", "-u", "scripts/run_keyframes_only.py", f"data/video/{folder_name}"]
    if force:
        cmd.append("--force")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    log = proc.stdout + "\n" + proc.stderr
    print(log)
    data_vol.commit()  # luu ket qua data/output/ vao Volume ngay ca khi loi giua chung (co resume)
    if proc.returncode != 0:
        raise RuntimeError(f"run_keyframes_only.py thoat loi (code {proc.returncode}) cho {folder_name}:\n{log[-3000:]}")
    return log
