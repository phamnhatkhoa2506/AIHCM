"""Cắt 1 VÙNG ảnh (crop theo box object đã detect) từ keyframe — dùng cho các script offline
cần crop trước khi encode bằng model khác (build_region_embeddings.py, audit_object_labels.py).

# 2026-08-20 (theo yeu cau nguoi dung: "dọn dẹp triệt để... Region-CLIP embedding không phải
# dùng SigLIP-2 à bạn" - phat hien qua cau hoi cua nguoi dung) - module nay TRUOC DAY con co
# encode_region()/encode_regions_batch() (encode local qua sentence_transformers clip-ViT-B-32)
# - DA XOA vi 0 caller thuc su: tinh nang Region-CLIP rerank LIVE (app.py) chay hoan toan tren
# server Modal rieng (offline/modal_infra/region_rerank_app.py) dung SigLIP2 (KHONG phai CLIP-
# 32), tu encode ca text lan doc region_embeddings_siglip.npy (sinh boi build_dense_region_
# embeddings_shard.py, cung dung SigLIP2 - xem docstring file do). encode_region()/
# encode_regions_batch() o day CHUA TUNG duoc goi tu duong live nao - offline/audit_object_
# labels.py (noi con dung crop_region() ben duoi) tu goi THANG Modal app "aic2026-region-clip"
# (region_clip_app.py, CLIP-ViT-B-32) cho phan fallback encode, khong qua ham local o day.
# Chi con crop_region() (thuan PIL, KHONG dung model gi) la con nguoi goi thuc su.
"""
from __future__ import annotations

import io

from PIL import Image

from keyframe_images import read_keyframe_bytes

Box = tuple[float, float, float, float]  # (ymin, xmin, ymax, xmax), normalize 0-1


def crop_region(video_id: str, local_idx: int, box: Box) -> Image.Image:
    img_bytes = read_keyframe_bytes(video_id, local_idx)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = img.size
    ymin, xmin, ymax, xmax = box
    left, top = max(0, int(xmin * w)), max(0, int(ymin * h))
    right, bottom = min(w, int(xmax * w)), min(h, int(ymax * h))
    right, bottom = max(right, left + 1), max(bottom, top + 1)  # box qua nho -> ep >=1px
    return img.crop((left, top, right, bottom))
