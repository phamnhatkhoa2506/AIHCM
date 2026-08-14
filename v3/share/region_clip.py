"""Region-CLIP: encode 1 VÙNG ảnh (crop theo box object đã detect), so với text mô tả thuộc
tính (màu áo, hoạ tiết...) — thay vì so cả câu với cả frame (cách Tầng 2 đang làm).

Cơ chế: crop theo box -> encode bằng CLIP IMAGE encoder gốc (`clip-ViT-B-32`, đúng model BTC
dùng tạo whole-frame feature — KHÁC với model multilingual chỉ có nhánh text). Encode text
thuộc tính bằng CLIP TEXT multilingual (`tiers.tier2_vector.encode_query`, đã verify cùng
không gian embedding với ảnh clip-ViT-B-32 trước đây) — 2 bên khớp không gian nên so được.

Ý tưởng gốc từng nhắc khi thiết kế Graph (region-CLIP làm attribute mở cho object node) —
nhưng đây là kỹ thuật ĐỘC LẬP, không cần Registry/VLM, áp thẳng vào Tầng 1/2 hiện có.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from config import CLIP_IMAGE_MODEL_NAME, MODEL_CACHE_DIR
from keyframe_images import read_keyframe_bytes

Box = tuple[float, float, float, float]  # (ymin, xmin, ymax, xmax), normalize 0-1

_image_model: SentenceTransformer | None = None


def _get_image_model() -> SentenceTransformer:
    global _image_model
    if _image_model is None:
        _image_model = SentenceTransformer(CLIP_IMAGE_MODEL_NAME, cache_folder=str(MODEL_CACHE_DIR))
    return _image_model


def crop_region(video_id: str, local_idx: int, box: Box) -> Image.Image:
    img_bytes = read_keyframe_bytes(video_id, local_idx)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = img.size
    ymin, xmin, ymax, xmax = box
    left, top = max(0, int(xmin * w)), max(0, int(ymin * h))
    right, bottom = min(w, int(xmax * w)), min(h, int(ymax * h))
    right, bottom = max(right, left + 1), max(bottom, top + 1)  # box qua nho -> ep >=1px
    return img.crop((left, top, right, bottom))


def encode_region(video_id: str, local_idx: int, box: Box) -> np.ndarray:
    """1 box — tiện cho test/demo, nhưng CHẬM khi gọi lặp lại nhiều lần (mỗi lần 1 lượt qua
    model riêng). Dùng encode_regions_batch() khi có >1 box — xem lý do trong docstring đó."""
    crop = crop_region(video_id, local_idx, box)
    vec = _get_image_model().encode([crop], convert_to_numpy=True, normalize_embeddings=True)
    return vec[0].astype(np.float32)


def encode_regions_batch(
    items: list[tuple[str, int, Box]], batch_size: int = 32
) -> np.ndarray:
    """Gộp N box thành 1 (hoặc vài) lượt qua model thay vì N lượt riêng lẻ — đây là chỗ chậm
    thật sự đã đo được (26s/35 box ~ 0.75s/box khi gọi từng cái). Model batch nội bộ đã tối ưu
    tính song song (đặc biệt có GPU), 1 lượt 35 ảnh nhanh hơn nhiều so với 35 lượt 1 ảnh.
    Trả về mảng (N, dim) đúng thứ tự `items`."""
    if not items:
        return np.zeros((0, 512), dtype=np.float32)
    crops = [crop_region(vid, li, box) for vid, li, box in items]
    vecs = _get_image_model().encode(
        crops, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )
    return vecs.astype(np.float32)


def score_region_vs_text(region_vec: np.ndarray, text_vec: np.ndarray) -> float:
    """text_vec: đã encode qua tier2_vector.encode_query (multilingual, cùng không gian)."""
    return float(region_vec @ text_vec.reshape(-1))


if __name__ == "__main__":
    import sys

    import pandas as pd

    from config import OBJECTS_INDEX_PATH
    from tiers.tier2_vector import encode_query

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    df = pd.read_parquet(OBJECTS_INDEX_PATH)
    row = df[df.label == "Person"].iloc[0]
    box = (row.ymin, row.xmin, row.ymax, row.xmax)
    print(f"Test frame: {row.video_id}/{row.local_idx}, box={box}")

    region_vec = encode_region(row.video_id, int(row.local_idx), box)
    for text in ["một người mặc áo đỏ", "một người mặc áo xanh", "một chiếc xe hơi"]:
        text_vec = encode_query(text)
        print(f"  '{text}' -> score={score_region_vs_text(region_vec, text_vec):.4f}")
