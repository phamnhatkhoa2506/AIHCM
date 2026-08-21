r"""Đường dẫn dữ liệu + cấu hình pipeline search — bản sao RIÊNG của app/ (tách khỏi v3/, xem
backend/bootstrap.py). Toàn bộ đường dẫn đọc từ biến môi trường (khác máy/khác ổ đĩa giữa các
thành viên) — giá trị hardcode chỉ là fallback mặc định. Xem `.env.example` ở gốc monorepo
(`D:\Programming\AIHCM\.env.example`) để biết đầy đủ danh sách biến."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_APP_ROOT = Path(__file__).resolve().parent.parent.parent  # .../app/backend/core -> .../app
_AIHCM_ROOT = _APP_ROOT.parent
load_dotenv(_AIHCM_ROOT / ".env")

def _env_path(name: str, default: str = None) -> Path:
    return Path(os.environ.get(name, default))

DATA_ROOT = _env_path("AIC_DATA_ROOT")
INDEX_DIR = _env_path("AIC_INDEX_DIR") 

INDEX_META_PATH = INDEX_DIR / "meta.parquet"
VIDEO_METADATA_PATH = INDEX_DIR / "video_metadata.parquet"
OBJECTS_INDEX_PATH = INDEX_DIR / "objects_index.parquet"

DENSE_DIR = INDEX_DIR
DENSE_META_PATH = DENSE_DIR / "dense_meta.parquet"
KEYFRAME_ROOT = _env_path("AIC_KEYFRAME_ROOT", r"D:\Programming\AIHCM\data\Keyframes")

MODEL_CACHE_DIR = _env_path("AIC_MODEL_CACHE_DIR", str(_APP_ROOT / ".cache" / "huggingface"))
