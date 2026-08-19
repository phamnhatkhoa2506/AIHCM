"""Đường dẫn dữ liệu Batch 1 AIC 2026 + cấu hình Tier 1 (vector search)."""
from pathlib import Path

DATA_ROOT = Path(r"D:\Programming\AIHCM\data\Competition")

# 2026-08-19 (theo yeu cau nguoi dung, nhanh git khac: "chạy mọi thứ local... path thì bạn
# đừng hardcode, bỏ vào file config hết") - goc du lieu "Our" (tu trich, dense corpus,
# offline/*.py + share/dense_volume_map.py dung) va thu muc output cua repo keyframe rieng
# (dung chung cho build/upload embeddings, xem cac script offline/*.py). TRUOC DAY hardcode
# rai rac trong nhieu file (dense_volume_map.py, offline/upload_dense_to_volume.py,
# offline/merge_dense_embeddings.py...) - gom VE 1 CHO de doi may/o dia chi can sua 1 dong.
OUR_DATA_ROOT = Path(r"D:\Programming\AIHCM\data\Our")
KEYFRAME_REPO_ROOT = Path(r"D:\Programming\AIHCM\keyframe")
KEYFRAME_OUTPUT_ROOT = KEYFRAME_REPO_ROOT / "data" / "output"

# File nay nam trong share/ (v3/share/config.py) - .parent.parent moi la goc v3/, noi thuc su
# chua index/ va .cache/. Di chuyen file nay sang thu muc khac PHAI xem lai 2 dong nay.
_V3_ROOT = Path(__file__).resolve().parent.parent

CLIP_FEATURES_DIR = DATA_ROOT / "clip-features-32"
MAP_KEYFRAMES_DIR = DATA_ROOT / "map-keyframes"
MEDIA_INFO_DIR = DATA_ROOT / "media-info"
OBJECTS_DIR = DATA_ROOT / "objects"

INDEX_DIR = _V3_ROOT / "index"
INDEX_MATRIX_PATH = INDEX_DIR / "clip_matrix.npy"
INDEX_META_PATH = INDEX_DIR / "meta.parquet"
INDEX_FAISS_PATH = INDEX_DIR / "clip.faiss"
VIDEO_METADATA_PATH = INDEX_DIR / "video_metadata.parquet"
OBJECTS_INDEX_PATH = INDEX_DIR / "objects_index.parquet"

# 2026-08-19 (theo yeu cau nguoi dung: "d:\...\index\dense\dense_meta.parquet ... mình thấy
# chưa có trong config") - truoc day DENSE_DIR va cac duong dan con (dense_meta.parquet,
# <model>_matrix.npy, <model>_faiss.index...) dinh nghia RIENG trong
# share/tiers/dense_search.py (DENSE_DIR = INDEX_DIR / "dense", roi noi tiep tung file) -
# chuyen goc DENSE_DIR ve day, dense_search.py doi sang import lai (khong con dinh nghia
# rieng). Cac file *_matrix.npy/*_faiss.index/dense_meta.parquet van dat TEN DUNG chuan cu
# (khong doi ten file that tren dia), chi gom DUONG DAN GOC ve 1 cho.
DENSE_DIR = INDEX_DIR / "dense"
DENSE_META_PATH = DENSE_DIR / "dense_meta.parquet"

# Cache model HuggingFace/sentence-transformers ngay trong workspace này, không dùng
# %USERPROFILE%\.cache mặc định.
MODEL_CACHE_DIR = _V3_ROOT / ".cache" / "huggingface"

# Model đúng tên với thư mục "clip-features-32" -> BTC dùng sentence-transformers clip-ViT-B-32
# để tạo feature ẢNH (không đổi, giữ nguyên clip-features-32 đã có).
CLIP_IMAGE_MODEL_NAME = "clip-ViT-B-32"

# Phase P1 — endpoint Modal deploy Qwen2.5-VL-7B-Instruct qua vLLM (xem modal_infra/).
MODAL_P1_URL = "https://khoap0410--aic2026-qwen25vl-p1-serve.modal.run"

# Vocab discovery (open-vocab, xem modal_infra/vocab_discovery_app.py) — app Modal RIÊNG,
# config tối ưu cho output ngắn. SỬA URL này sau khi `modal deploy vocab_discovery_app.py`.
MODAL_VOCAB_URL = "https://vnht1202--aic2026-qwen25vl-vocab-serve.modal.run"

# Encode TEXT: dùng bản multilingual (cùng không gian embedding với ảnh clip-ViT-B-32, do
# Nils Reimers distill riêng cho mục đích này) thay vì bản gốc chỉ luyện tiếng Anh.
# Đã verify: cosine(EN, VI cùng nghĩa) tăng từ ~0.6 (bản gốc) lên ~0.94-0.99 (bản này).
CLIP_TEXT_MODEL_NAME = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
