"""Đường dẫn dữ liệu Batch 1 AIC 2026 + cấu hình Tier 1 (vector search)."""
from pathlib import Path

from dotenv import load_dotenv

# File nay nam trong share/ (v3/share/config.py) - .parent.parent moi la goc v3/, noi thuc su
# chua index/ va .cache/. Di chuyen file nay sang thu muc khac PHAI xem lai dong nay.
_V3_ROOT = Path(__file__).resolve().parent.parent

# 2026-08-20 (theo yeu cau nguoi dung, phat hien qua debug hieu nang TRAKE "SigLIP-2 only van
# 132s" - do that: .env (AIC_LOCAL_MODELS=1, tao 2026-08-19) CHUA TUNG duoc code doc o dau ca -
# app.py/config.py KHONG goi load_dotenv(), nen bien nay khong bao gio thuc su duoc set, app
# VAN chay qua Modal (remote) nhu cu ma KHONG BAO GIO that su chuyen sang local - day la nguyen
# nhan that cua do cham khi tang top_k (truyen hang chuc nghin dong qua mang cho MOI anchor),
# KHONG phai do thuat toan cham. load_dotenv() o day (dau tien MOI module import, vi moi file
# deu "from config import ...") de .env thuc su co hieu luc TRUOC khi cac module khac doc
# os.environ.get("AIC_LOCAL_MODELS", ...) (local_text_encoders.py, dense_search.py).
load_dotenv(_V3_ROOT / ".env")


# ============================================================================
# DÙNG LIVE (app.py / share/ modules mà app.py gọi trực tiếp lúc chạy search)
# ============================================================================

# Video gốc (Videos_*.zip) - video_audio.py doc TRUC TIEP tu day de phat video trong Playback
# (app.py::read_video_bytes) VA de trich audio cho ASR (offline/build_asr_index.py).
DATA_ROOT = Path(r"D:\Programming\AIHCM\data\Competition")

INDEX_DIR = _V3_ROOT / "index"

# resources.py doc de xay row_pos (tier1_filter.py dung cho hard-filter khi KHONG co object/OCR
# filter nao) - KHONG nham voi DENSE_META_PATH (pipeline dense) ben duoi.
INDEX_META_PATH = INDEX_DIR / "meta.parquet"

# resources.py -> tier1_filter.by_metadata() (loc theo kenh/ngay/tu khoa trong UI).
VIDEO_METADATA_PATH = INDEX_DIR / "video_metadata.parquet"

# resources.py -> tier1_filter.by_objects() (hard-filter object/OCR) - AP DUNG CHO CA pipeline
# dense moi (xem dense_index_app.py docstring), khong rieng gi pipeline CLIP-32 cu.
OBJECTS_INDEX_PATH = INDEX_DIR / "objects_index.parquet"

# 2026-08-19 (theo yeu cau nguoi dung: "d:\...\index\dense\dense_meta.parquet ... mình thấy
# chưa có trong config") - goc thu muc + meta cua pipeline DENSE (SigLIP2/PE-Core/BEiT-3, tu
# trich) - dense_search.py (Tang 2/3 CHINH, app.py dang chay live) doc truc tiep tu day.
DENSE_DIR = INDEX_DIR / "dense"
DENSE_META_PATH = DENSE_DIR / "dense_meta.parquet"

# Cache model HuggingFace/sentence-transformers ngay trong workspace này, không dùng
# %USERPROFILE%\.cache mặc định. Dùng CHUNG cho cả model live (resources.py setdefault HF_HOME)
# lẫn các script offline bên dưới.
MODEL_CACHE_DIR = _V3_ROOT / ".cache" / "huggingface"


# ============================================================================
# CHỈ DÙNG OFFLINE (script build/audit 1 lần trong offline/, KHÔNG được app.py gọi lúc search)
# ============================================================================

# # 2026-08-19 (theo yeu cau nguoi dung, nhanh git khac: "chạy mọi thứ local... path thì bạn
# # đừng hardcode, bỏ vào file config hết") - goc du lieu "Our" (tu trich, dense corpus,
# # offline/*.py + share/dense_volume_map.py dung) va thu muc output cua repo keyframe rieng
# # (dung chung cho build/upload embeddings, xem cac script offline/*.py). TRUOC DAY hardcode
# # rai rac trong nhieu file (dense_volume_map.py, offline/upload_dense_to_volume.py,
# # offline/merge_dense_embeddings.py...) - gom VE 1 CHO de doi may/o dia chi can sua 1 dong.
# OUR_DATA_ROOT = Path(r"D:\Programming\AIHCM\data\Our")
# KEYFRAME_REPO_ROOT = Path(r"D:\Programming\AIHCM\keyframe")
# KEYFRAME_OUTPUT_ROOT = KEYFRAME_REPO_ROOT / "data" / "output"

# # 2026-08-20 (theo yeu cau nguoi dung: "dọn dẹp triệt để... hiện tại khi chạy hệ thống thì
# # không còn gọi CLIP") - pipeline CLIP-ViT-B-32 cu (online/search.py, tiers/tier2_vector.py,
# # tiers/tier3_temporal.py) DA XOA HAN - app.py va offline/benchmark/evaluate.py GIO DEU chay
# # tren pipeline dense. CLIP_FEATURES_DIR/MAP_KEYFRAMES_DIR/INDEX_MATRIX_PATH/INDEX_FAISS_PATH
# # duoi day CHI con offline/build_index.py dung (script build 1 LAN, da chay xong tren corpus
# # hien co) - giu lai vi build_index.py CUNG LA noi build ra INDEX_META_PATH (meta.parquet, xem
# # muc LIVE o tren) chu KHONG PHAI vi con "song" trong search. Neu can setup lai tu dau tren may/
# # corpus khac, chay lai build_index.py 1 lan la du - matrix.npy/clip.faiss no sinh ra KHONG con
# # ai doc nua (chi la byproduct cua cung 1 vong lap, khong dang tach rieng thanh 2 script).
# CLIP_FEATURES_DIR = DATA_ROOT / "clip-features-32"
# MAP_KEYFRAMES_DIR = DATA_ROOT / "map-keyframes"
# INDEX_MATRIX_PATH = INDEX_DIR / "clip_matrix.npy"
# INDEX_FAISS_PATH = INDEX_DIR / "clip.faiss"

# # offline/build_metadata.py doc de build VIDEO_METADATA_PATH (muc LIVE o tren).
# MEDIA_INFO_DIR = DATA_ROOT / "media-info"

# # offline/build_object_stats.py + offline/build_objects_index.py doc de build OBJECTS_INDEX_PATH
# # (muc LIVE o tren) - ban than OBJECTS_DIR (thu muc JSON Faster R-CNN goc) khong ai doc luc live.
# OBJECTS_DIR = DATA_ROOT / "objects"

# # Phase P1 — endpoint Modal deploy Qwen2.5-VL-7B-Instruct qua vLLM, offline/p1_extract.py goi
# # (xem modal_infra/).
# MODAL_P1_URL = "https://khoap0410--aic2026-qwen25vl-p1-serve.modal.run"

# # Vocab discovery (open-vocab), offline/vocab_discovery.py goi (xem modal_infra/vocab_discovery_
# # app.py) — app Modal RIÊNG, config tối ưu cho output ngắn. SỬA URL này sau khi
# # `modal deploy vocab_discovery_app.py`.
# MODAL_VOCAB_URL = "https://vnht1202--aic2026-qwen25vl-vocab-serve.modal.run"

# # Encode TEXT: dùng bản multilingual (cùng không gian embedding với ảnh clip-ViT-B-32, do
# # Nils Reimers distill riêng cho mục đích này) thay vì bản gốc chỉ luyện tiếng Anh.
# # Đã verify: cosine(EN, VI cùng nghĩa) tăng từ ~0.6 (bản gốc) lên ~0.94-0.99 (bản này).
# # 2026-08-20 (theo yeu cau nguoi dung: "mình không còn dùng cái này nữa") - da XOA
# # label_translate.py::suggest()/_load_embeddings() (goi y mo qua CLIP similarity, xac nhan 0
# # caller thuc su) - CLIP_TEXT_MODEL_NAME chi con 1 noi dung THAT: offline/audit_object_labels.py
# # (zero-shot label audit, text-only, script offline doc lap). Neu audit script cung khong con
# # can nua, xoa not hang nay + import trong audit_object_labels.py.
# CLIP_TEXT_MODEL_NAME = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
