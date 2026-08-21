# AIC 2026 — app/

FastAPI + HTML/CSS/JS UI cho AIC 2026 (thay Streamlit `v3/online/app.py`). Độc lập hoàn toàn với
`v3/` — search/submission logic là bản sao riêng ở `backend/core/` (xem `backend/bootstrap.py`).

## 1. Cài Python deps

```
uv sync
```

(dùng `pyproject.toml` ở gốc monorepo — `.venv` dùng chung cho `app/` và `v3/`). Nếu không dùng
`uv`, cài bằng pip từ danh sách `dependencies` trong `pyproject.toml`.

**Không cần cài gì thêm để chạy qua Modal** (mặc định) — 3 query encoder (SigLIP2/PE-Core/BEiT-3)
+ Region-CLIP rerank đều gọi Modal remote, chỉ cần tài khoản Modal đã `modal token set`.

Chỉ cần đọc tiếp mục 2-3 nếu muốn chạy model **local** (không qua Modal) — xem `AIC_LOCAL_MODELS`
ở `.env`.

## 2. `.env`

Copy `.env.example` (gốc monorepo) thành `.env`, chỉnh đường dẫn dữ liệu cho đúng máy:

```
AIC_DATA_ROOT=...\data\Videos
AIC_INDEX_DIR=...\data\Runtime
AIC_MODEL_CACHE_DIR=...\app\.cache\huggingface
AIC_KEYFRAME_ROOT=...\data\Keyframes

AIC_LOCAL_MODELS=1                     # 1 = chạy model local thay vì Modal, để trống = luôn Modal
# AIC_LOCAL_QUERY_ENCODER_SIGLIP=      # override RIÊNG/model — không đặt thì fallback AIC_LOCAL_MODELS
# AIC_LOCAL_QUERY_ENCODER_PE_CORE=
# AIC_LOCAL_QUERY_ENCODER_BEIT3=
# AIC_LOCAL_DENSE_INDEX=
AIC_LOCAL_REGION_CLIP=0                # ví dụ override riêng: tắt hẳn Region-CLIP local dù AIC_LOCAL_MODELS=1

NVIDIA_NIM_API_KEY=
GOOGLE_API_KEY=
```

Mỗi biến `AIC_LOCAL_<TÊN>` (0/1) ưu tiên tuyệt đối; không đặt thì fallback về `AIC_LOCAL_MODELS`
chung. Nhờ vậy có thể trộn — vd chỉ SigLIP2 local, PE-Core/BEiT-3 vẫn qua Modal.

## 3. Model local (chỉ cần nếu bật `AIC_LOCAL_MODELS`/biến riêng tương ứng)

| Model | Cần gì | Repo/checkpoint |
|---|---|---|
| SigLIP2 | tự tải qua HuggingFace Hub lần chạy đầu | không cần setup thêm |
| PE-Core | repo `perception_models` + weight (tự tải qua HF khi load) | `backend/third_party/perception_models` |
| BEiT-3 | repo `unilm` (chỉ cần `unilm/beit3`) + checkpoint `.pth` + `beit3.spm` | `backend/third_party/unilm`, checkpoint ở `app/.cache/beit3/` |
| Region-CLIP rerank | chạy `offline/build_dense_region_embeddings_shard.py` trước (xem README `v3/`) | — |

`perception_models`/`unilm` **không có trên PyPI** — phải `git clone` tay. Đã vendor sẵn vào
`backend/third_party/` (commit trong git repo) nên **clone repo chính về là có sẵn**, không cần
tự chạy lệnh `git clone` nào nữa. Nếu vì lý do gì đó thư mục này trống (vd `.gitignore` lỡ chặn,
hoặc pull thiếu), tự dựng lại bằng:

```
git clone --depth 1 https://github.com/facebookresearch/perception_models.git backend/third_party/perception_models
pip install -e backend/third_party/perception_models --no-deps

git clone --depth 1 https://github.com/microsoft/unilm.git backend/third_party/unilm
```

Checkpoint BEiT-3 (`.pth` ~445MB, `.spm` ~1.3MB) **không** vendor vào git (quá nặng cho 1 file
nhị phân) — để ở `app/.cache/beit3/` (bị `.gitignore`). Code **tự tải** khi thiếu (xem
`local_text_encoders.py::_load_beit3`, URL từ `BEIT3_CKPT_URL`/`BEIT3_SPM_URL`) — chỉ cần chạy
app với BEiT-3 local bật lên lần đầu, không cần tải tay. Muốn tải tay trước (đỡ chờ lúc chạy):

```
curl -L -o app/.cache/beit3/beit3.spm \
  https://github.com/addf400/files/releases/download/beit3/beit3.spm
curl -L -o app/.cache/beit3/beit3_base_patch16_384_coco_retrieval.pth \
  https://github.com/addf400/files/releases/download/beit3/beit3_base_patch16_384_coco_retrieval.pth
```

PE-Core weight (~1.79GB) tự tải qua HuggingFace Hub lúc `pe.CLIP.from_config(..., pretrained=True)`
chạy lần đầu — không có URL tải tay riêng.

**Nếu mạng không tới được CDN HuggingFace** (`us.aws.cdn.hf.co`) — đã gặp thật: `curl` treo ở 0
byte dù repo không gated/không lỗi (verify bằng `curl -sIL .../resolve/main/PE-Core-B16-224.pt`
vẫn redirect OK nhưng tải nội dung thì treo) — model vẫn có sẵn trên Modal volume
`aic2026-hf-cache` (container Modal đã tải sẵn cho `query_encoders_app.py`). Lấy qua Modal thay
vì CDN:

```
modal volume ls aic2026-hf-cache hub/models--facebook--PE-Core-B16-224/blobs   # xem đúng blob-hash
modal volume get aic2026-hf-cache \
  "hub/models--facebook--PE-Core-B16-224/blobs/<blob-hash>" weight.bin
```

rồi dựng lại đúng cấu trúc HF cache dưới `app/.cache/huggingface/hub/models--facebook--PE-Core-B16-224/`:

```
blobs/<blob-hash>                              # = weight.bin vừa tải, đổi tên đúng hash
snapshots/<commit-sha>/PE-Core-B16-224.pt      # symlink trỏ tới ../../blobs/<blob-hash>
refs/main                                      # 1 dòng text = <commit-sha>
```

(`commit-sha` lấy từ `curl https://huggingface.co/api/models/facebook/PE-Core-B16-224` — trường
`sha`). Lưu ý: `modal volume get` tải cả 1 THƯ MỤC (không chỉ định file) từng bị lỗi `[Errno 13]
Permission denied` trên Windows dù thư mục đích chưa tồn tại — tải đúng 1 FILE (path tới blob cụ
thể) thay vì tải nguyên thư mục để tránh lỗi này.

### Pin thư viện — ĐỪNG đổi tuỳ tiện

`pyproject.toml` pin cứng `timm==0.4.12`, `fairscale==0.4.0`, `torchscale==0.2.0` — 3 phiên bản
này giải quyết đúng 1 xung đột: `torchscale==0.2.0` (BEiT-3 cần) tự ép `timm==0.4.12` trong
metadata của chính nó, nhưng PE-Core lại cần API `timm.layers` (chỉ có ở bản `timm` mới hơn) — xử
lý bằng shim `sys.modules["timm.layers"] = timm.models.layers` trong
`local_text_encoders.py::_load_pe_core`. Bump version bất kỳ trong 3 gói này sẽ phá lại xung đột.

## 4. Chạy

```
uv run --python ../.venv/Scripts/python.exe uvicorn backend.main:app --reload --port 8800
```

(chạy từ thư mục `app/`). Mở `http://127.0.0.1:8800`.

## 5. Kiểm tra nhanh

```
curl http://127.0.0.1:8800/api/health
```

Muốn xác nhận model local thật sự nạp được (không chỉ import suông):

```python
import backend.bootstrap  # noqa
import local_text_encoders as L
print(L.ENCODERS["beit3"]("con chó").shape)   # -> (1, 768)
print(L.ENCODERS["pe_core"]("con chó").shape)
```
