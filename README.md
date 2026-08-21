# AIC 2026 — v3 — Setup môi trường LOCAL (không qua Modal)

Mặc định code gọi model qua Modal (server GPU/CPU luôn giữ ấm, xem `offline/modal_infra/`).
Tài liệu này hướng dẫn chạy **3 model text-encoder (SigLIP2/PE-Core/BEiT-3)** và
**Region-CLIP rerank** **trực tiếp trên máy**, không cần tài khoản Modal cho việc chạy
`online/app.py` hàng ngày.

> Bật chế độ local bằng 1 biến môi trường — không cần sửa code:
> ```
> AIC_LOCAL_MODELS=1
> ```
> Không set (hoặc set `0`) → giữ nguyên hành vi cũ, gọi Modal như trước.

## 1. Những gì chạy local vs những gì KHÔNG (quan trọng, đọc trước)

| Thành phần | Local được không? | Ghi chú |
|---|---|---|
| SigLIP2 / PE-Core / BEiT-3 (mã hoá câu truy vấn) | ✅ Có — `AIC_LOCAL_MODELS=1` | `share/local_text_encoders.py` |
| Region-CLIP rerank (thuộc tính màu/quần áo...) | ✅ Có — cùng biến trên | `share/tiers/dense_search.py::_LocalRegionRerank` |
| Tầng 2 — xếp hạng vector (embedding matrix + FAISS, ~7.4GB/3 model) | ✅ Có — cùng biến trên | `share/tiers/dense_search.py::_rank_single_local` — **mặc định (không set biến) đã CHUYỂN SANG Modal** (`aic2026-dense-index`) từ 2026-08-16, đây là phần RAM nặng nhất trên máy local trước đó. |
| LLM phân rã câu truy vấn (`extract_entities`) | ❌ Không — vốn đã KHÔNG chạy trên Modal | Gọi API NVIDIA NIM (`online/query_planner.py`), cần `NVIDIA_NIM_API_KEY`, luôn cần mạng. |
| VQA cho Q&A (trả lời câu hỏi trên ảnh) | ❌ Không — vốn đã KHÔNG chạy trên Modal | Cũng gọi NVIDIA NIM (`online/submission_pipeline.py`). |
| OWLv2 detection / OCR / ASR / Grounding DINO | Chỉ cần nếu BUILD LẠI index từ đầu | Không cần cho chạy `app.py` hàng ngày nếu `index/*.parquet` đã có sẵn — xem mục 6. |

Nói cách khác: việc "chạy local thay vì Modal" chỉ áp dụng cho 2 thứ THẬT SỰ đang chạy trên
Modal khi bạn dùng app hàng ngày (2 dòng đầu bảng). Phần LLM/VQA vốn dùng API ngoài
(NVIDIA NIM) từ trước, không liên quan Modal.

## 2. Yêu cầu hệ thống

- **Python 3.11** (khớp đúng bản Modal image đang dùng — bản khác có thể chạy được nhưng
  chưa test).
- **ffmpeg** có trên `PATH` (dùng cho `share/video_clip.py` — cắt clip video xem trước).
  Kiểm tra: `ffmpeg -version`.
- **git** (để clone 2 repo model không có trên PyPI: PE-Core, BEiT-3).
- **~10GB ổ đĩa trống** cho cache model (HuggingFace + checkpoint BEiT-3 + 2 repo clone),
  cộng thêm dung lượng data đã có sẵn (`index/`, ảnh keyframe...).
- **GPU không bắt buộc** cho chạy app tương tác — 3 model text-encoder chỉ mã hoá 1 CÂU
  NGẮN/lần (query người dùng gõ), CPU đủ nhanh (<1-2s/câu sau khi model đã nạp). GPU chỉ thật
  sự cần cho các script OFFLINE build index (mục 6) — encode hàng trăm nghìn ảnh thì CPU quá
  chậm.

## 3. Cài đặt

### 3.1. Tạo virtualenv

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

### 3.2. Cài `requirements.txt`

```powershell
pip install -r requirements.txt
```

Lệnh này cài được **hầu hết** thư viện, nhưng CHƯA đủ — làm tiếp 3.3 và 3.4 bên dưới
(2 model không có trên PyPI + 1 lần reinstall numpy bắt buộc nếu bạn cần build OCR local).

### 3.3. Clone PE-Core + BEiT-3 (không có trên PyPI)

```powershell
mkdir .cache
git clone --depth 1 https://github.com/facebookresearch/perception_models.git .cache\perception_models
pip install -e .cache\perception_models --no-deps
git clone --depth 1 https://github.com/microsoft/unilm.git .cache\unilm
```

`--no-deps` là **bắt buộc** — nếu không, `perception_models` sẽ tự kéo lại `torch`/`torchvision`
bản khác, đè lên bản đã pin ở `requirements.txt` và gây xung đột. `unilm` không cần
`pip install` gì cả — code import thẳng theo đường dẫn (`sys.path.insert`), xem
`share/local_text_encoders.py`.

BEiT-3 checkpoint (`beit3_base_patch16_384_coco_retrieval.pth`, ~700MB) + file tokenizer
(`beit3.spm`) sẽ **tự tải về** `.cache/beit3/` trong lần chạy đầu tiên (không cần tải tay).

### 3.4. Nếu cần build lại OCR (PaddleOCR) local — xem mục 6

Không cần cho chạy app hàng ngày. Nếu có build, đọc kỹ mục "Xung đột thư viện #1" trước.

### 3.5. File `.env`

Tạo `.env` ở gốc `v3/`:

```
NVIDIA_NIM_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
AIC_LOCAL_MODELS=1
```

`NVIDIA_NIM_API_KEY` **luôn cần** (LLM query-planning + VQA, xem mục 1) — lấy tại
https://build.nvidia.com/. `AIC_LOCAL_MODELS=1` bật chế độ local cho 2 thành phần mô tả ở
mục 1; bỏ dòng này (hoặc set `0`) để quay lại gọi Modal như cũ.

## 4. Chạy

```powershell
streamlit run online/app.py
```

**Lần chạy đầu tiên sẽ chậm** (có thể vài phút) — 3 model text-encoder + Region-CLIP tải
weight từ HuggingFace Hub và nạp vào RAM lần đầu (sau đó cache trong tiến trình Python, các
câu truy vấn tiếp theo trong CÙNG phiên Streamlit sẽ nhanh — không tải/nạp lại). Restart hẳn
Streamlit (không chỉ bấm Rerun) sẽ nạp lại từ đầu.

## 5. Xung đột thư viện — đọc kỹ trước khi cài lại từ đầu

### #1. `numpy<2` vs PaddleOCR (chỉ liên quan nếu build OCR local, mục 6)

`paddlepaddle-gpu`/`paddleocr` **không tương thích numpy 2.x** (lỗi runtime khó hiểu, không
phải lỗi cài đặt). Nhưng nhiều thư viện hiện đại (`transformers`, `torch` bản mới...) khi cài
sẽ **kéo numpy 2.x lên** nếu không ghim.

**Cách sửa** (đúng thứ tự — dùng lại chính xác cách `offline/modal_infra/ocr_app.py` làm):
```powershell
pip install "numpy<2"                      # ghim TRƯỚC
pip install paddlepaddle-gpu==3.3.1 --index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/
pip install paddleocr==2.7.3
pip install "numpy<2" --force-reinstall --no-deps   # QUAN TRỌNG: paddleocr có thể âm thầm
    # kéo lại numpy 2.x qua dependency của NÓ — reinstall numpy<2 LẦN NỮA sau cùng, --no-deps
    # để không kéo theo thứ khác.
```
Nếu máy không có GPU/CUDA khớp bản `cu129`, dùng `paddlepaddle` (bản CPU) thay `paddlepaddle-gpu`
— chậm hơn nhiều lần nhưng vẫn chạy được, không cần sửa gì khác.

### #2. `timm==0.4.12` (BEiT-3) vs PE-Core cần `timm.layers`

- BEiT-3 phụ thuộc `torchscale==0.2.0`, mà bản **PyPI** của `torchscale==0.2.0` tự pin cứng
  `timm==0.4.12` trong metadata của nó (khác bản GitHub mới hơn ghi `timm==0.6.13` — đã xác
  minh thật trên máy, không phải đoán).
- PE-Core (`perception_models/core/vision_encoder/pe.py`) lại viết
  `from timm.layers import DropPath` — submodule `timm.layers` **chỉ tồn tại từ các bản timm
  MỚI HƠN** 0.4.12, không có trong 0.4.12.
- 2 model cùng cần chạy trong 1 tiến trình Python (`share/local_text_encoders.py` load cả 3
  model) → **không thể** cài 2 bản `timm` khác nhau cùng lúc.

**Cách sửa: KHÔNG bump version timm.** Dùng shim (đã code sẵn trong
`share/local_text_encoders.py`, không cần bạn làm gì thêm khi dùng file này) — trước khi
import `pe.py`, gán:
```python
import timm.models.layers as _timm_layers_legacy
sys.modules["timm.layers"] = _timm_layers_legacy
```
`timm.models.layers` (đường dẫn CŨ, có sẵn trong 0.4.12) chứa **đúng các class** mà
`timm.layers` (đường dẫn MỚI) export — chỉ khác tên module, không khác nội dung. Nếu bạn viết
code mới import PE-Core, nhớ làm bước shim này TRƯỚC khi `import core.vision_encoder.pe`.

### #3. `torch._six` không còn tồn tại (BEiT-3/torchscale cũ)

`torchscale`/`unilm` (viết cho torch cũ hơn) import `torch._six` — module nội bộ đã bị xoá
khỏi các bản `torch` hiện đại. Sửa bằng shim tương tự (đã có sẵn trong code):
```python
import types
six_mod = types.ModuleType("torch._six")
six_mod.inf = float("inf")
sys.modules["torch._six"] = six_mod
```
Làm TRƯỚC khi import bất kỳ module nào từ `unilm/beit3`.

## 6. (Tuỳ chọn) Build lại index từ đầu — cần GPU

Chỉ cần nếu `index/` chưa có sẵn hoặc muốn build lại dữ liệu gốc (keyframe/OCR/ASR/object
detection). KHÔNG cần cho chạy `app.py` với index có sẵn. Các script này hiện viết cho chạy
trên Modal GPU (`offline/*.py` gọi `offline/modal_infra/*_app.py`) — tự chạy local nghĩa là
viết lại phần `@app.cls`/`.remote()` thành gọi hàm trực tiếp, dùng đúng danh sách thư viện +
model tương ứng trong `requirements.txt` (mục OWLv2/OCR/ASR/Grounding DINO) và cần GPU NVIDIA
thật (encode hàng trăm nghìn frame trên CPU sẽ mất hàng chục giờ đến vài ngày, không khả thi).

## 7. Quay lại dùng Modal

Xoá hoặc set `AIC_LOCAL_MODELS=0` trong `.env`, đảm bảo `modal` đã cài
(`pip install modal`) và đã đăng nhập (`modal setup`). Không cần đổi gì khác trong code.
