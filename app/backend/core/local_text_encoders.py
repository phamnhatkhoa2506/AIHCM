"""Encode TEXT (câu truy vấn) cho 3 model dense (SigLIP2/PE-Core/BEiT-3).

2 CHẾ ĐỘ MỖI MODEL, chọn RIÊNG qua 3 biến môi trường (mặc định TẮT — giữ nguyên hành vi cũ):
  - AIC_LOCAL_QUERY_ENCODER_SIGLIP
  - AIC_LOCAL_QUERY_ENCODER_PE_CORE
  - AIC_LOCAL_QUERY_ENCODER_BEIT3
  (0/1, không set = TẮT nếu AIC_LOCAL_MODELS cũng không set)
  - "0" hoặc không set: gọi .remote() tới Modal app nhẹ luôn giữ ấm
    (aic2026-query-encoders, min_containers=1, xem offline/modal_infra/query_encoders_app.py).
  - "1": load ĐÚNG model đó TRỰC TIẾP trên máy (README.md — "Chạy model local thay vì Modal").
    Nặng hơn (RAM/VRAM + tải model lần đầu) nhưng không cần tài khoản Modal/mạng ổn định cho
    từng query — dùng đúng LẠI code load()/encode_*_text() của query_encoders_app.py, chỉ bỏ
    phần bọc @app.cls/@modal.method (chạy in-process, không qua Modal container).

2026-08-20 (theo yêu cầu người dùng: "với chế độ local cho toàn bộ, thêm env variable để chọn
có load region clip embedding, dense_index, hay query encoder nhé... với query encoder thì
thêm option để chọn load cho từng model") — TRƯỚC ĐÂY chỉ có 1 công tắc DUY NHẤT AIC_LOCAL_MODELS
bật/tắt CẢ 3 model cùng lúc (không chọn được, vd chỉ muốn SigLIP2 local còn PE-Core/BEiT-3 vẫn
qua Modal). Giờ MỖI model có công tắc RIÊNG, cho phép MIX (vd SigLIP2 local, 2 model kia remote)
— chỉ nạp ĐÚNG model được bật local vào RAM, không còn ép nạp cả 3 dù chỉ cần 1. AIC_LOCAL_MODELS
(biến CŨ) VẪN GIỮ làm giá trị MẶC ĐỊNH CHUNG khi biến riêng không đặt — tương thích ngược, không
phá cấu hình cũ của ai đang dùng "AIC_LOCAL_MODELS=1" bật tất cả.

Giữ NGUYÊN interface cũ (ENCODERS dict, trả về np.ndarray shape (1, dim)) — dense_search.py
KHÔNG cần sửa gì (vẫn gọi ENCODERS[model](query) y hệt trước, bất kể chế độ nào)."""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from app_flags import call_modal_with_timeout

MODAL_APP_NAME = "aic2026-query-encoders"
MODAL_CLASS_NAME = "QueryEncoder"

# 2026-08-21 (theo yêu cầu người dùng: "bỏ vào third_party trong thư mục backend để người khác
# clone repo của mình về thì có code sẵn") - perception_models (PE-Core)/unilm (BEiT-3) không có
# trên PyPI, phải git clone tay - trước đây clone vào app/.cache/ (bị .gitignore bỏ qua, mỗi máy
# phải tự clone lại). Giờ vendor thẳng vào backend/third_party/ (COMMIT vào git repo) - clone
# repo chính xong là có sẵn code, không cần chạy thêm lệnh git clone nào nữa. Checkpoint BEiT-3
# (.pth ~445MB, binary nặng) vẫn để ở app/.cache/beit3/ (gitignored) - tự tải qua HTTP khi cần
# (xem _load_beit3 dưới), không vendor vào git vì quá nặng cho 1 file nhị phân.
from config import _APP_ROOT

THIRD_PARTY_DIR = _APP_ROOT / "backend" / "third_party"

SIGLIP_MODEL_NAME = "google/siglip2-base-patch16-224"
PE_CORE_MODEL_NAME = "PE-Core-B16-224"
BEIT3_CKPT_URL = "https://github.com/addf400/files/releases/download/beit3/beit3_base_patch16_384_coco_retrieval.pth"
BEIT3_SPM_URL = "https://github.com/addf400/files/releases/download/beit3/beit3.spm"
BEIT3_MAX_TEXT_LEN = 64


def _local_mode_for(component: str) -> bool:
    """component: "siglip"|"pe_core"|"beit3". Env RIÊNG AIC_LOCAL_QUERY_ENCODER_<TÊN> (0/1) ưu
    tiên tuyệt đối - nếu KHÔNG đặt, fallback về AIC_LOCAL_MODELS (công tắc chung "bật/tắt tất
    cả", giữ tương thích ngược với cấu hình cũ)."""
    specific = os.environ.get(f"AIC_LOCAL_QUERY_ENCODER_{component.upper()}")
    if specific is not None:
        return specific.strip().lower() in ("1", "true", "yes")
    return os.environ.get("AIC_LOCAL_MODELS", "0").strip().lower() in ("1", "true", "yes")


# ============================================================ Modal (mac dinh)
@lru_cache(maxsize=1)
def _remote_encoder():
    import modal

    Encoder = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)
    return Encoder()


# ============================================================ Local (AIC_LOCAL_QUERY_ENCODER_*=1)
class _LocalQueryEncoder:
    """Y HET query_encoders_app.py::QueryEncoder.load()/encode_*_text(), chi bo wrapper Modal.
    Xem README.md muc "Xung dot thu vien" cho ly do timm==0.4.12 + shim timm.layers, torch._six,
    perception_models/unilm phai clone tay (khong co tren PyPI).

    2026-08-20: MOI model nap CO DIEU KIEN (chi khi _local_mode_for(<model>) True luc singleton
    nay duoc tao - xem _local_encoder() duoi day) - model KHONG duoc bat local thi thuoc tinh
    tuong ung = None, KHONG ton RAM/thoi gian tai (vd chi bat SigLIP2 local thi PE-Core/BEiT-3
    hoan toan khong dung git-clone/tai checkpoint gi ca)."""

    def __init__(self) -> None:
        # 2026-08-21 (bug that, phat hien khi nguoi dung hoi "chay BEiT-3/PE-Core thi thieu repo
        # phai khong"): TRUOC DAY import `_V3_ROOT` - ten CU con sot lai tu thoi dung chung
        # v3/share/. Khi tach app/backend/core/ ra ban SAO RIENG (xem backend/bootstrap.py),
        # config.py da doi ten bien thanh `_APP_ROOT` nhung file nay KHONG duoc sua theo -> BAT KY
        # query encoder nao chay local (ke ca SigLIP2, khong rieng PE-Core/BEiT-3) deu chet NGAY
        # bang ImportError truoc khi kip toi buoc kiem tra repo/checkpoint. Da kiem chung bang
        # cach chay that voi AIC_LOCAL_QUERY_ENCODER_SIGLIP=1. Nghia la ca cau hinh
        # AIC_LOCAL_MODELS=1 trong .env.example xua nay KHONG THE chay duoc.
        from config import _APP_ROOT

        os.environ.setdefault("HF_HOME", str(_APP_ROOT / ".cache" / "huggingface"))
        cache_root = _APP_ROOT / ".cache"

        self._siglip_model = None
        self._siglip_processor = None
        if _local_mode_for("siglip"):
            self._load_siglip()

        self._pe_model = None
        self._pe_tokenizer = None
        if _local_mode_for("pe_core"):
            self._load_pe_core(cache_root)

        self._beit3_model = None
        self._beit3_sp = None
        if _local_mode_for("beit3"):
            self._load_beit3(cache_root)

    def _load_siglip(self) -> None:
        # ---- SigLIP2 ----
        from transformers import AutoModel, AutoProcessor

        self._siglip_model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME).eval()
        self._siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)

    def _load_pe_core(self, cache_root) -> None:
        import sys

        # ---- PE-Core ----
        # BUG THAT (xem query_encoders_app.py): timm==0.4.12 (BAT BUOC cho BEiT-3/torchscale==
        # 0.2.0) KHONG co submodule "timm.layers" (chi tu ban timm moi hon), nhung PE-Core's
        # pe.py lai `from timm.layers import DropPath`. Shim "timm.layers" -> "timm.models.layers"
        # (duong dan CU, cung class, chi khac ten module) TRUOC khi import pe.py.
        # 2026-08-21: vendor trong backend/third_party/ (commit vào git, xem ghi chú THIRD_PARTY_
        # DIR đầu file) thay vì app/.cache/ (gitignored, mỗi máy phải tự clone lại).
        perception_models_dir = THIRD_PARTY_DIR / "perception_models"
        if not perception_models_dir.exists():
            raise RuntimeError(
                f"Chưa có backend/third_party/perception_models — xem README.md mục setup PE-Core:\n"
                f"  git clone --depth 1 https://github.com/facebookresearch/perception_models.git "
                f"{perception_models_dir}\n"
                f"  pip install -e {perception_models_dir} --no-deps"
            )
        sys.path.insert(0, str(perception_models_dir))
        import timm.models.layers as _timm_layers_legacy

        sys.modules["timm.layers"] = _timm_layers_legacy

        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as pe_transforms

        self._pe_model = pe.CLIP.from_config(PE_CORE_MODEL_NAME, pretrained=True).eval()
        self._pe_tokenizer = pe_transforms.get_text_tokenizer(self._pe_model.context_length)

    def _load_beit3(self, cache_root) -> None:
        import sys
        import types

        import requests
        import torch

        # ---- BEiT-3 ----
        beit3_cache_dir = cache_root / "beit3"
        beit3_cache_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = beit3_cache_dir / "beit3_base_patch16_384_coco_retrieval.pth"
        spm_path = beit3_cache_dir / "beit3.spm"
        for url, path in ((BEIT3_CKPT_URL, ckpt_path), (BEIT3_SPM_URL, spm_path)):
            if not path.exists():
                r = requests.get(url, stream=True, timeout=300)
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)

        # torch moi khong con torch._six (module noi bo cu) - torchscale/unilm van import no.
        if not hasattr(torch, "_six"):
            six_mod = types.ModuleType("torch._six")
            six_mod.inf = float("inf")
            sys.modules["torch._six"] = six_mod

        # 2026-08-21: vendor trong backend/third_party/ (commit vào git) - xem ghi chú THIRD_
        # PARTY_DIR đầu file. Checkpoint/spm ở trên VẪN để app/.cache/beit3/ (binary lớn, tự tải
        # qua HTTP, không vendor vào git).
        unilm_dir = THIRD_PARTY_DIR / "unilm"
        if not unilm_dir.exists():
            raise RuntimeError(
                f"Chưa có backend/third_party/unilm — xem README.md mục setup BEiT-3:\n"
                f"  git clone --depth 1 https://github.com/microsoft/unilm.git {unilm_dir}"
            )
        sys.path.insert(0, str(unilm_dir / "beit3"))
        import modeling_finetune  # noqa: F401 - dang ky beit3_base_patch16_384_retrieval
        import sentencepiece as spm
        import utils as beit3_utils
        from timm.models import create_model

        self._beit3_sp = spm.SentencePieceProcessor()
        self._beit3_sp.Load(str(spm_path))
        self._beit3_bos, self._beit3_pad, self._beit3_eos, self._beit3_unk = 0, 1, 2, 3
        self._beit3_offset = 1

        beit3_model = create_model(
            "beit3_base_patch16_384_retrieval", pretrained=False, drop_path_rate=0.0,
            vocab_size=64010, checkpoint_activations=None,
        )
        beit3_utils.load_model_and_may_interpolate(
            ckpt_path=str(ckpt_path), model=beit3_model, model_key="model|module", model_prefix=""
        )
        self._beit3_model = beit3_model.eval()

    def encode_siglip_text(self, text: str) -> list[float]:
        import torch

        assert self._siglip_model is not None, (
            "SigLIP2 chưa được nạp local - đặt AIC_LOCAL_QUERY_ENCODER_SIGLIP=1 trước khi gọi."
        )
        inputs = self._siglip_processor(text=[text], return_tensors="pt", padding="max_length", truncation=True)
        with torch.no_grad():
            feats = self._siglip_model.get_text_features(**inputs)
        if hasattr(feats, "pooler_output"):
            feats = feats.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].numpy().tolist()

    def encode_pe_core_text(self, text: str) -> list[float]:
        import torch

        assert self._pe_model is not None, (
            "PE-Core chưa được nạp local - đặt AIC_LOCAL_QUERY_ENCODER_PE_CORE=1 trước khi gọi."
        )
        tokens = self._pe_tokenizer([text])
        with torch.no_grad():
            feats = self._pe_model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].float().numpy().tolist()

    def encode_beit3_text(self, text: str) -> list[float]:
        import torch

        assert self._beit3_model is not None, (
            "BEiT-3 chưa được nạp local - đặt AIC_LOCAL_QUERY_ENCODER_BEIT3=1 trước khi gọi."
        )
        raw_ids = self._beit3_sp.EncodeAsIds(text)
        ids = [(rid + self._beit3_offset) if rid != 0 else self._beit3_unk for rid in raw_ids]
        if len(ids) > BEIT3_MAX_TEXT_LEN - 2:
            ids = ids[: BEIT3_MAX_TEXT_LEN - 2]
        ids = [self._beit3_bos] + ids + [self._beit3_eos]
        n = len(ids)
        mask = [0] * n + [1] * (BEIT3_MAX_TEXT_LEN - n)
        ids = ids + [self._beit3_pad] * (BEIT3_MAX_TEXT_LEN - n)

        text_ids = torch.tensor([ids], dtype=torch.long)
        padding_mask = torch.tensor([mask], dtype=torch.bool)
        with torch.no_grad():
            _, language_cls = self._beit3_model(text_description=text_ids, padding_mask=padding_mask, only_infer=True)
        return language_cls[0].float().numpy().tolist()


@lru_cache(maxsize=1)
def _local_encoder() -> _LocalQueryEncoder:
    # singleton DUY NHAT (khong phai 3 cai rieng) - 3 model CO THE dung chung 1 instance vi moi
    # model chi nap khi flag rieng cua no BAT (xem __init__) - goi lai ham nay (cache) SAU KHI
    # bat them 1 flag moi trong CUNG session se KHONG nap bo sung (lru_cache giu instance CU) -
    # can restart process neu doi flag giua chung, giong het gioi han cu cua AIC_LOCAL_MODELS.
    return _LocalQueryEncoder()


# 2026-08-20 (theo yeu cau nguoi dung: "hơn 1 phút nhưng chưa hết exception nào được raise" -
# sau khi da them timeout cho NIM, xac nhan diem treo THAT SU la o day - .remote() KHONG co
# timeout, co the treo VO HAN neu container Modal cold-start qua lau/mang loi) - dung
# call_modal_with_timeout() (spawn+get(timeout=...), xem app_flags.py) thay vi .remote() truc
# tiep, nem ModalTimeoutError ro rang thay vi treo.
def encode_text_siglip(text: str) -> np.ndarray:
    vec = (_local_encoder().encode_siglip_text(text) if _local_mode_for("siglip") else
           call_modal_with_timeout(_remote_encoder().encode_siglip_text, text, context="encode SigLIP2",
                                    local_env_hint="AIC_LOCAL_QUERY_ENCODER_SIGLIP"))
    return np.asarray([vec], dtype=np.float32)


def encode_text_pe_core(text: str) -> np.ndarray:
    vec = (_local_encoder().encode_pe_core_text(text) if _local_mode_for("pe_core") else
           call_modal_with_timeout(_remote_encoder().encode_pe_core_text, text, context="encode PE-Core",
                                    local_env_hint="AIC_LOCAL_QUERY_ENCODER_PE_CORE"))
    return np.asarray([vec], dtype=np.float32)


def encode_text_beit3(text: str) -> np.ndarray:
    vec = (_local_encoder().encode_beit3_text(text) if _local_mode_for("beit3") else
           call_modal_with_timeout(_remote_encoder().encode_beit3_text, text, context="encode BEiT-3",
                                    local_env_hint="AIC_LOCAL_QUERY_ENCODER_BEIT3"))
    return np.asarray([vec], dtype=np.float32)


ENCODERS = {
    "siglip": encode_text_siglip,
    "pe_core": encode_text_pe_core,
    "beit3": encode_text_beit3,
}
