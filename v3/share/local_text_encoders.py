"""Encode TEXT (câu truy vấn) cho 3 model dense (SigLIP2/PE-Core/BEiT-3).

2 CHẾ ĐỘ, chọn qua biến môi trường AIC_LOCAL_MODELS (mặc định TẮT — giữ nguyên hành vi hiện tại):
  - AIC_LOCAL_MODELS không set / "0": gọi .remote() tới Modal app nhẹ luôn giữ ấm
    (aic2026-query-encoders, min_containers=1, xem offline/modal_infra/query_encoders_app.py).
  - AIC_LOCAL_MODELS=1: load CẢ 3 MODEL TRỰC TIẾP trên máy (README.md — "Chạy model local thay
    vì Modal"). Nặng hơn (RAM/VRAM + tải model lần đầu) nhưng không cần tài khoản Modal/mạng ổn
    định cho từng query — dùng đúng LẠI code load()/encode_*_text() của query_encoders_app.py,
    chỉ bỏ phần bọc @app.cls/@modal.method (chạy in-process, không qua Modal container).

Giữ NGUYÊN interface cũ (ENCODERS dict, trả về np.ndarray shape (1, dim)) — dense_search.py
KHÔNG cần sửa gì (vẫn gọi ENCODERS[model](query) y hệt trước, bất kể chế độ nào)."""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

MODAL_APP_NAME = "aic2026-query-encoders"
MODAL_CLASS_NAME = "QueryEncoder"

SIGLIP_MODEL_NAME = "google/siglip2-base-patch16-224"
PE_CORE_MODEL_NAME = "PE-Core-B16-224"
BEIT3_CKPT_URL = "https://github.com/addf400/files/releases/download/beit3/beit3_base_patch16_384_coco_retrieval.pth"
BEIT3_SPM_URL = "https://github.com/addf400/files/releases/download/beit3/beit3.spm"
BEIT3_MAX_TEXT_LEN = 64


def _local_mode() -> bool:
    return os.environ.get("AIC_LOCAL_MODELS", "0").strip().lower() in ("1", "true", "yes")


# ============================================================ Modal (mac dinh)
@lru_cache(maxsize=1)
def _remote_encoder():
    import modal

    Encoder = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)
    return Encoder()


# ============================================================ Local (AIC_LOCAL_MODELS=1)
class _LocalQueryEncoder:
    """Y HET query_encoders_app.py::QueryEncoder.load()/encode_*_text(), chi bo wrapper Modal.
    Xem README.md muc "Xung dot thu vien" cho ly do timm==0.4.12 + shim timm.layers, torch._six,
    perception_models/unilm phai clone tay (khong co tren PyPI)."""

    def __init__(self) -> None:
        import sys
        import types

        import requests
        import torch

        from config import _V3_ROOT

        os.environ.setdefault("HF_HOME", str(_V3_ROOT / ".cache" / "huggingface"))
        cache_root = _V3_ROOT / ".cache"

        # ---- SigLIP2 ----
        from transformers import AutoModel, AutoProcessor

        self._siglip_model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME).eval()
        self._siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)

        # ---- PE-Core ----
        # BUG THAT (xem query_encoders_app.py): timm==0.4.12 (BAT BUOC cho BEiT-3/torchscale==
        # 0.2.0) KHONG co submodule "timm.layers" (chi tu ban timm moi hon), nhung PE-Core's
        # pe.py lai `from timm.layers import DropPath`. Shim "timm.layers" -> "timm.models.layers"
        # (duong dan CU, cung class, chi khac ten module) TRUOC khi import pe.py.
        perception_models_dir = cache_root / "perception_models"
        if not perception_models_dir.exists():
            raise RuntimeError(
                f"Chưa clone perception_models — xem README.md mục setup PE-Core:\n"
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

        unilm_dir = cache_root / "unilm"
        if not unilm_dir.exists():
            raise RuntimeError(
                f"Chưa clone unilm — xem README.md mục setup BEiT-3:\n"
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

        inputs = self._siglip_processor(text=[text], return_tensors="pt", padding="max_length", truncation=True)
        with torch.no_grad():
            feats = self._siglip_model.get_text_features(**inputs)
        if hasattr(feats, "pooler_output"):
            feats = feats.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].numpy().tolist()

    def encode_pe_core_text(self, text: str) -> list[float]:
        import torch

        tokens = self._pe_tokenizer([text])
        with torch.no_grad():
            feats = self._pe_model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].float().numpy().tolist()

    def encode_beit3_text(self, text: str) -> list[float]:
        import torch

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
    return _LocalQueryEncoder()  # nap 3 model, chi 1 LAN (module-level cache) - cham lan dau


def encode_text_siglip(text: str) -> np.ndarray:
    vec = _local_encoder().encode_siglip_text(text) if _local_mode() else _remote_encoder().encode_siglip_text.remote(text)
    return np.asarray([vec], dtype=np.float32)


def encode_text_pe_core(text: str) -> np.ndarray:
    vec = _local_encoder().encode_pe_core_text(text) if _local_mode() else _remote_encoder().encode_pe_core_text.remote(text)
    return np.asarray([vec], dtype=np.float32)


def encode_text_beit3(text: str) -> np.ndarray:
    vec = _local_encoder().encode_beit3_text(text) if _local_mode() else _remote_encoder().encode_beit3_text.remote(text)
    return np.asarray([vec], dtype=np.float32)


ENCODERS = {
    "siglip": encode_text_siglip,
    "pe_core": encode_text_pe_core,
    "beit3": encode_text_beit3,
}
