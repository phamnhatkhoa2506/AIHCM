"""Encode TEXT (cau truy van) LOCAL cho 3 model dense (SigLIP2/PE-Core/BEiT-3) - GIONG HET
pattern tier2_vector.py::encode_query() dang dung cho CLIP hien tai (model text nho, encode
1 cau query khong can GPU/Modal, chi anh CORPUS moi can Modal GPU vi khoi luong lon).

Tranh han che "app not found"/Modal app bi stop giua chung ma dense_search.py gap phai khi
goi remote() lien tuc cho tung cau query - encode query xong CHAY 1 LAN, cache lai (lazy
singleton), khong con phu thuoc Modal cho duong online nay nua.

Model + checkpoint code THAM CHIEU dung y het cac Modal app tuong ung (siglip_app.py,
pe_core_app.py, beit3_app.py) - chi khac la load() 1 lan local (CPU, torch.cuda.is_available()
False tren may nay) thay vi trong container GPU.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from config import MODEL_CACHE_DIR

# BUG THAT (2026-08-14): SigLIP2 tai ve C:\Users\<user>\.cache\huggingface mac dinh thay vi
# v3/.cache/ nhu resources.py dang lam cho CLIP - vi module nay chua tung set HF_HOME truoc
# khi transformers duoc import lan dau (resources.py lam dieu nay o muc module-level nhung
# module do co the CHUA duoc import truoc local_text_encoders). Set NGAY DAY, truoc moi
# import transformers/huggingface_hub o duoi file, de nhat quan 1 noi luu checkpoint.
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(MODEL_CACHE_DIR / "hub"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

_V3_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_REPOS_DIR = _V3_ROOT / ".cache" / "local_repos"
_LOCAL_REPOS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================ SigLIP2
@lru_cache(maxsize=1)
def _load_siglip():
    from transformers import AutoModel, AutoProcessor

    model_name = "google/siglip2-base-patch16-224"
    model = AutoModel.from_pretrained(model_name).eval()
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def encode_text_siglip(text: str) -> np.ndarray:
    import torch

    model, processor = _load_siglip()
    inputs = processor(text=[text], return_tensors="pt", padding="max_length", truncation=True)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    if hasattr(feats, "pooler_output"):
        feats = feats.pooler_output
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.numpy().astype(np.float32)


# ============================================================ PE-Core
def _ensure_pe_core_repo() -> Path:
    repo_dir = _LOCAL_REPOS_DIR / "perception_models"
    if not (repo_dir / "core").exists():
        import subprocess

        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/facebookresearch/perception_models.git", str(repo_dir)],
            check=True,
        )
    return repo_dir


@lru_cache(maxsize=1)
def _load_pe_core():
    repo_dir = _ensure_pe_core_repo()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))

    import core.vision_encoder.pe as pe
    import core.vision_encoder.transforms as transforms

    model = pe.CLIP.from_config("PE-Core-B16-224", pretrained=True).eval()
    tokenizer = transforms.get_text_tokenizer(model.context_length)
    return model, tokenizer


def encode_text_pe_core(text: str) -> np.ndarray:
    import torch

    model, tokenizer = _load_pe_core()
    tokens = tokenizer([text])
    with torch.no_grad():
        feats = model.encode_text(tokens)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.float().numpy().astype(np.float32)


# ============================================================ BEiT-3
_BEIT3_CKPT_URL = "https://github.com/addf400/files/releases/download/beit3/beit3_base_patch16_384_coco_retrieval.pth"
_BEIT3_SPM_URL = "https://github.com/addf400/files/releases/download/beit3/beit3.spm"
_BEIT3_MAX_TEXT_LEN = 64


def _ensure_beit3_repo() -> Path:
    repo_dir = _LOCAL_REPOS_DIR / "unilm"
    if not (repo_dir / "beit3" / "modeling_finetune.py").exists():
        import subprocess

        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/microsoft/unilm.git", str(repo_dir)],
            check=True,
        )
    return repo_dir / "beit3"


def _ensure_beit3_weights() -> tuple[Path, Path]:
    import requests

    cache_dir = MODEL_CACHE_DIR / "beit3"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cache_dir / "beit3_base_patch16_384_coco_retrieval.pth"
    spm_path = cache_dir / "beit3.spm"
    for url, path in ((_BEIT3_CKPT_URL, ckpt_path), (_BEIT3_SPM_URL, spm_path)):
        if not path.exists():
            r = requests.get(url, stream=True, timeout=300)
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
    return ckpt_path, spm_path


@lru_cache(maxsize=1)
def _load_beit3():
    import types

    import torch

    # BUG THAT (xem beit3_app.py) - torch._six bi xoa khoi PyTorch 2.0+, beit3/utils.py van
    # `from torch._six import inf` - monkeypatch GIA truoc khi import module cua repo.
    if not hasattr(torch, "_six"):
        six_mod = types.ModuleType("torch._six")
        six_mod.inf = float("inf")
        sys.modules["torch._six"] = six_mod

    beit3_dir = _ensure_beit3_repo()
    if str(beit3_dir) not in sys.path:
        sys.path.insert(0, str(beit3_dir))
    ckpt_path, spm_path = _ensure_beit3_weights()

    import modeling_finetune  # noqa: F401 - dang ky beit3_base_patch16_384_retrieval
    import sentencepiece as spm
    import utils
    from timm.models import create_model

    model = create_model(
        "beit3_base_patch16_384_retrieval", pretrained=False, drop_path_rate=0.0,
        vocab_size=64010, checkpoint_activations=None,
    )
    utils.load_model_and_may_interpolate(
        ckpt_path=str(ckpt_path), model=model, model_key="model|module", model_prefix=""
    )
    model = model.eval()

    sp = spm.SentencePieceProcessor()
    sp.Load(str(spm_path))
    return model, sp


def encode_text_beit3(text: str) -> np.ndarray:
    import torch

    model, sp = _load_beit3()
    bos_id, pad_id, eos_id, unk_id, offset = 0, 1, 2, 3, 1

    raw_ids = sp.EncodeAsIds(text)
    ids = [(rid + offset) if rid != 0 else unk_id for rid in raw_ids]
    if len(ids) > _BEIT3_MAX_TEXT_LEN - 2:
        ids = ids[: _BEIT3_MAX_TEXT_LEN - 2]
    ids = [bos_id] + ids + [eos_id]
    n = len(ids)
    mask = [0] * n + [1] * (_BEIT3_MAX_TEXT_LEN - n)
    ids = ids + [pad_id] * (_BEIT3_MAX_TEXT_LEN - n)

    text_ids = torch.tensor([ids], dtype=torch.long)
    padding_mask = torch.tensor([mask], dtype=torch.bool)
    with torch.no_grad():
        _, language_cls = model(text_description=text_ids, padding_mask=padding_mask, only_infer=True)
    return language_cls.float().numpy().astype(np.float32)


ENCODERS = {
    "siglip": encode_text_siglip,
    "pe_core": encode_text_pe_core,
    "beit3": encode_text_beit3,
}
