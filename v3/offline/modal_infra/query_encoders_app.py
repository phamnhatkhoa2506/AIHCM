"""Modal app NHE, CPU-only, LUON GIU AM (min_containers=1) - encode QUERY TEXT (ngan, 1 cau)
cho ca 3 model dense (SigLIP2/PE-Core/BEiT-3), thay cho viec load ca 3 model TRUC TIEP tren
may local (share/local_text_encoders.py) - da gay lag may (2026-08-15, theo yeu cau nguoi dung
"host nhe 3 mo hinh embedding do do chay tren may minh cho do lag").

KHAC 3 app rieng (siglip_app.py/pe_core_app.py/beit3_app.py, dung GPU A10G cho encode CORPUS
hang loat offline): app nay CHI encode 1 cau text/lan (query online), khong can GPU - CPU du
nhanh (<1s/cau), va min_containers=1 giu 1 container AM SAN de tranh cold-start (~10-30s neu
phai khoi dong lai tu dau, dung nguoc lai muc dich "do lag").

3 model dung LAI DUNG code load() tu 3 app GPU (bo .cuda()/torch.autocast("cuda") vi chay CPU).

Deploy: modal deploy query_encoders_app.py
Goi tu: share/local_text_encoders.py (thay the ham load model local bang .remote() toi day)
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch", "torchvision", "transformers", "pillow", "accelerate",
        "ftfy", "regex", "timm==0.4.12", "huggingface_hub", "einops",
        "sentencepiece", "fairscale==0.4.0", "torchscale==0.2.0", "tensorboardX", "torchmetrics", "requests",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/facebookresearch/perception_models.git /root/perception_models",
        "cd /root/perception_models && pip install -e . --no-deps",
        "git clone --depth 1 https://github.com/microsoft/unilm.git /root/unilm",
    )
)

hf_cache_vol = modal.Volume.from_name("aic2026-hf-cache", create_if_missing=True)  # dung chung
app = modal.App("aic2026-query-encoders")

SIGLIP_MODEL_NAME = "google/siglip2-base-patch16-224"
PE_CORE_MODEL_NAME = "PE-Core-B16-224"
BEIT3_CKPT_URL = "https://github.com/addf400/files/releases/download/beit3/beit3_base_patch16_384_coco_retrieval.pth"
BEIT3_SPM_URL = "https://github.com/addf400/files/releases/download/beit3/beit3.spm"
BEIT3_CACHE_DIR = "/root/.cache/huggingface/beit3"
BEIT3_MAX_TEXT_LEN = 64


@app.cls(
    image=image,
    # KHONG khai bao gpu= -> mac dinh CPU (du cho encode 1 cau text ngan, xem docstring dau file).
    min_containers=1,  # LUON giu 1 container am - muc tieu chinh cua app nay (tranh cold-start).
    scaledown_window=30 * 60,
    timeout=120,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
)
# 2026-08-20 (theo yeu cau nguoi dung, phat hien qua bug that: "chạy nhiều lần cho hết cold-
# start nhưng vẫn chiếm nhiều thời gian" - pe_core 6.15s dù goi rieng le CHI ~0.8-1.8s) - MAC
# DINH 1 container Modal CHI xu ly 1 request/lan (input tiep theo phai XEP HANG cho xong cai
# truoc), dù code local goi .remote() "song song" qua ThreadPoolExecutor (_rank_rrf trong
# dense_search.py) - 2 model (siglip/pe_core) CUNG goi vao 1 container QueryEncoder nay, request
# thu 2 luon phai doi request dau xong -> "song song" tren code nhung TUAN TU tren server that.
# @modal.concurrent cho phep 1 container nhan NHIEU request dong thoi that su (an toan: ca 3 ham
# encode_*_text deu CHI DOC self._model, khong ghi/doi state gi giua cac call).
@modal.concurrent(max_inputs=8)
class QueryEncoder:
    @modal.enter()
    def load(self):
        import os
        import sys
        import types

        import requests
        import torch

        os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")

        # ---- SigLIP2 ----
        from transformers import AutoModel, AutoProcessor
        self._siglip_model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME).eval()
        self._siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)

        # ---- PE-Core ----
        # BUG THAT (2026-08-15): timm==0.4.12 (BAT BUOC cho BEiT-3/torchscale==0.2.0, xem duoi)
        # KHONG co submodule "timm.layers" (chi co tu ban timm moi hon) - nhung PE-Core's
        # pe.py lai `from timm.layers import DropPath`. 2 model KHONG the dung chung 1 ban
        # timm neu import thang. Fix: shim "timm.layers" -> "timm.models.layers" (duong dan
        # CU, co DUNG cac class giong het, chi doi ten module) TRUOC khi import pe.py.
        sys.path.insert(0, "/root/perception_models")
        import timm.models.layers as _timm_layers_legacy
        sys.modules["timm.layers"] = _timm_layers_legacy

        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as pe_transforms
        self._pe_model = pe.CLIP.from_config(PE_CORE_MODEL_NAME, pretrained=True).eval()
        self._pe_tokenizer = pe_transforms.get_text_tokenizer(self._pe_model.context_length)

        # ---- BEiT-3 (xem beit3_app.py - cung 1 bug torch._six + sentencepiece thu cong) ----
        os.makedirs(BEIT3_CACHE_DIR, exist_ok=True)
        ckpt_path = os.path.join(BEIT3_CACHE_DIR, "beit3_base_patch16_384_coco_retrieval.pth")
        spm_path = os.path.join(BEIT3_CACHE_DIR, "beit3.spm")
        for url, path in ((BEIT3_CKPT_URL, ckpt_path), (BEIT3_SPM_URL, spm_path)):
            if not os.path.exists(path):
                r = requests.get(url, stream=True, timeout=300)
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)

        if not hasattr(torch, "_six"):
            six_mod = types.ModuleType("torch._six")
            six_mod.inf = float("inf")
            sys.modules["torch._six"] = six_mod

        sys.path.insert(0, "/root/unilm/beit3")
        import modeling_finetune  # noqa: F401 - dang ky beit3_base_patch16_384_retrieval
        import sentencepiece as spm
        import utils as beit3_utils
        from timm.models import create_model

        self._beit3_sp = spm.SentencePieceProcessor()
        self._beit3_sp.Load(spm_path)
        self._beit3_bos, self._beit3_pad, self._beit3_eos, self._beit3_unk = 0, 1, 2, 3
        self._beit3_offset = 1

        beit3_model = create_model(
            "beit3_base_patch16_384_retrieval", pretrained=False, drop_path_rate=0.0,
            vocab_size=64010, checkpoint_activations=None,
        )
        beit3_utils.load_model_and_may_interpolate(
            ckpt_path=ckpt_path, model=beit3_model, model_key="model|module", model_prefix=""
        )
        self._beit3_model = beit3_model.eval()

    @modal.method()
    def encode_siglip_text(self, text: str) -> list[float]:
        import torch

        inputs = self._siglip_processor(text=[text], return_tensors="pt", padding="max_length", truncation=True)
        with torch.no_grad():
            feats = self._siglip_model.get_text_features(**inputs)
        if hasattr(feats, "pooler_output"):
            feats = feats.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].numpy().tolist()

    @modal.method()
    def encode_pe_core_text(self, text: str) -> list[float]:
        import torch

        tokens = self._pe_tokenizer([text])
        with torch.no_grad():
            feats = self._pe_model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].float().numpy().tolist()

    @modal.method()
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


@app.local_entrypoint()
def test():
    enc = QueryEncoder()
    for name, fn in [("siglip", enc.encode_siglip_text), ("pe_core", enc.encode_pe_core_text), ("beit3", enc.encode_beit3_text)]:
        v = fn.remote("a cat")
        print(f"{name}: dim={len(v)}")
