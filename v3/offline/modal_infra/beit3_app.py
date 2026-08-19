"""Modal app cho BEiT-3 (Microsoft, "Image as a Foreign Language", CVPR 2023) - model
image-text retrieval, ung vien so sanh them voi SigLIP2/PE-Core (xem hoi thoai 2026-08-13).

Dung ban "base", da fine-tune SAN cho retrieval tren COCO (384px) -
beit3_base_patch16_384_coco_retrieval.pth - khong dung ban pretrain thuan (khong toi uu cho
similarity image-text truc tiep).

Code kien truc KHONG co goi pip - phai clone thang tu github.com/microsoft/unilm (thu muc
beit3), dung chung pattern voi pe_core_app.py.

BUG DA BIET (kiem tra truoc, 2026-08-13): utils.py cua repo nay `from torch._six import inf` -
torch._six DA BI XOA khoi PyTorch tu ban 2.0+ - PHAI monkeypatch torch._six GIA truoc khi
import module cua repo, neu khong crash ngay luc import.

Checkpoint + beit3.spm (sentencepiece tokenizer) tai truc tiep tu GitHub Releases
(addf400/files), cache vao chung Volume aic2026-hf-cache (thu muc con /beit3) de khong tai lai
moi lan cold-start.

Chay thu (dev): modal serve beit3_app.py
Deploy that:    modal deploy beit3_app.py
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch", "torchvision", "pillow", "numpy", "einops", "sentencepiece",
        "ftfy", "timm==0.4.12", "fairscale==0.4.0", "torchscale==0.2.0",
        "tensorboardX", "torchmetrics", "requests",
        # KHONG dung transformers.XLMRobertaTokenizer nua (xem docstring duoi) -> bo
        # "transformers" khoi dependency, dung thang sentencepiece cho gon + on dinh hon.
        # BUG THAT (2026-08-13): doc setup.py tren GitHub cua torchscale tuong la
        # timm==0.6.13, nhung ban torchscale==0.2.0 THAT tren PyPI pin cung timm==0.4.12
        # (giong dung requirements.txt goc cua beit3) - GitHub main branch da update, khac
        # ban da publish. Dung dung timm==0.4.12 nhu pip resolver bao, khong doan mo.
    )
    .run_commands("git clone --depth 1 https://github.com/microsoft/unilm.git /root/unilm")
)

hf_cache_vol = modal.Volume.from_name("aic2026-hf-cache", create_if_missing=True)  # dung chung

app = modal.App("aic2026-beit3")

CKPT_URL = "https://github.com/addf400/files/releases/download/beit3/beit3_base_patch16_384_coco_retrieval.pth"
SPM_URL = "https://github.com/addf400/files/releases/download/beit3/beit3.spm"
CACHE_DIR = "/root/.cache/huggingface/beit3"
IMG_SIZE = 384
MAX_TEXT_LEN = 64  # num_max_bpe_tokens mac dinh cua RetrievalDataset trong repo goc


@app.cls(
    image=image,
    gpu="A10G",
    scaledown_window=5 * 60,
    timeout=1800,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
)
class BEiT3Encoder:
    @modal.enter()
    def load(self):
        import os
        import sys
        import types

        import requests
        import torch

        os.makedirs(CACHE_DIR, exist_ok=True)
        ckpt_path = os.path.join(CACHE_DIR, "beit3_base_patch16_384_coco_retrieval.pth")
        spm_path = os.path.join(CACHE_DIR, "beit3.spm")
        for url, path in ((CKPT_URL, ckpt_path), (SPM_URL, spm_path)):
            if not os.path.exists(path):
                print(f"tai {url} -> {path}", flush=True)
                r = requests.get(url, stream=True, timeout=300)
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)

        # BUG THAT (2026-08-13, xem docstring dau file): torch._six da bi xoa khoi PyTorch
        # 2.0+, nhung beit3/utils.py van `from torch._six import inf` - monkeypatch GIA truoc
        # khi import module cua repo, neu khong ImportError ngay.
        if not hasattr(torch, "_six"):
            six_mod = types.ModuleType("torch._six")
            six_mod.inf = float("inf")
            sys.modules["torch._six"] = six_mod

        sys.path.insert(0, "/root/unilm/beit3")
        import modeling_finetune  # noqa: F401  - dang ky beit3_base_patch16_384_retrieval qua @register_model
        import utils
        import sentencepiece as spm
        from timm.models import create_model
        from torchvision import transforms

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # BUG THAT (2026-08-13): transformers.XLMRobertaTokenizer(spm_path) crash luc load
        # ("Can't extract `str` to `Vec`") - ban transformers moi (5.15.0) khong tuong thich
        # cach nap sentencepiece file tho kieu nay. SUA: dung thang sentencepiece +
        # quy uoc offset CHUAN cua XLM-R/fairseq (giong het HF cai dat trong ban):
        #   id 0=<s>(bos) 1=<pad> 2=</s>(eos) 3=<unk>, token thuong = spm_id + 1
        #   (spm tu danh id 0 cho <unk> rieng, bi thay the boi unk_token_id=3 o day).
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(spm_path)
        self.bos_token_id, self.pad_token_id, self.eos_token_id, self.unk_token_id = 0, 1, 2, 3
        self._fairseq_offset = 1

        model = create_model(
            "beit3_base_patch16_384_retrieval",
            pretrained=False,
            drop_path_rate=0.0,
            vocab_size=64010,
            checkpoint_activations=None,
        )
        utils.load_model_and_may_interpolate(
            ckpt_path=ckpt_path, model=model, model_key="model|module", model_prefix=""
        )
        self.model = model.to(self.device).eval()

        # eval transform (is_train=False) dung DUNG mean/std goc cua repo (Inception, khong
        # phai ImageNet thuong) - xem datasets.py::build_transform.
        from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
        ])

    @modal.method()
    def encode_images(self, images_bytes: list[bytes]) -> list[list[float]]:
        import io

        import torch
        from PIL import Image

        imgs = torch.stack([
            self.transform(Image.open(io.BytesIO(b)).convert("RGB")) for b in images_bytes
        ]).to(self.device)
        with torch.no_grad():
            vision_cls, _ = self.model(image=imgs, only_infer=True)
        return vision_cls.float().cpu().numpy().tolist()

    @modal.method()
    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        import torch

        # tu ma hoa qua sentencepiece tho + offset chuan XLM-R (xem ghi chu load()) thay vi
        # tokenizer.encode() cua transformers.
        batch_tokens, batch_mask = [], []
        for t in texts:
            raw_ids = self._sp.EncodeAsIds(t)
            ids = [
                (rid + self._fairseq_offset) if rid != 0 else self.unk_token_id
                for rid in raw_ids
            ]
            if len(ids) > MAX_TEXT_LEN - 2:
                ids = ids[: MAX_TEXT_LEN - 2]
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
            n = len(ids)
            mask = [0] * n + [1] * (MAX_TEXT_LEN - n)
            ids = ids + [self.pad_token_id] * (MAX_TEXT_LEN - n)
            batch_tokens.append(ids)
            batch_mask.append(mask)

        text_ids = torch.tensor(batch_tokens, dtype=torch.long, device=self.device)
        padding_mask = torch.tensor(batch_mask, dtype=torch.bool, device=self.device)
        with torch.no_grad():
            _, language_cls = self.model(
                text_description=text_ids, padding_mask=padding_mask, only_infer=True
            )
        return language_cls.float().cpu().numpy().tolist()


@app.local_entrypoint()
def test():
    import urllib.request

    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    img_bytes = urllib.request.urlopen(url).read()
    enc = BEiT3Encoder()
    iv = enc.encode_images.remote([img_bytes])
    tv = enc.encode_texts.remote(["a cat"])
    print("img vec dim:", len(iv[0]), "text vec dim:", len(tv[0]))
