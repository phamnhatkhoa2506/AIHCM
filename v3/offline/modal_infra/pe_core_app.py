"""Modal app cho Perception Encoder (PE-Core, Meta) - khong co tich hop transformers chuan,
phai clone repo + pip install -e . (xem facebookresearch/perception_models). Checkpoint tai
tu HuggingFace (facebook/PE-Core-B16-224) qua pretrained=True cua thu vien.

Chay thu (dev): modal serve pe_core_app.py
Deploy that:    modal deploy pe_core_app.py
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "torchvision", "pillow", "ftfy", "regex", "timm", "huggingface_hub", "einops")
    .run_commands(
        "git clone --depth 1 https://github.com/facebookresearch/perception_models.git /root/perception_models",
        "cd /root/perception_models && pip install -e . --no-deps",
    )
)

hf_cache_vol = modal.Volume.from_name("aic2026-hf-cache", create_if_missing=True)

app = modal.App("aic2026-pe-core")

MODEL_NAME = "PE-Core-B16-224"


@app.cls(
    image=image,
    gpu="A10G",
    scaledown_window=5 * 60,
    timeout=1800,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
)
class PECoreEncoder:
    @modal.enter()
    def load(self):
        import os
        import sys

        os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
        sys.path.insert(0, "/root/perception_models")

        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms

        self.model = pe.CLIP.from_config(MODEL_NAME, pretrained=True).cuda().eval()
        self.preprocess = transforms.get_image_transform(self.model.image_size)
        self.tokenizer = transforms.get_text_tokenizer(self.model.context_length)

    @modal.method()
    def encode_images(self, images_bytes: list[bytes]) -> list[list[float]]:
        import io

        import torch
        from PIL import Image

        imgs = torch.stack([
            self.preprocess(Image.open(io.BytesIO(b)).convert("RGB")) for b in images_bytes
        ]).cuda()
        with torch.no_grad(), torch.autocast("cuda"):
            feats = self.model.encode_image(imgs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float().cpu().numpy().tolist()

    @modal.method()
    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        import torch

        tokens = self.tokenizer(texts).cuda()
        with torch.no_grad(), torch.autocast("cuda"):
            feats = self.model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float().cpu().numpy().tolist()


@app.local_entrypoint()
def test():
    import urllib.request

    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    img_bytes = urllib.request.urlopen(url).read()
    enc = PECoreEncoder()
    iv = enc.encode_images.remote([img_bytes])
    tv = enc.encode_texts.remote(["a cat"])
    print("img vec dim:", len(iv[0]), "text vec dim:", len(tv[0]))
