"""Modal app cho SigLIP (Google) - model image-text ung vien thay the/bo sung CLIP hien tai
(xem hoi thoai 2026-08-11 - VISIONE dung nhieu model ensemble, dang thu SigLIP truoc).

SigLIP2 co ho tro multilingual, dung ban base multilingual de tuong thich tieng Viet.

Chay thu (dev): modal serve siglip_app.py
Deploy that:    modal deploy siglip_app.py
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "pillow", "accelerate")
)

hf_cache_vol = modal.Volume.from_name("aic2026-hf-cache", create_if_missing=True)

app = modal.App("aic2026-siglip")

MODEL_NAME = "google/siglip2-base-patch16-224"  # SigLIP2 da tu no da-ngon-ngu (khong co bien
# the "-multilingual" rieng - bug that phat hien 2026-08-11: doan sai ten model, 401/404 that


@app.cls(
    image=image,
    gpu="A10G",
    scaledown_window=5 * 60,
    timeout=1800,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
)
class SigLIPEncoder:
    @modal.enter()
    def load(self):
        import os

        import torch
        from transformers import AutoModel, AutoProcessor

        os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)

    @modal.method()
    def encode_images(self, images_bytes: list[bytes]) -> list[list[float]]:
        import io

        import torch
        from PIL import Image

        imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in images_bytes]
        inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)
        # BUG THAT (2026-08-11): ban transformers nay, get_image_features() tra ve
        # BaseModelOutputWithPooling (object), KHONG PHAI tensor thang nhu CLIPModel - phai
        # lay .pooler_output rieng.
        if hasattr(feats, "pooler_output"):
            feats = feats.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().tolist()

    @modal.method()
    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        import torch

        inputs = self.processor(
            text=texts, return_tensors="pt", padding="max_length", truncation=True
        ).to(self.device)
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        if hasattr(feats, "pooler_output"):
            feats = feats.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().tolist()


@app.local_entrypoint()
def test():
    import urllib.request

    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    img_bytes = urllib.request.urlopen(url).read()
    enc = SigLIPEncoder()
    iv = enc.encode_images.remote([img_bytes])
    tv = enc.encode_texts.remote(["a cat"])
    print("img vec dim:", len(iv[0]), "text vec dim:", len(tv[0]))
