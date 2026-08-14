"""Modal app cho OWLv2 (Google, 2023) - closed-set detector cho bo keyframe DENSE, dung DUNG
514 nhan OpenImages da co san anh xa tieng Viet (index/label_vi.json, tai su dung nguyen
label_translate.py - khong can dich lai gi ca).

Vi sao OWLv2 thay vi Grounding DINO cho viec nay (xem hoi thoai 2026-08-14): Grounding DINO
ghep CA CAU thanh 1 prompt -> gioi han token text encoder (~256), khong the nhet 514 nhan
mot luc (loang do chu y, mat do chinh xac tung nhan). OWLv2 encode TUNG nhan RIENG thanh 1
embedding class roi so khop - kien truc nay von duoc benchmark voi vai tram-nghin lop cung
luc (LVIS ~1200 lop), khong bi van de do.

KHAC grounding_dino_app.py: OWLv2 la open-vocab THUAN (khong huan luyen rieng tren OpenImages)
nhung ep vocabulary CO DINH = 514 nhan closed-set -> ket qua tuong duong "closed-set detector"
ve mat chuc nang (nhan tra ve luon nam trong 514 nhan da biet, tuong thich truc tiep voi
objects_index.parquet/label_translate.py hien co).

Chay thu (dev): modal serve owlv2_app.py
Deploy that:    modal deploy owlv2_app.py
"""
import modal

MODEL_NAME = "google/owlv2-base-patch16-ensemble"
# 2026-08-14: da chan doan + FIX nghen co chai LOCAL CPU bang Modal Volume (xem duoi) - gio
# Modal GPU container that su la yeu to gioi han toc do, tra ve 10 (toi da tai khoan cho phep).
MAX_CONTAINERS = 10

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "transformers", "pillow", "accelerate")
)

hf_cache_vol = modal.Volume.from_name("aic2026-hf-cache", create_if_missing=True)
# 2026-08-14: da chan doan nghen co chai la LOCAL CPU (doc/giai nen + gui bytes anh qua mang) -
# KHONG PHAI Modal GPU. Giai phap: upload san toan bo dense keyframes len Volume nay 1 lan
# (offline/upload_dense_to_volume.py), container GPU tu doc thang tu day - loai bo hoan toan
# nghen local. Xem share/dense_volume_map.py cho quy uoc duong dan.
dense_kf_vol = modal.Volume.from_name("aic2026-dense-keyframes", create_if_missing=True)
DENSE_VOLUME_MOUNT = "/dense_keyframes"

app = modal.App("aic2026-owlv2")


@app.cls(
    image=image,
    gpu="A10G",
    max_containers=MAX_CONTAINERS,
    scaledown_window=5 * 60,
    # BUG THAT (2026-08-14, lap lai nhieu lan): .map(order_outputs=True) bi CHAN boi dung 1
    # batch xu ly bat thuong lau (anh loi/kich thuoc la) - cac batch SAU du da xong van phai
    # cho DUNG THU TU moi tra ve, "dong bang" ca luong tien do. timeout=600 (10 phut) qua dai -
    # ha xuong 120s (du gap 60x thoi gian 1 batch binh thuong ~2s warm) de tu huy nhanh hon
    # nhieu neu that su treo, giam thoi gian "dong bang" toi da tu 10 phut xuong 2 phut.
    timeout=120,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        DENSE_VOLUME_MOUNT: dense_kf_vol,
    },
)
class OWLv2Detector:
    @modal.enter()
    def load(self):
        import os

        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
        self.processor = Owlv2Processor.from_pretrained(MODEL_NAME)
        self.model = Owlv2ForObjectDetection.from_pretrained(MODEL_NAME).eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # FP16 (2026-08-14, toc do 1000 anh/phut qua cham - nguoi dung yeu cau toi uu): giam
        # nua bo nho + tang toc tren A10G (tensor core toi uu fp16), rui ro thap cho detection
        # (khong nhay cam chinh xac so nhu OCR ky tu). Batch size lon hon KHONG giup (da do
        # thuc te: batch 32 cham hon batch 8, chi phi chinh la encode 514 nhan/anh, khong phai
        # overhead batch).
        self.use_fp16 = self.device == "cuda"
        if self.use_fp16:
            self.model = self.model.half()
        self.model.to(self.device)
        # dam bao thay du lieu moi nhat upload len Volume (container co the mount truoc khi
        # upload xong neu warm san tu deploy cu).
        dense_kf_vol.reload()

    @modal.method()
    def detect_batch(
        self, images_bytes: list[bytes], labels: list[str], threshold: float = 0.15
    ) -> list[list[dict]]:
        """labels: DANH SACH nhan dung CHUNG cho ca batch (vd 514 nhan closed-set). Tra ve list
        (dung thu tu images_bytes), moi phan tu la list[{"label","score","ymin","xmin","ymax","xmax"}]
        (bbox da normalize [0,1], giong ocr_app.py - khac grounding_dino_app.py tra px tho)."""
        import io

        import torch
        from PIL import Image

        images = [Image.open(io.BytesIO(b)).convert("RGB") for b in images_bytes]
        text_queries = [labels] * len(images)  # OWLv2 nhan list[list[str]] - 1 list nhan/anh

        inputs = self.processor(text=text_queries, images=images, return_tensors="pt").to(self.device)
        if self.use_fp16:
            # BUG SE GAP (2026-08-14): processor tra pixel_values float32 mac dinh - model da
            # chuyen half() -> can ep KHOP dtype, khong thi loi "Input type and weight type
            # should be the same". input_ids/attention_mask giu nguyen long/int, KHONG ep half.
            inputs["pixel_values"] = inputs["pixel_values"].half()
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([img.size[::-1] for img in images])  # (h, w) moi anh
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, threshold=threshold, target_sizes=target_sizes, text_labels=text_queries
        )

        out = []
        for res, img in zip(results, images):
            w, h = img.size
            items = []
            text_labels_out = res.get("text_labels") or res.get("labels")
            for label, score, box in zip(text_labels_out, res["scores"], res["boxes"]):
                xmin, ymin, xmax, ymax = [float(x) for x in box.tolist()]
                items.append({
                    "label": str(label),
                    "score": float(score),
                    "xmin": max(0.0, xmin / w), "ymin": max(0.0, ymin / h),
                    "xmax": min(1.0, xmax / w), "ymax": min(1.0, ymax / h),
                })
            out.append(items)
        return out

    @modal.method()
    def detect_batch_from_volume(
        self, rel_paths: list[str], labels: list[str], threshold: float = 0.15
    ) -> list[list[dict]]:
        """NHU detect_batch nhung doc anh THANG TU Modal Volume (khong nhan bytes qua tham so)
        - rel_paths la duong dan tuong doi trong Volume (xem share/dense_volume_map.py), vd
        "L21_extracted/L21/L21_V009/shot0214_f0020910.jpg". Loai bo hoan toan buoc doc file +
        gui bytes tu may local -> khong con nghen co chai CPU/mang local (da chan doan 2026-08-14)."""
        import io
        from pathlib import Path

        import torch
        from PIL import Image

        images = []
        for rel in rel_paths:
            full_path = Path(DENSE_VOLUME_MOUNT) / rel
            with open(full_path, "rb") as f:
                images.append(Image.open(io.BytesIO(f.read())).convert("RGB"))
        text_queries = [labels] * len(images)

        inputs = self.processor(text=text_queries, images=images, return_tensors="pt").to(self.device)
        if self.use_fp16:
            inputs["pixel_values"] = inputs["pixel_values"].half()
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([img.size[::-1] for img in images])
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, threshold=threshold, target_sizes=target_sizes, text_labels=text_queries
        )

        out = []
        for res, img in zip(results, images):
            w, h = img.size
            items = []
            text_labels_out = res.get("text_labels") or res.get("labels")
            for label, score, box in zip(text_labels_out, res["scores"], res["boxes"]):
                xmin, ymin, xmax, ymax = [float(x) for x in box.tolist()]
                items.append({
                    "label": str(label),
                    "score": float(score),
                    "xmin": max(0.0, xmin / w), "ymin": max(0.0, ymin / h),
                    "xmax": min(1.0, xmax / w), "ymax": min(1.0, ymax / h),
                })
            out.append(items)
        return out


@app.local_entrypoint()
def test():
    """modal run owlv2_app.py — test nhanh voi 1 anh + vai nhan co dinh."""
    import urllib.request

    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    img_bytes = urllib.request.urlopen(url).read()

    detector = OWLv2Detector()
    results = detector.detect_batch.remote([img_bytes], ["cat", "remote control", "couch", "dog"])
    print(results)
