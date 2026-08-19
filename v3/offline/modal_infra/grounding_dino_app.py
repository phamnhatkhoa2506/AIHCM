"""Modal app cho Giai đoạn 3 (định vị bằng Grounding DINO) của thiết kế "VLM khám phá vocab mở
-> Grounding DINO định vị" (chốt hướng 2026-08-06). App nhẹ (không phải vLLM server) — dùng
trực tiếp `transformers.GroundingDinoForObjectDetection`, giống pattern
`region_clip_app.py` (@app.cls + @modal.enter + @modal.method, không cần HTTP server).

Model: IDEA-Research/grounding-dino-tiny — đã test local CPU (test_grounding_dino.py, xác
nhận đúng 3/3 case biết trước bị detector cũ gán nhãn sai: Dog->lion dance costume,
Camel->dragon dance costume, Watercraft->news studio backdrop).

TỐI ƯU TỐC ĐỘ: gom các ảnh CÙNG prompt (cùng tập phrase) thành 1 lượt batch qua model thay vì
gọi từng ảnh riêng — GroundingDinoProcessor hỗ trợ batch nhiều ảnh cùng lúc (kiểm chứng qua
inspect.signature: text nhận list[str], không chỉ 1 string). Vì nhiều frame CÙNG VIDEO thường
lặp lại đúng cụm phrase (vd "news studio graphic" xuất hiện ở rất nhiều frame bản tin), gom
theo prompt giống nhau tiết kiệm được nhiều lượt forward — driver (run_grounding_dino.py) lo
việc gom nhóm trước khi gửi, method ở đây chỉ nhận 1 prompt DÙNG CHUNG cho cả batch ảnh.

Chạy thử (dev, hot-reload): modal serve grounding_dino_app.py
Deploy thật:                modal deploy grounding_dino_app.py
"""
import modal

MODEL_NAME = "IDEA-Research/grounding-dino-tiny"

MAX_CONTAINERS = 8

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "pillow", "accelerate")
)

hf_cache_vol = modal.Volume.from_name("aic2026-hf-cache", create_if_missing=True)  # dung chung

app = modal.App("aic2026-grounding-dino")


@app.cls(
    image=image,
    gpu="A10G",
    max_containers=MAX_CONTAINERS,
    scaledown_window=5 * 60,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
)
class Detector:
    @modal.enter()
    def load(self):
        import torch
        from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

        self.processor = GroundingDinoProcessor.from_pretrained(MODEL_NAME, cache_dir="/root/.cache/huggingface")
        self.model = GroundingDinoForObjectDetection.from_pretrained(MODEL_NAME, cache_dir="/root/.cache/huggingface")
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    @modal.method()
    def detect_batch(
        self, images_bytes: list[bytes], prompt: str, threshold: float = 0.25, text_threshold: float = 0.20
    ) -> list[list[dict]]:
        """1 PROMPT dung chung cho ca batch anh (driver da gom nhom cung prompt truoc khi goi).
        Tra ve list (dung thu tu images_bytes), moi phan tu la list[{"label","score","box"}]
        (box = [xmin,xmax,ymin,ymax] px, chua normalize - driver tu chia w/h)."""
        import io

        import torch
        from PIL import Image

        images = [Image.open(io.BytesIO(b)).convert("RGB") for b in images_bytes]
        inputs = self.processor(images=images, text=[prompt] * len(images), return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[img.size[::-1] for img in images],
        )

        out = []
        for res in results:
            items = []
            text_labels = res.get("text_labels") or res.get("labels")  # fallback tuy version transformers
            for label, score, box in zip(text_labels, res["scores"], res["boxes"]):
                items.append({
                    "label": str(label),
                    "score": float(score),
                    "box": [round(float(x), 2) for x in box.tolist()],  # [xmin,ymin,xmax,ymax] px
                })
            out.append(items)
        return out


@app.local_entrypoint()
def test():
    """modal run grounding_dino_app.py — test nhanh voi 1 anh + prompt co dinh."""
    import urllib.request

    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    img_bytes = urllib.request.urlopen(url).read()

    detector = Detector()
    results = detector.detect_batch.remote([img_bytes], "a cat.")
    print(results)
