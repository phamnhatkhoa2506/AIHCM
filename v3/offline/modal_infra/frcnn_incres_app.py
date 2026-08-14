"""Modal app: Faster R-CNN Inception-ResNet-v2 (TF-Hub, train tren OpenImages V4) - BASELINE
de so sanh voi OWLv2 (xem owlv2_app.py) truoc khi quyet dinh chay full corpus dense bang model
nao. Day la DUNG kien truc BTC dung de tao objects_index.parquet goc (Faster R-CNN Inception-
ResNet-v2), khac OWLv2 (open-vocab, tach nhan thanh embedding rieng). Model nay la CLOSED-SET
that su (~600 lop OpenImages V4 boxable, co san trong checkpoint, khong can truyen list nhan).

TF-Hub handle: https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1
Output detection_class_entities: ten nhan dang string (vd "Person", "Footwear") - CUNG khong
gian nhan voi label_vi.json (OpenImages) nen khop truc tiep, khong can bang anh xa rieng.

Dung TF image tu Docker registry (KHONG debian_slim + pip install tensorflow) - bai hoc tu
paddlepaddle: pip install tensorflow tren base debian_slim de lai cuDNN/cuda mismatch kho debug
tren GPU, base image chinh chu tensorflow/tensorflow:*-gpu da build san dung phien ban khop.
"""
import modal

MAX_CONTAINERS = 10

image = (
    modal.Image.from_registry("tensorflow/tensorflow:2.15.0-gpu")
    # --no-deps: tensorflow-hub keo theo tf-keras -> ep nang cap tensorflow len >=2.21, PHA VO
    # ban TF 2.15 co san trong base image (da khop san cuDNN/CUDA) - giong bai hoc paddlepaddle
    # (pip install de len base debian_slim gay cuDNN mismatch tren GPU).
    .run_commands("pip install --no-deps tensorflow-hub")
    .pip_install("pillow", "numpy<2")
)

tfhub_cache_vol = modal.Volume.from_name("aic2026-tfhub-cache", create_if_missing=True)

app = modal.App("aic2026-frcnn-incres")


@app.cls(
    image=image,
    gpu="A10G",
    max_containers=MAX_CONTAINERS,
    scaledown_window=5 * 60,
    timeout=120,
    volumes={"/root/.cache/tfhub": tfhub_cache_vol},
)
class FasterRCNNDetector:
    @modal.enter()
    def load(self):
        import os
        os.environ.setdefault("TFHUB_CACHE_DIR", "/root/.cache/tfhub")
        import tensorflow as tf
        import tensorflow_hub as hub

        gpus = tf.config.list_physical_devices("GPU")
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)

        handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
        self.detector = hub.load(handle).signatures["default"]

    @modal.method()
    def detect_batch(self, images_bytes: list[bytes], threshold: float = 0.15) -> list[list[dict]]:
        """Khong nhan tham so labels - model closed-set co san ~600 lop OpenImages V4, tra ve
        detection_class_entities lam label. Model nay CHI nhan 1 anh/lan goi (khong ho tro batch
        native trong signature "default") - vong for trong 1 call Modal, van loi cho autoscale
        vi .map() o driver van chia nhieu container song song."""
        import io
        import numpy as np
        import tensorflow as tf
        from PIL import Image

        out = []
        for b in images_bytes:
            img = Image.open(io.BytesIO(b)).convert("RGB")
            w, h = img.size
            arr = np.array(img).astype(np.float32) / 255.0  # module can float32 [0,1], khong phai uint8
            arr = arr[np.newaxis, ...]  # (1, H, W, 3)
            tensor = tf.convert_to_tensor(arr, dtype=tf.float32)
            result = self.detector(tensor)
            result = {k: v.numpy() for k, v in result.items()}

            items = []
            scores = result["detection_scores"]
            boxes = result["detection_boxes"]  # (N, 4) = ymin, xmin, ymax, xmax da normalize [0,1]
            entities = result["detection_class_entities"]
            for score, box, ent in zip(scores, boxes, entities):
                if score < threshold:
                    continue
                ymin, xmin, ymax, xmax = [float(x) for x in box]
                label = ent.decode("utf-8") if isinstance(ent, bytes) else str(ent)
                items.append({
                    "label": label,
                    "score": float(score),
                    "xmin": max(0.0, xmin), "ymin": max(0.0, ymin),
                    "xmax": min(1.0, xmax), "ymax": min(1.0, ymax),
                })
            out.append(items)
        return out


@app.local_entrypoint()
def test():
    """modal run frcnn_incres_app.py — test nhanh voi 1 anh."""
    import urllib.request
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    img_bytes = urllib.request.urlopen(url).read()
    detector = FasterRCNNDetector()
    results = detector.detect_batch.remote([img_bytes])
    print(results)
