"""Modal app: encode ảnh (region crop) bằng CLIP image encoder (`clip-ViT-B-32`) trên A10G —
dùng để precompute region-CLIP cho `objects_index.parquet` thay vì chạy CPU local (đã đo:
CPU máy này chỉ ~4.5 box/s, cả 1.1 triệu object mất ~69 giờ).

Nhẹ hơn hẳn `app.py` (P1, vLLM serve 7B) — chỉ là 1 CLIP image encoder nhỏ, không cần vLLM,
không cần HTTP server, gọi trực tiếp qua Modal RPC (`modal.Cls.from_name`, xem
build_region_embeddings.py). Crop ảnh vẫn làm ở LOCAL (chỉ máy local có file zip keyframe) —
chỉ gửi bytes ảnh đã crop lên Modal để encode, không gửi cả frame gốc.

Lưu ý (2026-08-20, theo câu hỏi người dùng "Region-CLIP embedding không phải dùng SigLIP-2 à
bạn"): app này dùng CLIP-ViT-B-32 (KHÔNG phải SigLIP2) — ĐÃ BỊ THAY THẾ cho tính năng Region-
CLIP rerank LIVE trong app.py (giờ chạy hoàn toàn trên offline/modal_infra/region_rerank_app.py,
dùng SigLIP2, xem docstring file đó + build_dense_region_embeddings_shard.py). App này CHỈ còn
được offline/audit_object_labels.py gọi (audit chất lượng nhãn, không phải search live).

Deploy: modal deploy region_clip_app.py
"""
import modal

app = modal.App("aic2026-region-clip")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "sentence-transformers", "pillow"
)

MODEL_NAME = "clip-ViT-B-32"
hf_cache_vol = modal.Volume.from_name("aic2026-hf-cache", create_if_missing=True)


@app.cls(
    image=image,
    gpu="A10G",
    scaledown_window=5 * 60,
    max_containers=8,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
)
class Encoder:
    @modal.enter()
    def load(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(MODEL_NAME, cache_folder="/root/.cache/huggingface")

    @modal.method()
    def encode_batch(self, crops_bytes: list[bytes]) -> list[list[float]]:
        import io

        from PIL import Image

        imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in crops_bytes]
        vecs = self.model.encode(
            imgs, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return vecs.tolist()


@app.local_entrypoint()
def test():
    """modal run region_clip_app.py — test nhanh với 1 ảnh trắng giả."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (100, 100), color="red").save(buf, format="JPEG")
    encoder = Encoder()
    result = encoder.encode_batch.remote([buf.getvalue()])
    print(f"OK — nhận về {len(result)} vector, dim={len(result[0])}")
