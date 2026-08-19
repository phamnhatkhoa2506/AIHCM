"""Modal app: deploy Qwen2.5-VL-7B-Instruct qua vLLM trên A10G, expose OpenAI-compatible
endpoint (/v1/chat/completions) — dùng cho Phase P1 (trích xuất relation bằng VLM).

SỬA 2026-08-05 (lần 1): bản đầu dùng `@app.server(...)` dựa theo kết quả tra cứu web — KHÔNG
tồn tại trong `modal==1.3.5` thực tế đang cài (`AttributeError`). Đã verify lại trực tiếp bằng
`inspect.signature()` trên package đã cài — pattern đúng là `@app.function(...)` +
`@modal.web_server(port, ...)` trên 1 hàm thường, không cần class/`@modal.enter`/`@modal.exit`.

SỬA 2026-08-05 (lần 2): `vllm==0.21.0` deploy được thật (nên version này tồn tại, nghi ngờ
trước đó không đúng) nhưng engine chết lúc khởi động: Qwen2.5-VL mặc định `max_model_len`=128k
token, đòi 6.84 GiB KV cache trong khi A10G chỉ còn 1.8 GiB trống sau khi load weight. Bài
toán P1 không cần context dài -> giới hạn `--max-model-len` xuống (xem bên dưới).

Chạy thử (dev, hot-reload): modal serve app.py
Deploy that (URL cố định):  modal deploy app.py
"""
import json

import modal

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

MINUTES = 60
VLLM_PORT = 8000

# Số container tối đa chạy song song = số "GPU song song" đã bàn — chỉnh trực tiếp ở đây
# theo thời gian bạn chấp nhận chờ cho batch job (xem công thức: tổng thời gian ≈
# 156,965 frame × thời gian/frame / số container). Pilot: để thấp (2-4) trước khi tin số đo.
MAX_CONTAINERS = 1

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_LOG_STATS_INTERVAL": "1",
        }
    )
)

# Cache weights model qua các lần cold-start container — tránh tải lại HF mỗi lần lên container mới.
hf_cache_vol = modal.Volume.from_name("aic2026-hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("aic2026-vllm-cache", create_if_missing=True)

app = modal.App("aic2026-qwen25vl-p1")


@app.function(
    image=vllm_image,
    gpu="A10G",
    max_containers=MAX_CONTAINERS,
    scaledown_window=15 * MINUTES,  # giữ container "ấm" 15p sau request cuối, tránh cold-start lặp lại
    startup_timeout=10 * MINUTES,
    timeout=60 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.web_server(VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--served-model-name", MODEL_NAME, "p1-vlm",
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--uvicorn-log-level=info",
        # Bài toán P1 là phân loại đóng theo cặp cho sẵn (không phải scene-graph tự do)
        # -> output ngắn, không cần speculative decoding/reasoning parser như model text lớn.
        "--limit-mm-per-prompt", json.dumps({"image": 1, "video": 0, "audio": 0}),
        # FIX 2026-08-05: max_model_len mặc định của Qwen2.5-VL (128k) đòi 6.84 GiB KV cache,
        # A10G chỉ còn 1.8 GiB trống sau khi load weight -> ValueError lúc khởi động. P1 mỗi
        # request chỉ cần vài nghìn token (1 ảnh + danh sách cặp ngắn + JSON output ngắn),
        # không cần context dài -> giới hạn hẳn xuống, KV cache cần ~0.9 GiB, dư nhiều.
        "--max-model-len", "16384",
        "--gpu-memory-utilization", "0.90",
    ]
    print(*cmd)
    subprocess.Popen(cmd)


@app.local_entrypoint()
def test():
    """modal run app.py — chỉ nhắc lấy URL từ output `modal deploy`/`modal serve` hoặc dashboard
    (modal.com/apps) — bản modal đang cài không có API lấy URL kiểu Server.get_url.aio() mà bản
    thiết kế trước giả định (chưa verify được, không đoán thêm). Test schema P1 thật:
    dùng test_client.py với URL đó.
    """
    print("Lay URL server tu output cua 'modal deploy app.py' (hoac 'modal serve app.py' cho dev)")
    print("hoac tu dashboard: https://modal.com/apps -> aic2026-qwen25vl-p1 -> serve")
    print("Sau do chay: python test_client.py <url_do>")
