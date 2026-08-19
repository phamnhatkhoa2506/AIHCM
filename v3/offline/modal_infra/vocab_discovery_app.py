"""Modal app RIÊNG (tách khỏi app.py/P1) cho giai đoạn khám phá vocab mở — VLM chỉ liệt kê
ngắn gọn vật thể/trang phục KHÔNG thuộc 514 nhãn chuẩn đã có (xem thiết kế 2026-08-06: VLM đề
xuất vocab -> gộp tần suất -> Grounding DINO định vị bằng vocab đó).

Vì sao app RIÊNG thay vì dùng lại serve() của app.py (P1):
  - Output ở đây RẤT ngắn (~1 mảng string, tối đa vài cụm từ) so với P1 (JSON quan hệ từng cặp,
    có thể tới hàng nghìn token) -> max_model_len/KV-cache cần thấp hơn NHIỀU, tối ưu riêng được.
  - max_tokens output cố định nhỏ (không phụ thuộc số object trong frame như P1) -> throughput/
    container cao hơn, có thể tăng MAX_CONTAINERS nhiều hơn P1 mà vẫn rẻ.
  - Tách app để đổi config app này không ảnh hưởng P1 đang chạy ổn định.

Cùng model (Qwen2.5-VL-7B-Instruct) nên DÙNG CHUNG volume cache HF weights (aic2026-hf-cache)
với app.py để khỏi tải lại 7B weight — chỉ vllm cache/app definition là tách riêng.

Chạy thử (dev, hot-reload): modal serve vocab_discovery_app.py
Deploy thật (URL cố định):  modal deploy vocab_discovery_app.py
"""
import json

import modal

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

MINUTES = 60
VLLM_PORT = 8000

# Output ngan (~vai chuc token) -> moi request re hon P1 nhieu -> tang song song duoc nhieu
# hon. Pilot truoc voi so thap, do throughput thuc te roi tang dan (dung tin so ly thuyet).
MAX_CONTAINERS = 10  # gioi han tai khoan Modal hien tai la 10 container dong thoi (2026-08-06)

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

hf_cache_vol = modal.Volume.from_name("aic2026-hf-cache", create_if_missing=True)  # dung chung P1
vllm_cache_vol = modal.Volume.from_name("aic2026-vllm-cache-vocab", create_if_missing=True)  # rieng

app = modal.App("aic2026-qwen25vl-vocab")


@app.function(
    image=vllm_image,
    gpu="A10G",
    max_containers=MAX_CONTAINERS,
    scaledown_window=15 * MINUTES,
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
        "--served-model-name", MODEL_NAME, "vocab-vlm",
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--uvicorn-log-level=info",
        "--limit-mm-per-prompt", json.dumps({"image": 1, "video": 0, "audio": 0}),
        # Output cuc ngan (mang string, toi da vai cum tu) -> khong can context dai nhu P1
        # (16384). Anh van chiem phan lon context (image token), nhung phan output/text du
        # thap hon nhieu -> ha xuong 8192 la du du, giai phong GPU mem cho throughput/batch.
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.90",
        # SYSTEM_PROMPT nhet nguyen 514 nhan OpenImages (~1,282 token, do bang tay) - GIONG HET
        # nhau moi request (chi anh la doi). Bat prefix caching de vLLM tinh KV-cache doan do
        # 1 LAN roi tai su dung, khong phai tinh lai tu dau x177,321 frame - neu khong bat,
        # phan nay tro thanh chi phi lap lai rat lon (2026-08-06, phat hien khi ban hoi).
        "--enable-prefix-caching",
    ]
    print(*cmd)
    subprocess.Popen(cmd)


@app.local_entrypoint()
def test():
    print("Lay URL server tu output cua 'modal deploy vocab_discovery_app.py' hoac dashboard:")
    print("https://modal.com/apps -> aic2026-qwen25vl-vocab -> serve")
