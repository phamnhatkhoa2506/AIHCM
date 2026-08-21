"""Xác minh OCR bằng VLM (đọc toàn bộ chữ trong 1 ảnh) — bản trích riêng cho app/ từ
v3/online/submission_pipeline.py::vlm_read_text (chỉ lấy đúng phần này, không copy nguyên
submission_pipeline.py vì answer_kis/answer_trake/answer_qa/VQA chưa cần cho app/ ở giai đoạn
này). Xác minh LAZY, on-demand — người dùng tự bấm nút "🔍 VLM Verify" dưới mỗi kết quả khi nghi
ngờ OCR (PaddleOCR) bỏ sót/đọc sai, KHÔNG tự động chạy theo mỗi query/toàn corpus."""
from __future__ import annotations

import base64
import json
import os

import openai
from dotenv import load_dotenv
from openai import OpenAI

from app_flags import NIM_TIMEOUT_SECONDS, NIMTimeoutError

load_dotenv()
_nim_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_NIM_API_KEY", ""),
    timeout=NIM_TIMEOUT_SECONDS,
)

# Giống hệt registry bên v3 (đã test trên case thật, xem submission_pipeline.py để biết lý do
# chọn từng model) - "nvidia/llama-3.1-nemotron-nano-vl-8b-v1" mặc định vì đọc đúng nhất khi test.
VLM_OCR_MODELS: dict[str, str] = {
    "nvidia/llama-3.1-nemotron-nano-vl-8b-v1": "Nemotron Nano VL 8B (mặc định — đọc đúng nhất khi test thật)",
    "meta/llama-3.2-11b-vision-instruct": "Llama 3.2 11B Vision (model VQA cũ)",
    "nvidia/nemotron-nano-12b-v2-vl": "Nemotron Nano 12B v2 VL (to hơn, gần đúng)",
}
DEFAULT_VLM_OCR_MODEL = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"


def vlm_read_text(image_path: str, model: str = DEFAULT_VLM_OCR_MODEL) -> str:
    """Đọc TOÀN BỘ chữ nhìn thấy được trong 1 ảnh bằng VLM. Chỉ trả về chuỗi text thô, KHÔNG có
    bbox (mất thông tin vị trí so với PaddleOCR) - vì vậy chỉ dùng để xác minh HARD-FILTER
    (có/không có chữ X trong ảnh), không thay thế PaddleOCR cho soft-boost vị trí."""
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    system_content = (
        "Read ALL text visible anywhere in the image (signs, banners, subtitles, logos...), "
        "in its ORIGINAL language and script — do not translate. Transcribe as accurately as "
        "possible, preserving line breaks as ' / '. If there is no readable text, answer with "
        "an empty string. Reply with ONLY a JSON object: {\"text\": \"...\"} — no markdown fences, "
        "no explanation."
    )
    try:
        resp = _nim_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_content},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Đọc toàn bộ chữ trong ảnh."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    ],
                },
            ],
            max_tokens=300,
            temperature=0.1,
        )
    except openai.APITimeoutError as e:
        raise NIMTimeoutError(f"VLM đọc chữ (model={model})") from e
    content = resp.choices[0].message.content.strip()
    content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(content)["text"]
    except (json.JSONDecodeError, KeyError):
        return content  # phòng thân: trả nguyên text nếu model không theo đúng JSON
