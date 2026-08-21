"""Q&A — sinh câu trả lời (VQA) cho 1 frame cụ thể, bản trích riêng cho app/ từ
v3/online/submission_pipeline.py (`_vqa_answer_dense`, `_dense_asr_context_for`, `answer_qa`).
MẶC ĐỊNH KHÔNG gọi VQA thật (tốn phí/lần gọi API) — chỉ gọi khi người dùng CHỦ ĐỘNG bật checkbox
"Dùng LVLM tự động trả lời" (xem index.html), và CHỈ cho top-N ứng viên đầu (vqa_top_n) — các
rank thấp hơn để trống, người dùng tự điền tay trước khi nộp."""
from __future__ import annotations

import base64
import json

import openai
import pandas as pd

from app_flags import NIMTimeoutError
from tiers.dense_search import ASR_CONTEXT_WINDOW_SECONDS, _fps_by_video, _load_dense_asr
# 2026-08-21 (theo yêu cầu người dùng: "Mô hình trả lời Q&A dùng chung với mô hình đọc OCR
# luôn") - KHÔNG còn model VQA riêng (NIM_VQA_MODEL cũ) - dùng CHUNG registry + client + model
# mặc định với vlm_verify.py (dropdown "Model VLM đọc chữ" trong sidebar giờ điều khiển CẢ 2
# tính năng: xác minh OCR VÀ trả lời Q&A) - cùng là hỏi-đáp trên 1 ảnh, hợp lý dùng chung model.
from vlm_verify import DEFAULT_VLM_OCR_MODEL, VLM_OCR_MODELS, _nim_client


def _dense_asr_context_for(video_id: str, frame_id: int, window_seconds: float = ASR_CONTEXT_WINDOW_SECONDS) -> str:
    """Ghép các đoạn transcript ASR gần frame này làm ngữ cảnh bổ sung cho VQA. Trả "" nếu chưa
    có dữ liệu hoặc không có đoạn nào gần - KHÔNG làm VQA thất bại, chỉ bỏ qua ngữ cảnh."""
    asr = _load_dense_asr()
    if asr is None:
        return ""
    fps = _fps_by_video().get(video_id, 25.0)
    window = int(round(window_seconds * fps))
    sub = asr[
        (asr["video_id"] == video_id)
        & (asr["frame_idx_start"] - window <= frame_id)
        & (asr["frame_idx_end"] + window >= frame_id)
    ]
    if sub.empty:
        return ""
    return " ".join(sub.sort_values("frame_idx_start")["text_raw"].tolist())


def vqa_answer(image_path: str, question: str, asr_context: str = "", model: str | None = None) -> str:
    """Hỏi thẳng câu hỏi lên 1 frame - trả lời tự do (không qua Registry/gate). model: key trong
    vlm_verify.VLM_OCR_MODELS (dùng CHUNG dropdown "Model VLM đọc chữ" - xem ghi chú đầu file),
    None/không hợp lệ -> fallback về DEFAULT_VLM_OCR_MODEL."""
    if model not in VLM_OCR_MODELS:
        model = DEFAULT_VLM_OCR_MODEL
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    system_content = (
        "Answer the question about the image with a short, direct answer "
        "(a word, number, or short phrase) — no explanation. Answer in the same "
        "language as the question. Reply with ONLY a JSON object: "
        '{"answer": "..."} — no markdown fences.'
    )
    if asr_context:
        system_content += (
            " You are also given a spoken transcript near this moment in the video — use it "
            "ONLY if it helps answer (e.g. a number or name that is spoken but not visible in "
            "the image); ignore it if irrelevant to the question. Transcript: "
            f'"{asr_context}"'
        )

    try:
        resp = _nim_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_content},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    ],
                },
            ],
            max_tokens=100,
            temperature=0.1,
        )
    except openai.APITimeoutError as e:
        raise NIMTimeoutError(f"VQA (model={model})") from e
    content = resp.choices[0].message.content.strip()
    content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(content)["answer"]
    except (json.JSONDecodeError, KeyError):
        return content  # phòng thân: trả nguyên text nếu model không theo đúng JSON


def answer_for_results(
    result: pd.DataFrame, question: str, vqa_top_n: int, model: str | None = None, log=None,
) -> list[str]:
    """Sinh answer cho từng dòng trong `result` (ĐÃ xếp hạng) - chỉ gọi VQA thật cho top-N ứng
    viên đầu (tốn phí/lần), các rank thấp hơn DÙNG LẠI câu trả lời của rank 1 - giống hệt
    `answer_qa` bên v3."""
    answers: list[str] = []
    best_answer = ""
    for i, (_, row) in enumerate(result.iterrows()):
        if i < vqa_top_n:
            asr_context = _dense_asr_context_for(row["video_id"], int(row["frame_id"]))
            if log:
                with log.timed(f"VQA — gọi NIM cho ứng viên #{i + 1} ({row['video_id']})") as set_detail:
                    best_answer = vqa_answer(row["path"], question, asr_context, model=model)
                    set_detail(f'trả lời: "{best_answer}"')
            else:
                best_answer = vqa_answer(row["path"], question, asr_context, model=model)
        answers.append(best_answer)
    return answers
