"""Sinh câu trả lời ĐÚNG ĐỊNH DẠNG NỘP BÀI cho 3 dạng truy vấn BTC (xem
`Thong tin vong So tuyen AIC2026.pdf`, mục 1): Textual KIS, Q&A, TRAKE.

Ưu tiên CHẠY ĐÚNG ĐỊNH DẠNG trước — tái dùng nguyên `search()` (Tầng 1+2) và
`tiers.tier3_temporal.search()` (đã có, đã test) cho phần retrieval, chỉ thêm phần ĐỊNH DẠNG
output đúng như BTC yêu cầu + 1 khả năng hoàn toàn mới cho Q&A (sinh câu trả lời — VQA).
Thuật toán xếp hạng/chất lượng để tối ưu sau, không phải việc của file này.

Định dạng theo PDF:
  KIS:   <video_id>, <frame_id>
  Q&A:   <video_id>, <frame_id>, <answer>
  TRAKE: <video_id>, <frame_id_1>, ..., <frame_id_n>
Tối đa 100 câu trả lời/truy vấn (R@{1,5,20,50,100}) — mặc định top_k=100.
"""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/

import base64
import json
import os

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

import resources
from keyframe_images import read_keyframe_bytes
from query_planner import planned_search
from search import search
from steplog import StepLog
from tiers import tier3_temporal

SUBMISSION_TOP_K = 100


def answer_kis(query: str, top_k: int = SUBMISSION_TOP_K, log: StepLog | None = None, **filters) -> pd.DataFrame:
    """Textual KIS -> DataFrame [video_id, frame_id], đã xếp hạng (dòng đầu = rank 1, nộp
    theo đúng thứ tự này).

    SUA 2026-08-07: truoc day goi search() THUAN (chi CLIP Tang 2), bo qua HOAN TOAN
    query_planner (hard-filter entity, Region-CLIP thuoc tinh, secondary entity boost da xay
    dung rieng) - phat hien khi benchmark ra 0.0 diem: video DUNG nam trong top 100 nhung sai
    frame rat xa (vd kis_001: dung video nhung frame 384 thay vi [3454,3754]) vi CLIP thuan
    khong du manh phan biet chi tiet trong 1 video dai. Doi sang planned_search() de dung het
    may moc da xay."""
    r, _plan = planned_search(query, top_k=top_k, log=log, **filters)
    return r.rename(columns={"frame_idx": "frame_id"})[["video_id", "frame_id"]].reset_index(drop=True)


def answer_trake(anchors: list[str | dict], top_k: int = SUBMISSION_TOP_K, **filters) -> pd.DataFrame:
    """TRAKE -> DataFrame [video_id, frame_id_1, ..., frame_id_n], đã xếp hạng."""
    r = tier3_temporal.search(anchors, top_k=top_k, **filters)
    n = len(anchors)
    rename = {f"anchor{i}_frame_idx": f"frame_id_{i + 1}" for i in range(n)}
    cols = ["video_id"] + [f"anchor{i}_frame_idx" for i in range(n)]
    return r[cols].rename(columns=rename).reset_index(drop=True)


# ============================================================ Q&A — cần thêm VQA (chưa có trước đây)
# Dùng NVIDIA NIM thay vì Modal (2026-08-05): Modal chỉ đáng dùng cho batch lớn (P1, đã gác) —
# Q&A chỉ gọi VQA lẻ tẻ theo từng query, NIM (API hosted, không cold-start/container) đơn giản
# và ổn định hơn hẳn cho kiểu gọi này. Dùng lại model vision đã dùng ở v1 (agent/grounding.py).
NIM_VQA_MODEL = "meta/llama-3.2-11b-vision-instruct"

load_dotenv()
_nim_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.environ["NVIDIA_NIM_API_KEY"])


ASR_CONTEXT_WINDOW = 30  # so local_idx dem them 2 phia khi tim doan transcript gan frame -
# loi noi KHONG dong bo chinh xac voi hinh (xem hoi thoai 2026-08-10), nen phai dem ro rong
# hon la chi khop dung khoang local_idx_start..end cua doan ASR.


def _asr_context_for(video_id: str, local_idx: int, window: int = ASR_CONTEXT_WINDOW) -> str:
    """Tim cac doan transcript ASR (asr_text.parquet) GAN frame nay -> ghep lai thanh 1 doan
    text ngan lam NGU CANH BO SUNG cho VQA (2026-08-10, viec A trong 3 viec ASR — nhieu cau
    hoi Q&A hoi ve thu chi ton tai trong loi noi, khong suy ra duoc tu anh). Tra "" neu chua
    co asr_text.parquet hoac khong co doan nao gan — KHONG lam VQA that bai, chi bo qua ngu
    canh bo sung."""
    asr_index = resources.get().asr_index
    if asr_index is None:
        return ""
    sub = asr_index[
        (asr_index["video_id"] == video_id)
        & (asr_index["local_idx_start"] - window <= local_idx)
        & (asr_index["local_idx_end"] + window >= local_idx)
    ]
    if sub.empty:
        return ""
    return " ".join(sub.sort_values("local_idx_start")["text_raw"].tolist())


def _vqa_answer(video_id: str, local_idx: int, question: str, asr_context: str = "") -> str:
    """Hỏi thẳng câu hỏi lên 1 frame — KHÔNG qua Registry/gate (khác P1 ở tier4), vì Q&A cần
    trả lời tự do (màu sắc, số lượng...), không phải phân loại quan hệ đóng.

    NIM không đảm bảo structured output như vLLM tự host -> parse JSON có phòng thân (bỏ
    markdown fence nếu có), giống pattern đã dùng ở build_label_synonyms.py.

    asr_context: transcript lời nói GẦN frame này (xem _asr_context_for) — đưa vào prompt làm
    NGỮ CẢNH BỔ SUNG, không thay thế ảnh. Nhiều câu hỏi (số liệu đọc lên, tên được nhắc) chỉ
    trả lời đúng được nếu có cả 2 nguồn — chỉ dùng ảnh sẽ đoán mò."""
    img_bytes = read_keyframe_bytes(video_id, local_idx)
    img_b64 = base64.b64encode(img_bytes).decode()

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

    resp = _nim_client.chat.completions.create(
        model=NIM_VQA_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_content,
            },
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
    content = resp.choices[0].message.content.strip()
    content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(content)["answer"]
    except (json.JSONDecodeError, KeyError):
        return content  # phong than: tra nguyen text neu model khong theo dung JSON


def answer_qa(
    event_text: str,
    question: str,
    top_k: int = SUBMISSION_TOP_K,
    vqa_top_n: int = 5,
    log: StepLog | None = None,
    **filters,
) -> pd.DataFrame:
    """Q&A -> DataFrame [video_id, frame_id, answer], đã xếp hạng.

    vqa_top_n: chỉ thật sự GỌI VQA cho top-N ứng viên đầu (tốn tiền thật/lần gọi) — các rank
    thấp hơn dùng lại câu trả lời của rank 1 (nếu rank 1 sai video/frame thì answer đúng hay
    sai không quan trọng nữa, R-Score đã = 0 vì điều kiện video/frame không khớp trước).

    SUA 2026-08-07: doi tu search() thuan sang planned_search() - cung ly do nhu answer_kis()."""
    r, _plan = planned_search(event_text, top_k=top_k, log=log, **filters)
    r = r.rename(columns={"frame_idx": "frame_id"})

    answers = []
    best_answer = ""
    for i, row in r.iterrows():
        if i < vqa_top_n:
            asr_context = _asr_context_for(row["video_id"], int(row["local_idx"]))
            if log:
                with log.timed(f"VQA — gọi NIM cho ứng viên #{i + 1} ({row['video_id']})") as set_detail:
                    best_answer = _vqa_answer(row["video_id"], int(row["local_idx"]), question, asr_context)
                    detail = f'trả lời: "{best_answer}"'
                    if asr_context:
                        detail += f'  \n(có transcript ASR gần đó: "{asr_context[:150]}{"..." if len(asr_context) > 150 else ""}")'
                    set_detail(detail)
            else:
                best_answer = _vqa_answer(row["video_id"], int(row["local_idx"]), question, asr_context)
        answers.append(best_answer)
    r["answer"] = answers
    # giữ local_idx/pts_time/score thêm (không bắt buộc theo định dạng nộp bài, nhưng tiện cho
    # UI hiển thị ảnh + debug) — 3 cột chính vẫn là video_id/frame_id/answer.
    return r[["video_id", "frame_id", "local_idx", "pts_time", "score", "answer"]].reset_index(drop=True)


if __name__ == "__main__":
    import sys

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=== KIS demo ===")
    print(answer_kis("một diễn giả mặc áo đỏ phát biểu", top_k=5).to_string(index=False))

    print("\n=== TRAKE demo ===")
    print(
        answer_trake(
            ["phóng viên đứng trước ống kính giới thiệu", "phóng viên kết thúc bản tin"], top_k=3
        ).to_string(index=False)
    )
