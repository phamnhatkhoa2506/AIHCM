"""Sinh câu trả lời ĐÚNG ĐỊNH DẠNG NỘP BÀI cho 3 dạng truy vấn BTC (xem
`Thong tin vong So tuyen AIC2026.pdf`, mục 1): Textual KIS, Q&A, TRAKE.

Tái dùng nguyên `tiers.dense_search.search_dense()` (Tầng 1+2) và `tiers.dense_temporal.search()`
(đã có, đã test) cho phần retrieval — ĐÚNG pipeline dense mà app.py (UI live) đang chạy, chỉ
thêm phần ĐỊNH DẠNG output đúng như BTC yêu cầu + 1 khả năng hoàn toàn mới cho Q&A (sinh câu trả
lời — VQA). Thuật toán xếp hạng/chất lượng để tối ưu sau, không phải việc của file này.

# 2026-08-20 (theo yeu cau nguoi dung: "dọn dẹp triệt để... không còn gọi CLIP") - answer_kis/
# answer_trake TRUOC DAY dung pipeline CLIP-32/keyframe BTC goc (planned_search() ->
# search()/tier2_vector.py, tier3_temporal.search()) - pipeline nay KHONG con duoc app.py goi
# (UI live da chuyen han sang dense_search/dense_temporal tu 2026-08-15/18), chi con ton tai de
# lam BASELINE SO SANH trong offline/benchmark/evaluate.py. Da xac nhan qua benchmark that
# (memory du an: SigLIP2 R@1=0.30 vs CLIP-32 0.16) dense pipeline thang ro, khong con can baseline
# nua - doi thang sang search_dense()/dense_temporal.search() (dung CHUNG code voi app.py, benchmark
# gio do DUNG THU app.py dang chay, khong phai 1 pipeline rieng dang chet dan). Xoa han online/
# search.py, tiers/tier2_vector.py, tiers/tier3_temporal.py, query_planner.planned_search()/
# _apply_region_clip_rerank() (xem git history neu can xem lai pipeline cu).

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

import openai
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from app_flags import DISABLE_LLM_ENTITY_HARD_FILTER, NIM_TIMEOUT_SECONDS, NIMTimeoutError
from query_planner import extract_entities
from steplog import StepLog
from tiers import dense_temporal
from tiers.dense_search import (
    ASR_CONTEXT_WINDOW_SECONDS,
    _fps_by_video,
    _load_dense_asr,
    apply_region_clip_rerank,
    search_dense,
)

SUBMISSION_TOP_K = 100
DEFAULT_BENCHMARK_DENSE_MODEL = "siglip"  # dung model thang benchmark ro nhat (xem memory du an)


def answer_kis(
    query: str, top_k: int = SUBMISSION_TOP_K, dense_model: str = DEFAULT_BENCHMARK_DENSE_MODEL,
    log: StepLog | None = None, **filters,
) -> pd.DataFrame:
    """Textual KIS -> DataFrame [video_id, frame_id], đã xếp hạng (dòng đầu = rank 1, nộp
    theo đúng thứ tự này). Gọi search_dense() TRỰC TIẾP (KHÔNG qua query_planner/LLM entity
    hard-filter - giống hành vi MẶC ĐỊNH của app.py khi checkbox "Dùng LLM phân rã câu" tắt,
    xem app.py::DISABLE_LLM_ENTITY_HARD_FILTER) - benchmark đo ĐÚNG pipeline app.py chạy thật."""
    r = search_dense(query, dense_model, top_k=top_k, log=log, **filters)
    return r[["video_id", "frame_id"]].reset_index(drop=True)


def answer_trake(
    anchors: list[str | dict], top_k: int = SUBMISSION_TOP_K,
    dense_model: str = DEFAULT_BENCHMARK_DENSE_MODEL, log: StepLog | None = None, **filters,
) -> pd.DataFrame:
    """TRAKE -> DataFrame [video_id, frame_id_1, ..., frame_id_n], đã xếp hạng."""
    r = dense_temporal.search(anchors, top_k=top_k, dense_model=dense_model, log=log, **filters)
    n = len(anchors)
    rename = {f"anchor{i}_frame_id": f"frame_id_{i + 1}" for i in range(n)}
    cols = ["video_id"] + [f"anchor{i}_frame_id" for i in range(n)]
    return r[cols].rename(columns=rename).reset_index(drop=True)


# ============================================================ Q&A — cần thêm VQA (chưa có trước đây)
# Dùng NVIDIA NIM thay vì Modal (2026-08-05): Modal chỉ đáng dùng cho batch lớn (P1, đã gác) —
# Q&A chỉ gọi VQA lẻ tẻ theo từng query, NIM (API hosted, không cold-start/container) đơn giản
# và ổn định hơn hẳn cho kiểu gọi này. Dùng lại model vision đã dùng ở v1 (agent/grounding.py).
NIM_VQA_MODEL = "meta/llama-3.2-11b-vision-instruct"

load_dotenv()
# timeout=NIM_TIMEOUT_SECONDS (2026-08-20, theo yeu cau nguoi dung - xem app_flags.py): tranh
# treo VO HAN khi NIM khong phan hoi (khong dat truoc day, xem BUG THAT trong query_distill.py).
_nim_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_NIM_API_KEY"],
    timeout=NIM_TIMEOUT_SECONDS,
)


def _dense_asr_context_for(video_id: str, frame_id: int, window_seconds: float = ASR_CONTEXT_WINDOW_SECONDS) -> str:
    """Tim cac doan transcript ASR (index/dense/asr_text.parquet, xem
    offline/build_dense_asr_index.py) GAN frame nay -> ghep lai thanh 1 doan text ngan lam NGU
    CANH BO SUNG cho VQA - BAN DENSE (2026-08-15, migrate theo yeu cau nguoi dung "sua Q&A theo
    bo du lieu moi") - dung frame_idx_start/end (da tinh qua fps) thay vi local_idx_start/end
    cua ban BTC cu. Tra "" neu chua co du lieu hoac khong co doan nao gan - KHONG lam VQA that
    bai, chi bo qua ngu canh bo sung."""
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


def _vqa_answer_dense(image_path: str, question: str, asr_context: str = "") -> str:
    """Hỏi thẳng câu hỏi lên 1 frame — KHÔNG qua Registry/gate (khác P1 ở tier4), vì Q&A cần
    trả lời tự do (màu sắc, số lượng...), không phải phân loại quan hệ đóng.

    NIM không đảm bảo structured output như vLLM tự host -> parse JSON có phòng thân (bỏ
    markdown fence nếu có), giống pattern đã dùng ở build_label_synonyms.py.

    asr_context: transcript lời nói GẦN frame này (xem _dense_asr_context_for) — đưa vào prompt
    làm NGỮ CẢNH BỔ SUNG, không thay thế ảnh. Nhiều câu hỏi (số liệu đọc lên, tên được nhắc) chỉ
    trả lời đúng được nếu có cả 2 nguồn — chỉ dùng ảnh sẽ đoán mò.

    image_path: đường dẫn file ảnh cục bộ (bộ dense nằm THẲNG trên đĩa - xem dense_meta.parquet
    "path"), KHÁC BTC (phải đọc qua Keyframes_*.zip, xem keyframe_images.py)."""
    with open(image_path, "rb") as f:
        img_bytes = f.read()
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

    try:
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
    except openai.APITimeoutError as e:
        # 2026-08-20 (theo yeu cau nguoi dung) - RAISE ro rang thay vi de treo vo han/loi tho.
        raise NIMTimeoutError(f"VQA (model={NIM_VQA_MODEL})") from e
    content = resp.choices[0].message.content.strip()
    content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(content)["answer"]
    except (json.JSONDecodeError, KeyError):
        return content  # phong than: tra nguyen text neu model khong theo dung JSON


# 2026-08-17 (theo yeu cau nguoi dung "thêm 1 số mô hình to hơn xem thử") - registry model VLM
# CO THE CHON cho vlm_read_text (giong het pattern OCR_MATCH_ALGORITHMS/SCORE_ALGORITHMS -
# dense_search.py) - TEST TRUC TIEP tren case that (bien 3 dong "GIÁO HỘI PHẬT GIÁO HÒA HẢO/
# BAN TRỊ SỰ TRUNG ƯƠNG/BAN QUẢN TỰ AN HÒA TỰ" ma PaddleOCR bo sot/doc sai, xem hoi thoai
# 2026-08-17) qua het cac model vision co tren NIM (client.models.list(), loc theo tu khoa
# vision/vl):
#   - nvidia/llama-3.1-nemotron-nano-vl-8b-v1: DOC DUNG dong "BAN QUAN TU AN HOA TU" tren ANH
#     GOC (full-frame, KHONG can crop rieng) - CHINH XAC NHAT trong tat ca model test duoc,
#     dat lam MAC DINH MOI thay the model VQA cu cho rieng tinh nang doc chu nay.
#   - meta/llama-3.2-11b-vision-instruct: model VQA cu (NIM_VQA_MODEL) - GIU LAM LUA CHON (dung
#     chung ha tang voi VQA that su, khong phai luon kem hon - tren anh GOC tung cho ra
#     "BANGIÁN TY AN HOÀI" SAI, nhung tren anh CROP GAN lai DOC DUNG).
#   - nvidia/nemotron-nano-12b-v2-vl: model to hon (12B) - doc gan dung ("ban quan tu an hoa
#     to" tren anh goc, sai 1 ky tu cuoi "to" thay "tu"; DUNG HOAN TOAN tren anh crop).
# CA 3 model DEU KHONG doc duoc 2 dong CHU NHO hon phia tren bien ("GIÁO HỘI..."/"BAN TRỊ SỰ...")
# du crop cận hay khong - gioi han THAT cua ca 3 (co the do do phan giai anh bi nen/giam khi
# gui qua API hosted), KHONG phai do chon sai model - da disclose ro trong UI help text.
# Model KHONG kha dung qua NIM tai khoan hien tai (404) hoac khong on dinh (504 lien tuc), DA
# LOAI KHOI danh sach: meta/llama-3.2-90b-vision-instruct, microsoft/phi-3-vision-128k-instruct,
# nvidia/neva-22b, nvidia/vila.
VLM_OCR_MODELS: dict[str, str] = {
    "nvidia/llama-3.1-nemotron-nano-vl-8b-v1": "Nemotron Nano VL 8B (mặc định — đọc đúng nhất khi test thật)",
    "meta/llama-3.2-11b-vision-instruct": "Llama 3.2 11B Vision (model VQA cũ)",
    "nvidia/nemotron-nano-12b-v2-vl": "Nemotron Nano 12B v2 VL (to hơn, gần đúng)",
}
DEFAULT_VLM_OCR_MODEL = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"


def vlm_read_text(image_path: str, model: str = DEFAULT_VLM_OCR_MODEL) -> str:
    """Doc TOAN BO chu nhin thay duoc trong 1 anh bang VLM (2026-08-17, theo yeu cau nguoi
    dung: "dùng 1 VLM nhỏ để đọc chữ thôi, bỏ qua box cho các trường hợp như thế này" - case
    that: bien 3 dong "GIÁO HỘI PHẬT GIÁO HÒA HẢO/BAN TRỊ SỰ TRUNG ƯƠNG/BAN QUẢN TỰ AN HÒA TỰ"
    bi PaddleOCR bo sot 2 dong dau + doc sai dong con lai o CA 4 frame trong shot, xem hoi thoai).

    THIET KE: xac minh LAZY, on-demand (nguoi dung TU bam nut khi nghi ngo PaddleOCR bo sot/doc
    sai - xem app.py::_render_vlm_ocr_verify) - KHONG tu dong chay theo moi query/toan corpus
    ("dữ liệu chữ này khá sparse, chạy hết thì rất phí" - nguyen van nguoi dung). Tai dung
    _nim_client/NIM_VQA_MODEL da co san (dung chung ha tang voi _vqa_answer_dense, khong deploy
    them Modal/vLLM rieng). CHI tra ve chuoi text tho, KHONG co bbox (mat thong tin vi tri so
    voi PaddleOCR) - vi vay CHI dung de xac minh HARD-FILTER (co/khong co chu X trong anh),
    KHONG dung thay the PaddleOCR cho soft-boost vi tri khung ve (_ocr_box_position_score_for_
    frame van can bbox that, xem dense_search.py)."""
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
        # 2026-08-20 (theo yeu cau nguoi dung) - RAISE ro rang - app.py::_render_vlm_ocr_verify
        # da co san except Exception BAO NGOAI, hien "LỖI: ..." ngay trong popover, khong can
        # sua them o do.
        raise NIMTimeoutError(f"VLM đọc chữ (model={model})") from e
    content = resp.choices[0].message.content.strip()
    content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(content)["text"]
    except (json.JSONDecodeError, KeyError):
        return content  # phong than: tra nguyen text neu model khong theo dung JSON


def answer_qa(
    event_text: str,
    question: str,
    top_k: int = SUBMISSION_TOP_K,
    vqa_top_n: int = 5,
    dense_model: str = "rrf",
    use_region_clip_rerank: bool = True,
    use_lvlm: bool = False,
    use_llm_entity: bool = False,
    log: StepLog | None = None,
    **filters,
) -> pd.DataFrame:
    """Q&A -> DataFrame [video_id, frame_id, path, score, answer, is_backfill], đã xếp hạng.

    use_lvlm (2026-08-20, theo yêu cầu người dùng: "Dùng LVLM để trả lời giờ là 1 option và để
    mặc định là không dùng") - MẶC ĐỊNH False: KHÔNG gọi VQA thật nữa (answer để RỖNG, người
    dùng tự gõ câu trả lời trong Playback trước khi nộp — xem app.py::_render_playback). True
    -> giữ NGUYÊN hành vi cũ (gọi NIM VQA thật cho top vqa_top_n ứng viên).

    use_llm_entity (2026-08-20, theo yêu cầu người dùng: "trước đó mình muốn thêm option cho
    [LLM phân rã câu]... mặc định là tắt", dùng CHUNG 1 checkbox với app.py cho cả KIS/Q&A) -
    MẶC ĐỊNH False: KHÔNG gọi extract_entities() (LLM thật, ~3s + phí API) - dùng STUB rỗng,
    Region-CLIP rerank/ASR audio_mentions tự động không có gì để chạy (PHỤ THUỘC vào cờ này).

    MIGRATE sang BỘ DENSE (2026-08-15, theo yêu cầu người dùng "sửa Q&A theo bộ dữ liệu mới")
    — dùng search_dense() + extract_entities() (ĐÚNG pattern đường "1 câu" chính trong app.py)
    thay vì planned_search() (BTC cũ). Ảnh đọc THẲNG từ đĩa local (row["path"]), ASR context từ
    index/dense/asr_text.parquet (frame_idx đã remap qua fps — xem dense_search.py), thay vì
    asr_index BTC cũ (local_idx).

    use_region_clip_rerank (2026-08-15, thêm sau theo yêu cầu người dùng "tích hợp Rerank như
    KIS"): QUAN TRỌNG — rerank PHẢI chạy TRƯỚC vòng lặp VQA bên dưới (không phải sau), vì
    vqa_top_n chỉ hỏi VQA thật cho top-N sau khi đã xếp hạng lại — rerank sau sẽ hỏi nhầm ảnh.

    vqa_top_n: chỉ thật sự GỌI VQA cho top-N ứng viên đầu (tốn tiền thật/lần gọi) — các rank
    thấp hơn dùng lại câu trả lời của rank 1 (nếu rank 1 sai video/frame thì answer đúng hay
    sai không quan trọng nữa, R-Score đã = 0 vì điều kiện video/frame không khớp trước)."""
    # loai cac tham so CHI danh cho pipeline BTC cu (khong ap dung cho search_dense)
    filters.pop("use_suppression", None)
    filters.pop("include_open_vocab", None)
    filters.pop("ocr_region", None)

    if use_llm_entity:
        plan = extract_entities(event_text, log=log)
    else:
        plan = {
            "entities": [], "secondary_entities": [], "attributes": [],
            "audio_mentions": [], "clip_text": event_text, "unresolved": [],
            "resolved_must_have_labels": [], "resolved_min_count": {},
        }
        if log:
            log.add("LLM phân rã câu (NIM)", "TẮT (checkbox người dùng) — bỏ qua hoàn toàn", 0.0)
    user_must_have = filters.pop("must_have_labels", None) or []
    user_min_count = filters.pop("min_count", None) or {}
    # 2026-08-17 (GAC LAI de TEST - xem share/app_flags.py cho ly do day du: LLM plan_query()
    # bia them entity khong co that, resolve nham thanh hard-filter SAI, bop hep sai corpus).
    # TAM THOI chi giu must_have_labels/min_count NGUOI DUNG tu truyen vao (khung Object
    # canvas), BO QUA phan LLM tu trich - dung CHUNG 1 co voi online/app.py qua app_flags.py.
    if DISABLE_LLM_ENTITY_HARD_FILTER:
        merged_must_have = user_must_have or None
        merged_min_count = user_min_count or None
    else:
        merged_must_have = list({*user_must_have, *plan["resolved_must_have_labels"]}) or None
        merged_min_count = {**user_min_count, **plan["resolved_min_count"]} or None

    # Region-CLIP rerank (giong het pattern KIS trong app.py) - can pool RONG HON top_k de
    # rerank co gi ma chon, roi cat lai dung top_k SAU KHI rerank, TRUOC KHI vao vong lap VQA.
    attributes = (plan.get("attributes") or []) if use_region_clip_rerank else []
    search_top_k = top_k * 4 if attributes else top_k

    r = search_dense(
        event_text, dense_model, top_k=search_top_k,
        must_have_labels=merged_must_have, min_count=merged_min_count,
        audio_mentions=plan.get("audio_mentions") or None,
        log=log, **filters,
    )
    if attributes and not r.empty:
        r = apply_region_clip_rerank(r, attributes, top_k, log=log)

    if not use_lvlm:
        # 2026-08-20 (theo yeu cau nguoi dung: "Dùng LVLM để trả lời giờ là 1 option và để mặc
        # định là không dùng") - KHONG goi VQA cho BAT KY dong nao, "answer" de RONG toan bo -
        # nguoi dung tu go trong Playback (xem app.py) truoc khi nop.
        r["answer"] = ""
        cols = ["video_id", "frame_id", "path", "score", "answer"]
        if "is_backfill" in r.columns:
            cols.append("is_backfill")
        return r[cols].reset_index(drop=True)

    answers = []
    best_answer = ""
    for i, row in r.iterrows():
        if i < vqa_top_n:
            asr_context = _dense_asr_context_for(row["video_id"], int(row["frame_id"]))
            if log:
                with log.timed(f"VQA — gọi NIM cho ứng viên #{i + 1} ({row['video_id']})") as set_detail:
                    best_answer = _vqa_answer_dense(row["path"], question, asr_context)
                    detail = f'trả lời: "{best_answer}"'
                    if asr_context:
                        detail += f'  \n(có transcript ASR gần đó: "{asr_context[:150]}{"..." if len(asr_context) > 150 else ""}")'
                    set_detail(detail)
            else:
                best_answer = _vqa_answer_dense(row["path"], question, asr_context)
        answers.append(best_answer)
    r["answer"] = answers
    cols = ["video_id", "frame_id", "path", "score", "answer"]
    if "is_backfill" in r.columns:
        cols.append("is_backfill")
    return r[cols].reset_index(drop=True)


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
