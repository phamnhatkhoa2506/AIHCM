"""FastAPI backend cho UI mới (thay Streamlit online/app.py) — bọc lại NGUYÊN pipeline Python
đã có trong v3/share + v3/online (không viết lại thuật toán search/submission). Chạy:

    uv run --python ../.venv/Scripts/python.exe uvicorn backend.main:app --reload --port 8800

(từ thư mục app/). Giai đoạn 1 (đang làm): Search (KIS/QA/TRAKE) + results grid + Submit.
Playback/Video và VLM/OCR verify sẽ thêm ở các giai đoạn sau (xem plan di chuyển UI)."""
from __future__ import annotations

import backend.bootstrap  # noqa: F401  # PHẢI import đầu tiên — chèn sys.path tới backend/core/

import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger("aic_app")

from app_flags import ModalTimeoutError, ModalUnavailableError, NIMTimeoutError
from query_planner import extract_entities
from steplog import StepLog
from tiers import dense_temporal
from tiers.dense_search import _fps_by_video, _frame_idx_by_video, apply_region_clip_rerank, search_dense

from label_translate import list_labels_vi
from vlm_verify import DEFAULT_VLM_OCR_MODEL, vlm_read_text
from video_audio import local_video_path, read_video_bytes
from video_clip import extract_single_frame, get_shot_clip_bytes
from vqa import answer_for_results
from spatial_canvas import boxes_to_spatial

from backend.schemas import (
    SearchRequest, SearchResponse, SearchResultRow, VideoMetaResponse,
    VlmVerifyRequest, VlmVerifyResponse,
)
from backend.submissions import router as submissions_router

APP_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = APP_DIR / "static"

app = FastAPI(title="AIC 2026")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

app.include_router(submissions_router, prefix="/api/submissions", tags=["submissions"])


# 2026-08-21 (theo yêu cầu người dùng: "xem ta đã handle exception cho toàn bộ app tốt chưa") -
# BUG THẬT đã kiểm chứng bằng curl: bất kỳ exception nào KHÔNG thuộc 3 loại Modal/NIM cụ thể
# (vd dense_model sai tên, KeyError/IndexError từ dữ liệu bất thường, file index thiếu...) đều
# rơi vào handler MẶC ĐỊNH của Starlette -> trả "Internal Server Error" dạng `text/plain`,
# KHÔNG PHẢI JSON. Frontend gọi `res.json()` trên body này sẽ tự ném SyntaxError (JSON không hợp
# lệ) -> catch bắt được NHƯNG hiện nhầm "Lỗi kết nối" (connection error) dù đây là SERVER CRASH,
# và người dùng/dev không thấy được lý do thật (traceback chỉ nằm trong console uvicorn, không
# về tới client). Bọc CHUNG 1 handler cho MỌI route (kể cả những route quên bọc try/except riêng
# như /api/video_meta, /api/labels) - log đủ traceback ở server, trả JSON {"detail": ...} cho
# client tự hiển thị thay vì phải đoán qua lỗi parse JSON.
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Lỗi không bắt được ở %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Lỗi server không mong muốn ({type(exc).__name__}: {exc})"},
    )


def _row_to_result(row, thumb_path: str, answer_text: str | None = None) -> SearchResultRow:
    return SearchResultRow(
        video_id=row["video_id"],
        frame_id=int(row["frame_id"]) if "frame_id" in row else None,
        score=float(row["score"]) if "score" in row and row["score"] is not None else None,
        thumb_url=f"/api/thumb?path={thumb_path}",
        answer_text=answer_text,
    )


@app.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    log = StepLog()
    filt = req.filters.model_dump(exclude_none=True)
    # 2026-08-21 (theo yêu cầu người dùng: "làm cái canvas vẽ OCR và Object như bên streamlit
    # v3") - canvas TOÀN CỤC (key_suffix="global" bên v3) chỉ áp dụng cho KIS/Q&A - xem
    # backend/core/spatial_canvas.py + SearchRequest.boxes.
    if req.boxes:
        filt["spatial_boxes"] = boxes_to_spatial(req.boxes, req.canvas_w, req.canvas_h)
        filt["spatial_op"] = req.spatial_op
    try:
        if req.use_llm_entity and req.query:
            plan = extract_entities(req.query, log=log)
            filt.setdefault("must_have_labels", plan.get("resolved_must_have_labels"))
            filt.setdefault("min_count", plan.get("resolved_min_count"))

        if req.mode in ("kis", "qa"):
            if not req.query:
                raise HTTPException(400, "Thiếu 'query' cho mode kis/qa")
            result = search_dense(
                req.query, req.dense_model, top_k=req.top_k,
                ocr_algorithm=req.ocr_algorithm, score_algorithm=req.score_algorithm,
                distill_model=req.distill_model, multi_clause=req.multi_clause, log=log, **filt,
            )
            # 2026-08-21 (theo yêu cầu người dùng, y hệt v3 answer_qa) - use_lvlm MẶC ĐỊNH False
            # (tốn phí/lần gọi API NIM) - chỉ gọi VQA thật cho top vqa_top_n ứng viên khi người
            # dùng CHỦ ĐỘNG bật checkbox "Dùng LVLM tự động trả lời", rank thấp hơn dùng lại
            # answer của rank 1. Tắt (mặc định) -> answer_text=None cho mọi dòng, tự gõ tay.
            answers: list[str | None] = [None] * len(result)
            if req.mode == "qa" and req.use_lvlm and req.question and not result.empty:
                answers = answer_for_results(
                    result, req.question, req.vqa_top_n, model=req.vlm_model, log=log,
                )
            rows = [
                _row_to_result(r, r["path"], answers[i]) for i, (_, r) in enumerate(result.iterrows())
            ]
        elif req.mode in ("trake", "temporal"):
            # TRAKE và Temporal dùng CHUNG luồng tìm kiếm nhiều mốc (giống hệt v3
            # `is_temporal = mode in ("Temporal", "TRAKE")`) - chỉ khác lúc NỘP BÀI (frontend
            # rút chuỗi mốc về 1 frame median cho Temporal, xem app.js::addToSubmission).
            if not req.anchors:
                raise HTTPException(400, f"Thiếu 'anchors' cho mode {req.mode}")
            # Canvas RIÊNG/mốc (giống v3: mỗi mốc 1 objects_canvas riêng) - anchor nào không vẽ
            # gì thì gửi thẳng string (dense_temporal._normalize_anchor chấp nhận cả 2 dạng).
            # anchor_spatial_op: AND/OR RIÊNG/mốc (theo yêu cầu người dùng, KHÔNG có ở v3).
            anchor_boxes = req.anchor_boxes or [None] * len(req.anchors)
            anchor_ops = req.anchor_spatial_op or [None] * len(req.anchors)
            anchors_arg: list = []
            for text, boxes, op in zip(req.anchors, anchor_boxes, anchor_ops):
                if boxes or op:
                    entry: dict = {"text": text}
                    if boxes:
                        entry["spatial_boxes"] = boxes_to_spatial(boxes, req.canvas_w, req.canvas_h)
                    if op:
                        entry["spatial_op"] = op
                    anchors_arg.append(entry)
                else:
                    anchors_arg.append(text)
            result = dense_temporal.search(
                anchors_arg, top_k=req.top_k, dense_model=req.dense_model,
                ocr_algorithm=req.ocr_algorithm, score_algorithm=req.score_algorithm,
                distill_model=req.distill_model, spatial_op=req.spatial_op,
                max_gap_seconds=req.max_gap_seconds, log=log, **{
                    k: v for k, v in filt.items() if k in (
                        "must_have_labels", "min_count", "ocr_text", "asr_text",
                        "video_ids",
                    )
                },
            )
            n = len(req.anchors)
            rows = []
            for _, r in result.iterrows():
                frame_ids = [int(r[f"anchor{i}_frame_id"]) for i in range(n)]
                # 2026-08-21 (theo yêu cầu người dùng: "kết quả... không giống bên Streamlit v3")
                # - v3 hiện MỖI mốc 1 ảnh RIÊNG (không phải 1 ảnh chung đại diện cả chuỗi) - lấy
                # đúng path của TỪNG anchor{i}_path thay vì chỉ anchor0.
                thumbs = [f"/api/thumb?path={r[f'anchor{i}_path']}" for i in range(n)]
                pts_times = [float(r[f"anchor{i}_pts_time"]) for i in range(n)]
                rows.append(SearchResultRow(
                    video_id=r["video_id"], frame_ids=frame_ids,
                    thumb_url=thumbs[0], thumb_urls=thumbs, pts_times=pts_times,
                    score=float(r["score"]) if "score" in r and r["score"] is not None else None,
                ))
        else:
            raise HTTPException(400, f"mode không hợp lệ: {req.mode!r}")

        return SearchResponse(rows=rows, step_log=log.steps)
    except (ModalTimeoutError, ModalUnavailableError, NIMTimeoutError) as e:
        return SearchResponse(rows=[], step_log=log.steps, error=str(e))
    except HTTPException:
        raise  # lỗi request hợp lệ (400 thiếu query/anchors) - giữ nguyên status code, không nuốt
    except Exception as e:
        # 2026-08-21 - trước đây MỌI lỗi khác (dense_model sai, KeyError dữ liệu bất thường...)
        # rơi thẳng ra ngoài thành 500 trắng, mất luôn step_log đã chạy được tới đó - giờ trả
        # SearchResponse hợp lệ (200) với error message, y hệt đường xử lý Modal/NIM ở trên, để
        # UI hiện lỗi NGAY TRONG khung kết quả (renderSteplog đọc data.error) thay vì crash cả
        # trang. Vẫn log traceback đầy đủ ở server để debug.
        logger.exception("Lỗi search (mode=%s)", req.mode)
        return SearchResponse(rows=[], step_log=log.steps, error=f"{type(e).__name__}: {e}")


@app.post("/api/vlm_verify", response_model=VlmVerifyResponse)
def vlm_verify(req: VlmVerifyRequest) -> VlmVerifyResponse:
    """Xác minh OCR bằng VLM cho 1 frame cụ thể - lazy, on-demand (nút "🔍 VLM Verify" dưới mỗi
    kết quả), xem backend/core/vlm_verify.py."""
    p = Path(req.path)
    if not p.exists():
        raise HTTPException(404, "Không tìm thấy ảnh")
    try:
        text = vlm_read_text(str(p), model=req.model or DEFAULT_VLM_OCR_MODEL)
        return VlmVerifyResponse(text=text)
    except NIMTimeoutError as e:
        return VlmVerifyResponse(text="", error=str(e))


@app.get("/api/shot_clip")
def shot_clip(video_id: str, frame_id: int):
    """Cắt + trả về đoạn video ĐÚNG SHOT chứa frame_id (ranh giới lấy từ dense_meta.parquet
    shot_idx, xem tiers/dense_search.py::get_shot_frame_range) — giống hệt nút "🎬" bên v3
    (video_clip.py::get_shot_clip_bytes). Lazy on-demand, có cache đĩa (app/.cache/video_clips)."""
    try:
        clip_bytes = get_shot_clip_bytes(video_id, frame_id)
    except Exception as e:
        raise HTTPException(500, f"Không cắt được video ({e})") from e
    return Response(content=clip_bytes, media_type="video/mp4")


@app.get("/api/video_meta", response_model=VideoMetaResponse)
def video_meta(video_id: str):
    """fps + frame lớn nhất của video - cho Playback giới hạn slider/nudge (giống
    `_fps_by_video()`/`_frame_idx_by_video()` bên v3, xem _playback_dialog)."""
    try:
        fps = _fps_by_video().get(video_id, 25.0)
        arr = _frame_idx_by_video().get(video_id)
        max_frame = int(arr.max()) if arr is not None and len(arr) else 0
    except Exception as e:
        raise HTTPException(500, f"Không đọc được metadata video ({e})") from e
    return VideoMetaResponse(fps=fps, max_frame=max_frame)


@app.get("/api/frame")
def frame(video_id: str, t: float):
    """Trích 1 frame DUY NHẤT tại đúng giây `t` (không giới hạn theo các frame dense-sampled có
    sẵn) - cho Playback xem thử khi kéo/nudge/gõ số (giống extract_single_frame bên v3)."""
    try:
        p = extract_single_frame(video_id, t)
    except Exception as e:
        raise HTTPException(500, f"Không trích được frame xem trước ({e})") from e
    return FileResponse(p, media_type="image/jpeg")


@app.get("/api/video")
def video(video_id: str, request: Request):
    """Video gốc TOÀN BỘ (không cắt) - cho panel "xem tự do quanh đó" bên phải Playback (giống
    `st.video(read_video_bytes(video_id))` bên v3).

    2026-08-21 (bug thật: "thanh thời gian của video không thể kéo qua về cũng như chọn thời
    điểm", kèm "tự nhiên về frame 0") - trả `Response(content=...)` nguyên khối như trước là
    HTTP 200 KHÔNG hỗ trợ Range: đã kiểm chứng bằng curl gửi `Range: bytes=0-1023` -> server vẫn
    đáp 200 + nguyên 154MB, KHÔNG có `accept-ranges`/206. Trình duyệt CHỈ tua được video khi
    server đáp 206 Partial Content, nên: (1) thanh tua chết cứng, (2) `videoEl.currentTime = t`
    của Playback bị bỏ qua âm thầm, currentTime kẹt 0 -> "timeupdate" bắn 0 -> code đồng bộ 2
    chiều hiểu nhầm người dùng vừa tua về 0, ghi đè frame về 0. Cả 2 triệu chứng CHUNG 1 gốc.

    Video đã giải nén sẵn ra đĩa -> dùng FileResponse (Starlette tự xử lý Range -> 206, stream
    theo chunk, KHÔNG nạp cả 154MB vào RAM mỗi request như trước). Chỉ khi chưa giải nén (còn
    trong zip) mới fallback đọc bytes rồi tự cắt Range bằng tay."""
    local_path = local_video_path(video_id)
    if local_path is not None:
        return FileResponse(local_path, media_type="video/mp4")

    try:
        video_bytes = read_video_bytes(video_id)
    except Exception as e:
        raise HTTPException(500, f"Không tải được video gốc ({e})") from e

    total = len(video_bytes)
    range_header = request.headers.get("range")
    if not range_header or not range_header.startswith("bytes="):
        return Response(
            content=video_bytes, media_type="video/mp4",
            headers={"accept-ranges": "bytes", "content-length": str(total)},
        )
    # "bytes=START-END" (END có thể trống = tới hết file)
    start_s, _, end_s = range_header.removeprefix("bytes=").partition("-")
    start = int(start_s) if start_s else 0
    end = int(end_s) if end_s else total - 1
    start = max(0, start)
    end = min(end, total - 1)
    if start > end:
        raise HTTPException(416, "Range không hợp lệ")
    chunk = video_bytes[start:end + 1]
    return Response(
        content=chunk, status_code=206, media_type="video/mp4",
        headers={
            "accept-ranges": "bytes",
            "content-range": f"bytes {start}-{end}/{total}",
            "content-length": str(len(chunk)),
        },
    )


@app.get("/api/thumb")
def thumb(path: str):
    p = Path(path)
    if not p.exists():
        raise HTTPException(404, "Không tìm thấy ảnh")
    return FileResponse(p, media_type="image/jpeg")


@app.get("/api/labels")
def labels():
    """514 nhãn tiếng Việt closed-set (label_vi.json) — cho dropdown "Object" trên canvas vẽ
    khung (xem static/js/canvas.js), y hệt list_labels_vi() bên v3."""
    try:
        return JSONResponse({"labels": list_labels_vi()})
    except Exception as e:
        raise HTTPException(500, f"Không đọc được danh sách nhãn ({e})") from e


@app.get("/api/health")
def health():
    return JSONResponse({"ok": True, "ts": time.time()})


# 2026-08-21 (bug that: mount o "/" khien "/static/css/..." -> tim static/static/css/... (2 lan
# "static", khong ton tai) - index.html/app.js dang tham chieu CSS/JS qua tien to "/static/..."
# nen phai mount dung tai "/static", roi khai bao rieng route GET "/" tra ve index.html.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
