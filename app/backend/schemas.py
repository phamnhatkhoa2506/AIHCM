"""Pydantic request/response schema cho API search + submission. Field name bám sát tham số
gốc của search_dense()/dense_temporal.search() (share/tiers/) để backend chỉ cần **filters
truyền thẳng, không phải map lại tên."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Filters(BaseModel):
    authors: list[str] | None = None
    date_from: str | None = None
    date_to: str | None = None
    keywords_any: list[str] | None = None
    video_ids: list[str] | None = None
    ocr_text: str | None = None
    asr_text: str | None = None


class CanvasBox(BaseModel):
    """1 khung vẽ trên canvas OCR/Object — px thô trong không gian canvas_w x canvas_h (giống
    hệt objects_canvas_component/index.html bên v3, xem backend/core/spatial_canvas.py)."""
    x0: float
    y0: float
    x1: float
    y1: float
    kind: str = "ocr"       # "ocr" | "object"
    text: str = ""          # kind="ocr"
    label: str = ""         # kind="object" — nhãn tiếng Việt đã chọn từ dropdown 514 nhãn
    minCount: int = 1       # kind="object"


class SearchRequest(BaseModel):
    mode: str = Field(description='"kis" | "qa" | "trake" | "temporal"')
    query: str | None = None          # kis: câu truy vấn. qa: "Mô tả sự kiện" (dùng để retrieval)
    question: str | None = None       # qa: "Câu hỏi" — CHỈ dùng khi gọi VQA thật (use_lvlm=True)
    anchors: list[str] | None = None  # trake/temporal: mô tả từng mốc, thứ tự = thứ tự mốc
    # 2026-08-21 (theo yêu cầu người dùng: canvas riêng/mốc cho TRAKE/Temporal, giống hệt v3
    # "mỗi mốc 1 objects_canvas riêng") - song song 1-1 với `anchors` (None = mốc đó không vẽ gì).
    anchor_boxes: list[list[CanvasBox] | None] | None = None
    # 2026-08-21 (theo yêu cầu người dùng: "quan hệ riêng cho từng box") - AND/OR RIÊNG cho từng
    # mốc (song song 1-1 với anchors), None/thiếu = dùng chung `spatial_op` toàn cục bên dưới.
    anchor_spatial_op: list[str | None] | None = None
    # 2026-08-21 (theo yêu cầu người dùng, y hệt v3) - ràng buộc CỨNG (không phải phạt điểm): mốc
    # sau phải cách mốc trước không quá N giây, quá thì loại hẳn khỏi xét dù điểm khớp cao.
    max_gap_seconds: float = 25.0     # trake/temporal - khớp dense_temporal.MAX_ANCHOR_GAP_SECONDS
    dense_model: str = "siglip"       # siglip | pe_core | beit3 | rrf
    top_k: int = 100
    ocr_algorithm: str = "flexible"
    score_algorithm: str = "cosine"
    distill_model: str | None = None
    multi_clause: bool = False        # tách câu theo dấu chấm, chưng cất+xếp hạng riêng từng mệnh đề, lấy MAX
    use_llm_entity: bool = False      # bật lại LLM entity hard-filter (mặc định tắt, xem app_flags.py)
    use_region_clip_rerank: bool = False
    # 2026-08-21 (theo yêu cầu người dùng, y hệt v3 online/app.py::_render_qa_query_inline) -
    # QA MẶC ĐỊNH TẮT (tốn phí/lần gọi API NIM) - answer để trống, người tự gõ trước khi nộp.
    use_lvlm: bool = False
    vqa_top_n: int = 3                # chỉ N ứng viên đầu được gọi VQA thật, còn lại dùng lại answer rank 1
    # 2026-08-21 (theo yêu cầu người dùng: "Mô hình trả lời Q&A dùng chung với mô hình đọc OCR
    # luôn") - CÙNG 1 dropdown/model với "Model VLM đọc chữ (xác minh OCR)" trong sidebar (xem
    # vqa.py) - không có lựa chọn model riêng cho VQA nữa.
    vlm_model: str | None = None
    # 2026-08-21 (theo yêu cầu người dùng: "làm cái canvas vẽ OCR và Object như bên streamlit
    # v3") - canvas TOÀN CỤC (giống key_suffix="global" bên v3) - chỉ áp dụng cho KIS/Q&A. TRAKE/
    # Temporal có canvas RIÊNG/mốc qua anchor_boxes bên dưới (xem backend/core/spatial_canvas.py).
    boxes: list[CanvasBox] | None = None
    canvas_w: int = 480
    canvas_h: int = 270
    spatial_op: str = "and"  # "and" | "or"
    filters: Filters = Filters()


class SearchResultRow(BaseModel):
    video_id: str
    frame_id: int | None = None
    frame_ids: list[int] | None = None  # trake/temporal: 1 phần tử/mốc
    score: float | None = None
    thumb_url: str
    # 2026-08-21 (theo yêu cầu người dùng: "kết quả... không giống bên Streamlit v3" - v3 hiện
    # MỖI mốc 1 ảnh riêng + timeline, không phải 1 ảnh chung cho cả chuỗi) - trake/temporal: 1
    # thumb/mốc (song song với frame_ids); None (kis/qa) = chỉ dùng thumb_url.
    thumb_urls: list[str] | None = None
    pts_times: list[float] | None = None  # trake/temporal: giây trong video gốc, 1 phần tử/mốc (cho timeline)
    answer_text: str | None = None      # qa: chỉ có giá trị nếu use_lvlm=True (xem SearchRequest)
    # 2026-08-21 (theo yêu cầu người dùng: "khi lọc bằng ASR... không biết cụ thể nội dung toàn
    # câu... chữ được khớp sẽ được bôi màu trong câu tương ứng") - CHỈ có giá trị khi filters.
    # asr_text đang lọc VÀ frame này thật sự khớp 1 đoạn ASR (xem dense_search.py::
    # _annotate_asr_match) - asr_match_start/end là OFFSET KÝ TỰ trong asr_match_text để frontend
    # tô đậm đúng đoạn khớp, không phải tô cả câu.
    asr_match_text: str | None = None
    asr_match_start: int | None = None
    asr_match_end: int | None = None


class SearchResponse(BaseModel):
    rows: list[SearchResultRow]
    step_log: list[dict] = []
    error: str | None = None


class VlmVerifyRequest(BaseModel):
    path: str            # đường dẫn tuyệt đối tới ảnh frame (lấy từ thumb_url của SearchResultRow)
    model: str | None = None  # key trong vlm_verify.VLM_OCR_MODELS, None = dùng mặc định


class VlmVerifyResponse(BaseModel):
    text: str
    error: str | None = None


class SubmissionItem(BaseModel):
    video_id: str
    frame_id: int | None = None
    frame_ids: list[int] | None = None
    answer_text: str | None = None  # Q&A
    mode: str | None = None  # "kis" | "qa" | "trake" — chỉ để HIỂN THỊ tag [KIS]/[QA]/[TRAKE] trong panel


class SubmissionBucket(BaseModel):
    query_key: str
    mode: str
    items: list[SubmissionItem]


class AutofillRequest(BaseModel):
    items: list[SubmissionItem]  # kết quả đang hiển thị, ĐÚNG thứ tự rank hiện tại (top -> thấp)


class AutofillResponse(BaseModel):
    added: int
    items: list[SubmissionItem]


class VideoMetaResponse(BaseModel):
    """Cho Playback (xem app.js::openPlayback) - giới hạn slider/nudge frame theo ĐÚNG video,
    khớp `_fps_by_video()`/`_frame_idx_by_video()` bên v3."""
    fps: float
    max_frame: int
