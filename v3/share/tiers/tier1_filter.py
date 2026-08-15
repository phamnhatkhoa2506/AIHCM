"""TẦNG 1 — Lọc thô: định nghĩa tập ứng viên (video_id, local_idx) TRƯỚC khi vào Tầng 2.
Chỉ include/exclude, KHÔNG tính điểm — filter càng chặt thì Tầng 2 càng ít việc.

3 nguồn lọc, dùng riêng hoặc cùng lúc:
  - by_metadata(): video_metadata.parquet (kênh/ngày đăng/từ khoá) — lọc theo VIDEO.
  - by_objects():  objects_index.parquet (nhãn OpenImages V4, đã lọc score>=0.3) — lọc theo FRAME.
  - by_text():     ocr_text.parquet (chữ trên màn hình, OCR) — lọc theo FRAME.

apply() gộp cả 3 (AND) thành 1 kết quả duy nhất. Trả về:
  - None                         -> không lọc gì, Tầng 2 tìm trên toàn corpus.
  - set[(video_id, local_idx)]   -> đúng tập frame được phép (có thể rỗng nếu bộ lọc quá hẹp).
"""
from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd

import resources


def _strip_accents(s: str) -> str:
    """Bỏ dấu tiếng Việt + hạ chữ thường — dùng để so khớp 'mon ngon' với 'Món Ngon'.
    keywords_text đã có sẵn bản sao không dấu trộn lẫn (dữ liệu gốc từ YouTube tags), nhưng
    title thì KHÔNG — nếu chỉ .lower() thôi thì gõ không dấu sẽ không khớp được title."""
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d")  # "đ" không tách qua NFD như các ký tự có dấu phụ khác


def _ordered_words_match(needle: str, haystack_norm: str) -> bool:
    """True nếu các TỪ trong `needle` (đã strip_accents) xuất hiện ĐÚNG THỨ TỰ trong
    `haystack_norm` (SUBSEQUENCE theo từ — KHÔNG bắt buộc liền kề/sát nhau nữa).

    SỬA (2026-08-15, theo yêu cầu người dùng - bug thật phát hiện qua test: query OCR "giá vàng
    tăng" ra 0 kết quả dù có 169 dòng OCR thật chứa cả 3 từ đó, vì bản tin luôn chèn thêm chữ ở
    giữa như "giá vàng NHÃN SÁNG 13/8 tăng" — thuật toán CŨ đòi hỏi 3 từ đứng SÁT NHAU tuyệt
    đối, quá chặt so với văn phong thật). GIỮ NGUYÊN thứ tự (không phải bag-of-words) để tránh
    khớp nhầm câu đảo nghĩa (vd "tăng giá vàng" != "giá vàng tăng"), chỉ nới lỏng yêu cầu SÁT
    NHAU. So khớp theo TỪ NGUYÊN VẸN (==, không phải substring ký tự) — tự động tránh luôn bug
    "áo" lọt vào giữa "báo" mà bản substring cũ phải dùng mẹo đệm khoảng trắng để né."""
    needle_words = _strip_accents(needle).split()
    if not needle_words:
        return True
    haystack_words = haystack_norm.split()
    it = iter(haystack_words)
    return all(nw in it for nw in needle_words)


def by_metadata(
    authors: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    keywords_any: list[str] | None = None,
) -> set[str] | None:
    """Lọc theo video_metadata (kênh/ngày/từ khoá/tên chương trình) -> tập video_id được phép.

    keywords_any so khớp CẢ title lẫn keywords_text (substring, không dấu) — gộp làm 1 ô
    "từ khoá" duy nhất ở UI thay vì tách riêng tìm-theo-tên-chương-trình, vì ở quy mô 873
    video thì không cần bộ máy full-text search riêng (BM25/inverted index) — substring quét
    toàn bộ đã đủ nhanh (<10ms) và đây là lọc CỨNG (có/không), không phải xếp hạng liên quan."""
    if not authors and not date_from and not date_to and not keywords_any:
        return None

    df = resources.get().video_meta
    if authors:
        df = df[df["author"].isin(authors)]
    if date_from:
        df = df[df["publish_date"] >= pd.to_datetime(date_from)]
    if date_to:
        df = df[df["publish_date"] <= pd.to_datetime(date_to)]
    if keywords_any:
        # dem khoang trang 2 dau ca needle va haystack de containment check roi dung ranh gioi
        # TU, khong phai ky tu tho - bug that phat hien 2026-08-09: "tin" (tin tuc) lot vao
        # giua "tinh" trong "tinh nguyen" (tinh nguyen = volunteer, khong lien quan tin tuc)
        # sau khi bo dau, giong het lop bug da vay o label_translate.py ("ao" lot vao "bao").
        needles = [f" {_strip_accents(k)} " for k in keywords_any]
        haystack = " " + df["title"].apply(_strip_accents) + " " + df["keywords_text"].apply(_strip_accents) + " "
        df = df[haystack.apply(lambda h: any(n in h for n in needles))]
    return set(df["video_id"])


def by_objects(
    must_have_labels: list[str] | None = None,
    min_count: dict[str, int] | None = None,
    use_suppression: bool = True,
    include_open_vocab: bool = True,
) -> set[tuple[str, int]] | None:
    """Lọc theo Objects (nhãn OpenImages V4 tiếng Anh) -> tập (video_id, local_idx) được phép.

    Đây là bộ lọc CỨNG, đúng nguyên tắc: filter định nghĩa tập ứng viên TRƯỚC, Tầng 2 (vector)
    chỉ xếp hạng TRONG tập đó — không phải vector trước rồi giao (mất recall với query có
    ràng buộc chặt, vd "2 người" mà CLIP không xếp hạng cao).
    """
    if not must_have_labels and not min_count:
        return None

    objects_index = resources.get().objects_index
    if objects_index is None:
        raise RuntimeError("objects_index.parquet chưa build — chạy build_objects_index.py trước")

    need: dict[str, int] = {}
    for lb in must_have_labels or []:
        need[lb] = max(need.get(lb, 1), 1)
    for lb, c in (min_count or {}).items():
        need[lb] = max(need.get(lb, 1), c)

    # include_open_vocab=False -> CHI khop tren closed_set (514 nhan goc BTC), bo qua han
    # nhan open_vocab_dino - dung de so sanh/fallback ve dung hanh vi truoc khi co Grounding
    # DINO. Khac voi use_suppression: cai do chi tat co che LOAI closed_set sai, van cho phep
    # open_vocab tham gia khop; cai nay tat HAN open_vocab, khong lien quan suppress.
    scope = objects_index if include_open_vocab else objects_index[objects_index["source"] == "closed_set"]

    # cot "suppressed" (2026-08-07): nhan closed_set nao bi open-vocab (Grounding DINO) de len
    # (IoU cao, cung vi tri) coi nhu SAI, khong duoc dung de khop nua - xem
    # resources._mark_suppressed_closed_rows(). Dong van con nguyen (audit), chi loai khoi day.
    # use_suppression=False -> tat han co che nay (fallback ve hanh vi cu, dung khi muon so
    # sanh/nghi ngo suppress sai) - KHONG can build lai gi, chi doi 1 cot boolean luc filter.
    not_suppressed = ~scope["suppressed"] if use_suppression else True

    allowed: set[tuple[str, int]] | None = None
    for lb, min_c in need.items():
        sub = scope[(scope["label"] == lb) & not_suppressed]
        cnt = sub.groupby(["video_id", "local_idx"]).size()
        keys = set(cnt[cnt >= min_c].index)
        allowed = keys if allowed is None else (allowed & keys)
        if not allowed:
            return set()
    return allowed


def by_text(
    ocr_text: str | None = None,
    ocr_region: tuple[float, float, float, float] | None = None,
    ocr_region_iou_threshold: float = 0.05,
) -> set[tuple[str, int]] | None:
    """Lọc theo chữ trên màn hình (OCR) -> tập (video_id, local_idx) được phép.

    Filter CỨNG (AND với object/metadata) — khác secondary_entities (soft-boost) vì khớp
    chuỗi ký tự chính xác gần như không có false-positive, đáng tin hơn hẳn object detection
    (score mờ) hay CLIP (score clustering đã đo). Nếu user chắc chắn có chữ "X", ứng viên
    KHÔNG có chữ đó bị loại thẳng, không chỉ giảm điểm.

    ocr_region: (ymin,xmin,ymax,xmax) chuẩn hoá [0,1] — vùng user khoanh tay trên canvas
    (xem app.py, streamlit-drawable-canvas). ocr_region_iou_threshold THẤP (0.05, không phải
    0.5 như suppression) vì đây là ước lượng thô "khoảng đó" của tay người, không phải 2 bbox
    object thật chồng khít — ngưỡng cao sẽ loại oan kết quả đúng chỉ vì tay vẽ hơi lệch."""
    if not ocr_text and ocr_region is None:
        return None

    ocr_index = resources.get().ocr_index
    if ocr_index is None:
        raise RuntimeError("ocr_text.parquet chưa build — chạy build_ocr_index.py trước")

    df = ocr_index
    if ocr_text:
        df = df[df["text_norm"].apply(lambda t: _ordered_words_match(ocr_text, t))]
    if ocr_region is not None and len(df):
        boxes = df[["ymin", "xmin", "ymax", "xmax"]].to_numpy(dtype=float)
        iou = resources._box_iou_matrix(boxes, np.array([ocr_region], dtype=float))
        df = df[iou.flatten() > ocr_region_iou_threshold]

    allowed: set[tuple[str, int]] = set()
    for row in df.itertuples(index=False):
        for li in range(row.local_idx_start, row.local_idx_end + 1):
            allowed.add((row.video_id, li))
    return allowed


def apply(
    authors: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    keywords_any: list[str] | None = None,
    must_have_labels: list[str] | None = None,
    min_count: dict[str, int] | None = None,
    use_suppression: bool = True,
    include_open_vocab: bool = True,
    ocr_text: str | None = None,
    ocr_region: tuple[float, float, float, float] | None = None,
) -> set[tuple[str, int]] | None:
    """Gộp by_metadata() + by_objects() + by_text() thành 1 tập (video_id, local_idx) duy
    nhất — Tầng 2 chỉ cần đọc kết quả này, không cần biết filter tới từ nguồn nào."""
    video_allowed = by_metadata(authors, date_from, date_to, keywords_any)
    frame_allowed = by_objects(must_have_labels, min_count, use_suppression, include_open_vocab)
    text_allowed = by_text(ocr_text, ocr_region)

    frame_sets = [s for s in (frame_allowed, text_allowed) if s is not None]
    if video_allowed is None and not frame_sets:
        return None

    if frame_sets:
        combined = frame_sets[0]
        for s in frame_sets[1:]:
            combined = combined & s
        if video_allowed is None:
            return combined
        return {(v, l) for v, l in combined if v in video_allowed}

    row_pos = resources.get().row_pos
    assert video_allowed is not None
    return {(vid, li) for vid, li in row_pos if vid in video_allowed}
