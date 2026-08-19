"""Orchestrator — gọi tuần tự 4 tầng (xem tiers/__init__.py):

    Tầng 1 (lọc thô) -> Tầng 2 (CLIP, đang dùng) -> Tầng 3 (temporal, stub) -> Tầng 4 (graph, stub)

Hệ thống hiện đang dừng ở Tầng 2 — Tầng 3/4 là tham số optional, mặc định None (pass-through),
để sẵn chỗ cắm khi triển khai mà không phải sửa lại chữ ký hàm `search()` hay code gọi nó
(app.py) sau này.
"""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/

import pandas as pd

from steplog import StepLog
from tiers import tier1_filter, tier2_vector, tier3_temporal, tier4_graph


def search(
    query: str,
    top_k: int = 10,
    *,
    # Tầng 1 — lọc thô
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
    # Tầng 3 — temporal (chưa triển khai, để sẵn tham số)
    anchors: list[str | dict] | None = None,
    # Tầng 4 — graph (chưa triển khai, để sẵn tham số)
    relations: list[dict] | None = None,
    log: StepLog | None = None,
) -> pd.DataFrame:
    """Trả về top_k frame khớp nhất với query text, sắp theo score giảm dần.

    authors/date_from/date_to/keywords_any/must_have_labels/min_count: lọc cứng Tầng 1
    (video_metadata + Objects) — xem tiers/tier1_filter.py.
    anchors/relations: tham số dành cho Tầng 3/4 — hiện raise NotImplementedError nếu truyền
    vào (không âm thầm bỏ qua ràng buộc), None thì bỏ qua bình thường.
    log: truyền StepLog() để ghi lại từng bước (tên/chi tiết/thời gian) — xem steplog.py.
    """
    if anchors:
        # Temporal: mỗi anchor cần pool riêng (Tầng 1+2 chạy lại cho từng anchor) — không
        # phải hậu xử lý kết quả 1-query, nên đi thẳng vào tier3_temporal.search(), bỏ qua
        # nhánh 1-query bên dưới. `query` không dùng ở nhánh này.
        result = tier3_temporal.search(
            anchors,
            top_k=top_k,
            authors=authors,
            date_from=date_from,
            date_to=date_to,
            keywords_any=keywords_any,
            must_have_labels=must_have_labels,
            min_count=min_count,
            use_suppression=use_suppression,
            include_open_vocab=include_open_vocab,
            ocr_text=ocr_text,
            ocr_region=ocr_region,
            log=log,
        )
    else:
        if log:
            with log.timed("Tầng 1 — lọc thô") as set_detail:
                candidates = tier1_filter.apply(
                    authors=authors,
                    date_from=date_from,
                    date_to=date_to,
                    keywords_any=keywords_any,
                    must_have_labels=must_have_labels,
                    min_count=min_count,
                    use_suppression=use_suppression,
                    include_open_vocab=include_open_vocab,
                    ocr_text=ocr_text,
                    ocr_region=ocr_region,
                )
                set_detail("không lọc (toàn corpus)" if candidates is None else f"{len(candidates)} frame ứng viên")
        else:
            candidates = tier1_filter.apply(
                authors=authors,
                date_from=date_from,
                date_to=date_to,
                keywords_any=keywords_any,
                must_have_labels=must_have_labels,
                min_count=min_count,
                use_suppression=use_suppression,
                include_open_vocab=include_open_vocab,
                ocr_text=ocr_text,
                ocr_region=ocr_region,
            )

        if log:
            with log.timed("Tầng 2 — encode query (CLIP multilingual)") as set_detail:
                query_vec = tier2_vector.encode_query(query)
                set_detail(f'"{query}"')
        else:
            query_vec = tier2_vector.encode_query(query)

        if log:
            with log.timed("Tầng 2 — xếp hạng cosine") as set_detail:
                result = tier2_vector.rank(query_vec, candidates, top_k)
                set_detail(f"top_k={top_k}, trả về {len(result)} dòng")
        else:
            result = tier2_vector.rank(query_vec, candidates, top_k)

    result = tier4_graph.apply(result, relations)
    return result


if __name__ == "__main__":
    import sys

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    query = " ".join(sys.argv[1:]) or "một người phụ nữ mặc áo dài đỏ"
    print(f"Query: {query}\n")
    result = search(query, top_k=10)
    print(result.to_string(index=False))
