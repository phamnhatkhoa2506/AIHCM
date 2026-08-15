"""TANG 3 (TRAKE/temporal) - MIGRATE sang BO DENSE (2026-08-15, theo yeu cau nguoi dung "tan
dung khung keo tha tu KIS/Q&A cho Temporal"). Truoc day gac lai (xem tier3_temporal.py cu, van
giu nguyen KHONG doi cho BTC) vi tuong dense_meta.parquet thieu pts_time/fps - THUC RA khong
con la rao can: pts_time = frame_id / fps, fps da co san qua dense_search._fps_by_video()
(dung chung voi ASR boost, khong can rebuild gi ca).

Thuat toan ANCHOR-CHAIN qua QUY HOACH DONG (DP) - GIU NGUYEN nguyen tac voi tier3_temporal.py
cu (Tang khac Tang 2 o cho KHONG dung 1 query duy nhat - moi anchor chay search_dense() RIENG
de co 1 pool ung vien, sau do join theo video + thoi gian, khong phai hau xu ly ket qua 1-query
co san), CHI khac nguon du lieu/ham goi (search_dense thay vi tier1_filter+tier2_vector).

Khung ve tay (spatial_boxes tu canvas, xem app.py): SUA 2026-08-15 (theo yeu cau nguoi dung
"co the them dropbox tuy vao so luong event chu nhi") - gio MOI anchor co 1 canvas RIENG cua
no (app.py::_render_filter_canvas goi 1 lan/moc, KHONG con dung 1 canvas chung nua). spatial_boxes
tham so global cua search()/_run_anchor_pool VAN giu de tuong thich nguoc (vd goi tu noi khac
voi 1 bo khung chung cho ca chuoi) nhung app.py hien LUON truyen None o do - khung THAT DUNG
duoc lay tu anchor["spatial_boxes"] (per-anchor), MERGE (khong loai) voi spatial_boxes global
neu co ca 2. Canvas rieng cho tung moc nam TRUC TIEP trong luong trang chinh (khong bo trong
expander) de tranh bug mount height=0 cua streamlit-drawable-canvas (xem app.py docstring).
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np
import pandas as pd

from steplog import StepLog
from tiers.dense_search import _fps_by_video, search_dense

DEFAULT_COARSE_K = 1000


def _run_anchor_pool(
    anchor_text: str,
    dense_model: str,
    coarse_k: int,
    *,
    must_have_labels: list[str] | None,
    min_count: dict[str, int] | None,
    ocr_text: str | None,
    spatial_boxes: list[dict] | None,
    spatial_op: str,
    log: StepLog | None,
) -> dict[tuple[str, int], tuple[float, str]]:
    """1 anchor -> {(video_id, frame_id): (score, path)}, coarse_k ung vien/anchor."""
    ranked = search_dense(
        anchor_text, dense_model, top_k=coarse_k,
        must_have_labels=must_have_labels, min_count=min_count, ocr_text=ocr_text,
        spatial_boxes=spatial_boxes, spatial_op=spatial_op, log=log,
    )
    return {
        (row.video_id, int(row.frame_id)): (float(row.score), row.path)
        for row in ranked.itertuples(index=False)
    }


@dataclass
class _Pick:
    frame_id: int
    pts_time: float
    score: float
    path: str


def _temporal_join(
    anchor_pools: list[dict[tuple[str, int], tuple[float, str]]],
    fps_map: dict[str, float],
    aggregate: str = "min",
) -> list[tuple[str, list[_Pick], float]]:
    """Anchor-chain qua QUY HOACH DONG (DP) - COPY NGUYEN nguyen tac tu tier3_temporal.py::
    _temporal_join (ban BTC), CHI khac cho lay pts_time = frame_id / fps thay vi doc san tu
    meta.parquet (dense khong co cot pts_time), va key la frame_id (KHONG phai local_idx - bo
    dense khong co khai niem local_idx, frame_id CHINH LA frame_idx tren truc video goc, dung
    THANG lam frame_id nop bai, xem dense_search.py docstring)."""
    n = len(anchor_pools)
    if n == 0:
        return []

    per_anchor_by_video: list[dict[str, list[tuple[float, int, float, str]]]] = []
    for pool in anchor_pools:
        by_video: dict[str, list[tuple[float, int, float, str]]] = {}
        for (video_id, frame_id), (score, path) in pool.items():
            fps = fps_map.get(video_id, 25.0)
            pts_time = frame_id / fps
            by_video.setdefault(video_id, []).append((pts_time, frame_id, score, path))
        for v in by_video:
            by_video[v].sort(key=lambda t: t[0])
        per_anchor_by_video.append(by_video)

    common_videos = set(per_anchor_by_video[0])
    for k in range(1, n):
        common_videos &= set(per_anchor_by_video[k])

    results: list[tuple[str, list[_Pick], float]] = []
    for video_id in common_videos:
        cand_by_anchor = [per_anchor_by_video[k][video_id] for k in range(n)]

        dp_val: list[list[float]] = [[] for _ in range(n)]
        back: list[list[int]] = [[] for _ in range(n)]

        dp_val[0] = [sc for (_t, _fi, sc, _p) in cand_by_anchor[0]]
        back[0] = [-1] * len(cand_by_anchor[0])

        for k in range(1, n):
            prev_times = [t for (t, *_r) in cand_by_anchor[k - 1]]
            prev_dp = dp_val[k - 1]
            prefix_max_val: list[float] = []
            prefix_max_idx: list[int] = []
            best_val, best_idx = -np.inf, -1
            for j, v in enumerate(prev_dp):
                if v > best_val:
                    best_val, best_idx = v, j
                prefix_max_val.append(best_val)
                prefix_max_idx.append(best_idx)

            cur_dp: list[float] = []
            cur_back: list[int] = []
            for (t, _fi, sc, _p) in cand_by_anchor[k]:
                pos = bisect.bisect_left(prev_times, t) - 1
                if pos < 0:
                    cur_dp.append(-np.inf)
                    cur_back.append(-1)
                    continue
                pred_val, pred_idx = prefix_max_val[pos], prefix_max_idx[pos]
                if pred_val == -np.inf:
                    cur_dp.append(-np.inf)
                    cur_back.append(-1)
                    continue
                val = min(sc, pred_val) if aggregate == "min" else sc + pred_val
                cur_dp.append(val)
                cur_back.append(pred_idx)
            dp_val[k] = cur_dp
            back[k] = cur_back

        if not dp_val[n - 1] or max(dp_val[n - 1]) == -np.inf:
            continue

        best_j = int(np.argmax(dp_val[n - 1]))
        best_val = dp_val[n - 1][best_j]
        agg = best_val if aggregate == "min" else best_val / n

        chosen_idx = [0] * n
        chosen_idx[n - 1] = best_j
        for k in range(n - 1, 0, -1):
            chosen_idx[k - 1] = back[k][chosen_idx[k]]

        chosen: list[_Pick] = []
        for k in range(n):
            t, fi, sc, p = cand_by_anchor[k][chosen_idx[k]]
            chosen.append(_Pick(frame_id=fi, pts_time=t, score=sc, path=p))
        results.append((video_id, chosen, agg))

    results.sort(key=lambda r: -r[2])
    return results


def _normalize_anchor(a: str | dict) -> dict:
    """Giong tier3_temporal.py::_normalize_anchor (ban BTC), THEM spatial_boxes rieng cho tung
    anchor (2026-08-15, canvas rieng/moc) - anchor la string (khong filter rieng) hoac dict
    {"text", "must_have_labels", "min_count", "ocr_text", "spatial_boxes"}."""
    if isinstance(a, str):
        return {"text": a, "must_have_labels": None, "min_count": None, "ocr_text": None, "spatial_boxes": None}
    return {
        "text": a["text"],
        "must_have_labels": a.get("must_have_labels"),
        "min_count": a.get("min_count"),
        "ocr_text": a.get("ocr_text"),
        "spatial_boxes": a.get("spatial_boxes"),
    }


def search(
    anchors: list[str | dict],
    top_k: int = 10,
    *,
    dense_model: str = "rrf",
    must_have_labels: list[str] | None = None,
    min_count: dict[str, int] | None = None,
    ocr_text: str | None = None,
    spatial_boxes: list[dict] | None = None,
    spatial_op: str = "and",
    coarse_k: int = DEFAULT_COARSE_K,
    aggregate: str = "min",
    log: StepLog | None = None,
    **_ignored,  # phong than: nhan them cac tham so BTC cu (authors/date_from/...) ma khong loi
    # neu app.py con truyen qua **common_filters - CHUA ho tro loc metadata rieng cho Temporal
    # (co the them sau, khong phai trong tam lan sua nay).
) -> pd.DataFrame:
    """Diem vao chinh Tang 3 tren BO DENSE - thay the hoan toan tier3_temporal.search() (BTC)
    khi dung bo dense. must_have_labels/min_count/ocr_text/spatial_boxes: filter CHUNG cho MOI
    anchor (khung ve tay tu canvas AP DUNG DONG LOAT cho tat ca moc, xem docstring dau file) -
    anchor.must_have_labels/min_count/ocr_text (dict) la filter RIENG cho DUNG moc do, AND them
    vao filter chung.

    Tra 1 dong/video, cot: video_id, score, anchor{i}_frame_id/pts_time/path."""
    if len(anchors) < 2:
        raise ValueError("Temporal cần >=2 anchor theo thứ tự thời gian (1 anchor thì dùng Tầng 2 thường)")

    anchors = [_normalize_anchor(a) for a in anchors]
    fps_map = _fps_by_video()

    anchor_pools = []
    for i, a in enumerate(anchors):
        anchor_must_have = list({*(must_have_labels or []), *(a["must_have_labels"] or [])}) or None
        anchor_min_count = {**(min_count or {}), **(a["min_count"] or {})} or None
        anchor_ocr_text = a["ocr_text"] or ocr_text
        # 2026-08-15: MERGE khung ve tay GLOBAL (thuong None tu app.py, giu de tuong thich
        # nguoc) voi khung RIENG cua moc nay (tu canvas rieng/anchor) - khong loai cai nao.
        anchor_spatial_boxes = [*(spatial_boxes or []), *(a["spatial_boxes"] or [])] or None

        if log:
            with log.timed(f"Tầng 3 — pool riêng cho anchor {i} ('{a['text']}')") as set_detail:
                pool = _run_anchor_pool(
                    a["text"], dense_model, coarse_k,
                    must_have_labels=anchor_must_have, min_count=anchor_min_count,
                    ocr_text=anchor_ocr_text, spatial_boxes=anchor_spatial_boxes, spatial_op=spatial_op,
                    log=log,
                )
                set_detail(f"{len(pool)} ứng viên (coarse_k={coarse_k})")
        else:
            pool = _run_anchor_pool(
                a["text"], dense_model, coarse_k,
                must_have_labels=anchor_must_have, min_count=anchor_min_count,
                ocr_text=anchor_ocr_text, spatial_boxes=anchor_spatial_boxes, spatial_op=spatial_op,
                log=None,
            )
        anchor_pools.append(pool)

    if log:
        with log.timed("Tầng 3 — anchor-chain join") as set_detail:
            joined = _temporal_join(anchor_pools, fps_map, aggregate=aggregate)[:top_k]
            set_detail(f"{len(joined)} video khớp đủ cả {len(anchors)} anchor đúng thứ tự")
    else:
        joined = _temporal_join(anchor_pools, fps_map, aggregate=aggregate)[:top_k]

    rows = []
    for video_id, picks, score in joined:
        row: dict = {"video_id": video_id, "score": score}
        for i, p in enumerate(picks):
            row[f"anchor{i}_frame_id"] = p.frame_id
            row[f"anchor{i}_pts_time"] = p.pts_time
            row[f"anchor{i}_path"] = p.path
        rows.append(row)

    return pd.DataFrame(rows)
