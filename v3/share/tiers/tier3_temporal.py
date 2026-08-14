"""TẦNG 3 — Temporal: thuật toán ANCHOR-CHAIN đã thiết kế trước đó (greedy earliest-feasible
theo timestamp: A -> B -> C, mỗi anchor chọn frame sớm nhất thoả sau anchor trước).

Khác Tầng 2 ở chỗ: KHÔNG dùng 1 query duy nhất. Mỗi anchor (mô tả 1 khoảnh khắc, theo đúng
thứ tự thời gian) chạy Tầng 1 + Tầng 2 RIÊNG để có 1 pool ứng viên — sau đó join theo
video + timestamp, không phải lọc/xếp hạng thêm trên 1 kết quả đã có sẵn.

Đây chính là bài toán TRAKE (BTC AIC 2026, xem `Thong tin vong So tuyen AIC2026.pdf`).

RỦI RO ĐÃ BIẾT (chưa xử lý ở bản này): cửa sổ chấm điểm TRAKE mỗi khoảnh khắc <10 frame,
nhưng keyframe BTC cấp cách nhau trung vị ~2.16s (~65 frame @30fps). Nghĩa là kể cả chọn
đúng keyframe gần nhất, có thể vẫn NGOÀI cửa sổ chấp nhận. Bản này chỉ chọn trong tập
keyframe có sẵn (chưa decode thêm frame từ video gốc) — xem ghi chú cuối file.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np
import pandas as pd

import resources
from steplog import StepLog
from tiers import tier1_filter, tier2_vector

DEFAULT_COARSE_K = 1000


def _run_anchor_pool(
    anchor_text: str,
    candidates: set[tuple[str, int]] | None,
    coarse_k: int,
) -> dict[tuple[str, int], float]:
    """1 anchor -> {(video_id, local_idx): score}, coarse_k ứng viên/anchor (Tầng 1 + Tầng 2)."""
    vec = tier2_vector.encode_query(anchor_text)
    ranked = tier2_vector.rank(vec, candidates, top_k=coarse_k)
    return {
        (row.video_id, int(row.local_idx)): float(row.score) for row in ranked.itertuples(index=False)
    }


@dataclass
class _Pick:
    local_idx: int
    frame_idx: int
    pts_time: float
    score: float


def _temporal_join(
    anchor_pools: list[dict[tuple[str, int], float]],
    aggregate: str = "min",
) -> list[tuple[str, list[_Pick], float]]:
    """Anchor-chain qua QUY HOACH DONG (DP), khong phai tham lam (sua 2026-08-11 — xem hoi
    thoai: greedy-earliest-feasible chi lay frame SOM NHAT thoa dieu kien sau moc truoc, co
    the bo lo to hop diem TONG THE cao hon vi khong xet "trade-off" giua chon som (diem thap)
    voi cho frame diem cao hon xuat hien sau. DP van GIU DUNG rang buoc thu tu thoi gian
    nghiem ngat (pts_time[k] > pts_time[k-1]), chi khac o cho TOI UU thay vi THAM LAM.

    aggregate="min": toi da hoa GIA TRI NHO NHAT trong day diem da chon (bottleneck DP —
    dp[k][j] = min(score[k][j], max qua cac tien nhiem hop le cua dp[k-1])).
    aggregate="mean" (hoac khac): toi da hoa TONG diem (dp[k][j] = score[k][j] + max qua cac
    tien nhiem hop le cua dp[k-1]) - chia n o cuoi khong doi thu tu so sanh nen khong can chia
    trong dp.

    Ca 2 kieu deu la prefix-max DP O(sum kich thuoc pool) / video (ung vien da sort theo thoi
    gian), khong phai O(n^2) tho."""
    meta = resources.get().meta
    n = len(anchor_pools)
    if n == 0:
        return []

    per_anchor_by_video: list[dict[str, list[tuple[float, int, int, float]]]] = []
    for pool in anchor_pools:
        by_video: dict[str, list[tuple[float, int, int, float]]] = {}
        for (video_id, local_idx), score in pool.items():
            row = meta.iloc[resources.get().row_pos[(video_id, local_idx)]]
            by_video.setdefault(video_id, []).append(
                (float(row.pts_time), local_idx, int(row.frame_idx), score)
            )
        for v in by_video:
            by_video[v].sort(key=lambda t: t[0])
        per_anchor_by_video.append(by_video)

    common_videos = set(per_anchor_by_video[0])
    for k in range(1, n):
        common_videos &= set(per_anchor_by_video[k])

    results: list[tuple[str, list[_Pick], float]] = []
    for video_id in common_videos:
        cand_by_anchor = [per_anchor_by_video[k][video_id] for k in range(n)]

        # dp_val[k]: gia tri DP tot nhat dat duoc tai moi ung vien cua anchor k (cung thu tu
        # da sort theo thoi gian). back[k]: chi so ung vien anchor k-1 da chon (backtrack).
        dp_val: list[list[float]] = [[] for _ in range(n)]
        back: list[list[int]] = [[] for _ in range(n)]

        dp_val[0] = [sc for (_t, _li, _fi, sc) in cand_by_anchor[0]]
        back[0] = [-1] * len(cand_by_anchor[0])

        for k in range(1, n):
            prev_times = [t for (t, *_r) in cand_by_anchor[k - 1]]
            prev_dp = dp_val[k - 1]
            # prefix-max cua dp[k-1] theo thoi gian tang dan - voi moi ung vien anchor k, chi
            # can nhi phan tim vi tri cuoi cung co prev_time < time hien tai, tra ve prefix-max
            # da tinh san (O(1)/truy van sau khi build 1 lan O(m)).
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
            for (t, _li, _fi, sc) in cand_by_anchor[k]:
                pos = bisect.bisect_left(prev_times, t) - 1  # ung vien cuoi cung co time < t
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
            continue  # khong co chuoi hop le nao (1 anchor nao do khong co ung vien sau moc truoc)

        best_j = int(np.argmax(dp_val[n - 1]))
        best_val = dp_val[n - 1][best_j]
        agg = best_val if aggregate == "min" else best_val / n

        # backtrack lay lai chuoi ung vien da chon
        chosen_idx = [0] * n
        chosen_idx[n - 1] = best_j
        for k in range(n - 1, 0, -1):
            chosen_idx[k - 1] = back[k][chosen_idx[k]]

        chosen: list[_Pick] = []
        for k in range(n):
            t, li, fi, sc = cand_by_anchor[k][chosen_idx[k]]
            chosen.append(_Pick(local_idx=li, frame_idx=fi, pts_time=t, score=sc))
        results.append((video_id, chosen, agg))

    results.sort(key=lambda r: -r[2])
    return results


def _normalize_anchor(a: str | dict) -> dict:
    """Anchor co the la string thuong (khong filter rieng, tuong thich nguoc) hoac dict
    {"text": str, "must_have_labels": list|None, "min_count": dict|None, "ocr_text": str|None,
    "ocr_region": tuple|None} - filter RIENG cho DUNG khoanh khac nay (vd anchor 1 can "Person"
    xuat hien, anchor 2 can chu "HET" tren man hinh) - khac han cac filter global (authors/date)
    o tren, ap dung CHUNG ca chuoi vi luon la CUNG 1 video."""
    if isinstance(a, str):
        return {"text": a, "must_have_labels": None, "min_count": None, "ocr_text": None, "ocr_region": None}
    return {
        "text": a["text"],
        "must_have_labels": a.get("must_have_labels"),
        "min_count": a.get("min_count"),
        "ocr_text": a.get("ocr_text"),
        "ocr_region": a.get("ocr_region"),
    }


def search(
    anchors: list[str | dict],
    top_k: int = 10,
    *,
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
    coarse_k: int = DEFAULT_COARSE_K,
    aggregate: str = "min",
    log: StepLog | None = None,
) -> pd.DataFrame:
    """Điểm vào chính của Tầng 3 — thay thế hoàn toàn Tầng 1+2 dạng 1-query khi có anchors
    (không phải hậu xử lý kết quả Tầng 2 có sẵn, vì mỗi anchor cần pool riêng).

    anchors: mỗi phần tử là string (không filter riêng) HOẶC dict (xem _normalize_anchor) —
    filter object/OCR RIÊNG cho từng mốc, AND thêm vào bộ lọc chung (authors/date/must_have_labels
    ở trên vẫn áp dụng cho MỌI anchor vì luôn cùng 1 video).

    Trả 1 dòng/video, cột: video_id, score, anchor{i}_local_idx/frame_idx/pts_time.
    """
    if len(anchors) < 2:
        raise ValueError("Temporal cần >=2 anchor theo thứ tự thời gian (1 anchor thì dùng Tầng 2 thường)")

    anchors = [_normalize_anchor(a) for a in anchors]

    if log:
        with log.timed("Tầng 1 — lọc thô (dùng chung cho mọi anchor)") as set_detail:
            candidates = tier1_filter.apply(
                authors=authors, date_from=date_from, date_to=date_to,
                keywords_any=keywords_any, must_have_labels=must_have_labels, min_count=min_count,
                use_suppression=use_suppression, include_open_vocab=include_open_vocab,
                ocr_text=ocr_text, ocr_region=ocr_region,
            )
            set_detail("không lọc (toàn corpus)" if candidates is None else f"{len(candidates)} frame ứng viên")
    else:
        candidates = tier1_filter.apply(
            authors=authors, date_from=date_from, date_to=date_to,
            keywords_any=keywords_any, must_have_labels=must_have_labels, min_count=min_count,
            use_suppression=use_suppression, include_open_vocab=include_open_vocab,
            ocr_text=ocr_text, ocr_region=ocr_region,
        )

    anchor_pools = []
    for i, a in enumerate(anchors):
        # loc RIENG cho anchor nay (object/OCR khac nhau moi khoanh khac) - AND them vao
        # candidates CHUNG (khong thay the) - chi tinh khi anchor nay CO khai bao filter rieng,
        # tranh goi tier1_filter thua cho truong hop thuong (anchor la string).
        has_own_filter = a["must_have_labels"] or a["min_count"] or a["ocr_text"] or a["ocr_region"]
        if has_own_filter:
            own = tier1_filter.apply(
                must_have_labels=a["must_have_labels"], min_count=a["min_count"],
                use_suppression=use_suppression, include_open_vocab=include_open_vocab,
                ocr_text=a["ocr_text"], ocr_region=a["ocr_region"],
            )
            anchor_candidates = (own if candidates is None else
                                  (candidates & own if own is not None else candidates))
        else:
            anchor_candidates = candidates

        if log:
            with log.timed(f"Tầng 2 — pool riêng cho anchor {i} ('{a['text']}')") as set_detail:
                pool = _run_anchor_pool(a["text"], anchor_candidates, coarse_k)
                extra = " (có filter riêng)" if has_own_filter else ""
                set_detail(f"{len(pool)} ứng viên (coarse_k={coarse_k}){extra}")
        else:
            pool = _run_anchor_pool(a["text"], anchor_candidates, coarse_k)
        anchor_pools.append(pool)

    if log:
        with log.timed("Tầng 3 — anchor-chain join") as set_detail:
            joined = _temporal_join(anchor_pools, aggregate=aggregate)[:top_k]
            set_detail(f"{len(joined)} video khớp đủ cả {len(anchors)} anchor đúng thứ tự")
    else:
        joined = _temporal_join(anchor_pools, aggregate=aggregate)[:top_k]

    rows = []
    for video_id, picks, score in joined:
        row: dict = {"video_id": video_id, "score": score}
        for i, p in enumerate(picks):
            row[f"anchor{i}_local_idx"] = p.local_idx
            row[f"anchor{i}_frame_idx"] = p.frame_idx
            row[f"anchor{i}_pts_time"] = p.pts_time
        rows.append(row)

    return pd.DataFrame(rows)


def apply(candidates: pd.DataFrame, anchors: list[str] | None = None) -> pd.DataFrame:
    """Giữ lại điểm vào cũ cho tương thích orchestrator: nếu không có anchors, pass-through.
    Nếu có anchors, search.py nên gọi search() ở trên trực tiếp thay vì đi qua đây — xem
    search.py::search() (nhánh temporal bỏ qua Tầng 1/2 dạng 1-query hoàn toàn)."""
    if not anchors:
        return candidates
    raise RuntimeError(
        "Có anchors: gọi tiers.tier3_temporal.search(anchors, ...) trực tiếp từ search.py, "
        "không đi qua apply() — temporal cần pool riêng cho từng anchor, không hậu xử lý được."
    )


# Ghi chú rủi ro mật độ keyframe (chưa xử lý):
# Khi triển khai thật cho TRAKE, nên đo tỷ lệ % khoảnh khắc gán nhãn tay có 1 keyframe nằm
# trong cửa sổ <10 frame — nếu thấp, cần thêm bước decode frame trực tiếp từ video gốc quanh
# pts_time đã chọn ở đây (dùng làm điểm neo thô), không chỉ chọn trong tập keyframe có sẵn.
