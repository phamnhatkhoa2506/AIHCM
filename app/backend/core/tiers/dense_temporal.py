"""TANG 3 (TRAKE/temporal) - MIGRATE sang BO DENSE (2026-08-15, theo yeu cau nguoi dung "tan
dung khung keo tha tu KIS/Q&A cho Temporal"). Truoc day gac lai (xem tier3_temporal.py cu, van
giu nguyen KHONG doi cho BTC) vi tuong dense_meta.parquet thieu pts_time/fps - THUC RA khong
con la rao can: pts_time = frame_id / fps, fps da co san qua dense_search._fps_by_video()
(dung chung voi ASR boost, khong can rebuild gi ca).

Thuat toan ANCHOR-CHAIN qua QUY HOACH DONG (DP) - GIU NGUYEN nguyen tac voi tier3_temporal.py
cu (Tang khac Tang 2 o cho KHONG dung 1 query duy nhat - moi anchor chay search_dense() RIENG
de co 1 pool ung vien, sau do join theo video + thoi gian, khong phai hau xu ly ket qua 1-query
co san), CHI khac nguon du lieu/ham goi (search_dense thay vi filter+tier2_vector).

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
import contextlib
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from steplog import StepLog
from tiers.dense_search import _fps_by_video, search_dense

DEFAULT_COARSE_K = 1000

# 2026-08-20 (theo yeu cau nguoi dung, sau khi toi uu toc do Modal light-payload van con ~51s
# cho 3 moc "vi nuong/xot cham/dia trang tri" - "Ok giảm luôn bạn"): he so mo rong pool Level-2
# (backfill/nang cap, xem "widen_k" duoi day) HA tu x20 xuong x8 - it ung vien hon -> nhanh hon
# ro ret (encode+rank scale gan tuyen tinh theo so ung vien qua Modal, xem dense_search.py::
# _rank_single_remote), DANH DOI: giam kha nang tim ra video dung khi cac moc RAT khac biet ve
# ngu nghia (nhu case "cắt nấm/củ năng" truoc do, GT o rank ~1783 trong top-1000/anchor - x8 =
# 8000 van du bat duoc case do, chi cac case can toi tan rank ~15000-20000 moi bi anh huong).
WIDEN_POOL_FACTOR = 8


def _run_anchor_pool(
    anchor_text: str,
    dense_model: str,
    coarse_k: int,
    *,
    must_have_labels: list[str] | None,
    min_count: dict[str, int] | None,
    ocr_text: str | None,
    asr_text: str | None,
    video_ids: list[str] | None,
    spatial_boxes: list[dict] | None,
    spatial_op: str,
    ocr_algorithm: str,
    score_algorithm: str,
    distill_model: str | None,
    log: StepLog | None,
) -> dict[tuple[str, int], tuple[float, str]]:
    """1 anchor -> {(video_id, frame_id): (score, path)}, coarse_k ung vien/anchor."""
    ranked = search_dense(
        anchor_text, dense_model, top_k=coarse_k,
        must_have_labels=must_have_labels, min_count=min_count, ocr_text=ocr_text,
        asr_text=asr_text, video_ids=video_ids, spatial_boxes=spatial_boxes, spatial_op=spatial_op, ocr_algorithm=ocr_algorithm,
        score_algorithm=score_algorithm, distill_model=distill_model, log=log,
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


# 2026-08-18 (theo yeu cau nguoi dung, phat hien qua case that "Một đoàn đông cua-rơ...Sau đó
# là cảnh cận hơn..." - GT range_frame_id chi CACH NHAU 20 frame (~0.8s o fps thuong gap) nhung
# he thong chon 2 moc CACH NHAU 65 GIAY): DP goc (bisect + prefix-max) CHI doi hoi mốc sau DEN
# SAU mốc truoc ve thoi gian, KHONG co rang buoc/phat gi ve KHOANG CACH - san sang chon ung vien
# cach xa hang chuc giay mien diem CLIP cao hon. Theo tai lieu goc BTC (xem plan file dau du an):
# "cửa sổ chấm điểm TRAKE thường dưới 10 frame" tren truc video goc - cuc ky hep (< 1 giay o fps
# thuong gap) - 2 moc GT gan nhu LUON sat nhau, khong phai cach vai chuc giay.
#
# MAX_ANCHOR_GAP_SECONDS la NGUONG TUNABLE (khong phai gia tri "dung tuyet doi" - do lay mau
# dense hien tai (~0.55-2.65s/frame) THO HON nhieu so voi cua so cham diem that (<10 frame goc),
# nen ep qua chat (vd 1-2s) co the khong con ung vien nao kha thi trong nhieu video. Chon muc
# trung dung (15s) - du hep de loai cac cap "cach xa hang chuc giay" ro rang sai (nhu case
# 65 giay o tren), nhung van du rong cho pool ung vien thua (do mat do dense) khong bi trang.
#
# 2026-08-20 (theo yeu cau nguoi dung, phat hien qua case that GT L26_V194 [4700,5125,5450,5850],
# TRAKE 4-moc "bột măng tây... rời chảo dầu"): 15s VAN QUA CHAT - 2/3 khoang cach GT that la
# 17.0s va 16.0s, VUOT nguong -> DAY LA RANG BUOC CUNG (_temporal_join dung sliding-window LOAI
# HAN ung vien ngoai [t-max_gap, t), khong phai phat diem) nen thuat toan KHONG BAO GIO xet duoc
# cap moc dung, buoc phai chon 1 cum frame SAI nhung gan nhau de thoa man rang buoc. Tang len
# 25s - du rong cho case 17s nay + margin, van loai duoc case 65s ro rang sai da phat hien truoc.
MAX_ANCHOR_GAP_SECONDS = 25.0


def _temporal_join(
    anchor_pools: list[dict[tuple[str, int], tuple[float, str]]],
    fps_map: dict[str, float],
    aggregate: str = "mean",
    max_gap_seconds: float = MAX_ANCHOR_GAP_SECONDS,
) -> list[tuple[str, list[_Pick], float]]:
    """Anchor-chain qua QUY HOACH DONG (DP) - COPY NGUYEN nguyen tac tu tier3_temporal.py::
    _temporal_join (ban BTC), CHI khac cho lay pts_time = frame_id / fps thay vi doc san tu
    meta.parquet (dense khong co cot pts_time), va key la frame_id (KHONG phai local_idx - bo
    dense khong co khai niem local_idx, frame_id CHINH LA frame_idx tren truc video goc, dung
    THANG lam frame_id nop bai, xem dense_search.py docstring).

    max_gap_seconds (2026-08-18): chi cho phep mốc SAU chon ung vien co pts_time nam trong
    [t - max_gap_seconds, t) so voi mốc TRUOC - xem MAX_ANCHOR_GAP_SECONDS. Dung SLIDING WINDOW
    MAXIMUM (deque don dieu) thay vi prefix-max toan cuc nhu ban goc - van giu O(n) amortized
    (khong phai O(n^2)) vi ca 2 dau cua so CHI DI TOI (khong lui) khi t (mốc sau) tang dan, dung
    tinh chat cac ung vien da duoc sap theo pts_time tang dan tu truoc."""
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

            cur_dp: list[float] = []
            cur_back: list[int] = []
            # 2026-08-18: SLIDING WINDOW MAXIMUM (deque don dieu, chi so vao prev_dp GIAM DAN
            # gia tri) thay cho prefix-max toan cuc cua ban goc - dieu kien "mốc sau chon ung
            # vien trong [t-max_gap, t)" thay vi "bat ky luc nao truoc t" (xem MAX_ANCHOR_GAP_
            # SECONDS). `added_ptr` danh dau da dua vao deque nhung ung vien prev co pts_time <
            # t hay chua - CHI TANG (khong lui) vi t (mốc sau) duyet theo thu tu TANG DAN.
            # 2026-08-21: dung collections.deque thay cho list. Doan ma nay tu nhan la "O(n)
            # amortized" nhung voi list THUAN thi dq.pop(0) (bo dau cua so ben trai) la O(n) -
            # phai dich CA mang moi lan - nen thuc te la O(n^2) khi NHIEU ung vien cua CUNG 1
            # video (vd loc video_ids ve 1 video + pool mo rong 8000/moc). deque.popleft() moi
            # dung la O(1), tra lai dung do phuc tap ma ghi chu goc mo ta.
            dq: deque[int] = deque()  # chi so vao prev_times/prev_dp, prev_dp[dq[i]] GIAM DAN
            added_ptr = 0
            for (t, _fi, sc, _p) in cand_by_anchor[k]:
                right_ptr = bisect.bisect_left(prev_times, t)
                while added_ptr < right_ptr:
                    j = added_ptr
                    while dq and prev_dp[dq[-1]] <= prev_dp[j]:
                        dq.pop()
                    dq.append(j)
                    added_ptr += 1
                min_allowed_time = t - max_gap_seconds
                while dq and prev_times[dq[0]] < min_allowed_time:
                    dq.popleft()
                if not dq:
                    cur_dp.append(-np.inf)
                    cur_back.append(-1)
                    continue
                pred_idx = dq[0]
                pred_val = prev_dp[pred_idx]
                if pred_val == -np.inf:
                    cur_dp.append(-np.inf)
                    cur_back.append(-1)
                    continue
                # 2026-08-18 (theo yeu cau nguoi dung, phat hien qua GT that L26_V072
                # [2475,3125,3425,3775] - moc "lua xuat hien" diem CLIP thap bat thuong ~0.156
                # do la hanh dong chung chung, kho phan biet): default DA DOI tu "min" sang
                # "mean" - "min" khien 1 moc yeu keo TRAN diem ca chuoi xuong bang no, lam video
                # DUNG (co 3/4 moc diem cao) xep hang NGANG HANG voi cac chuoi khac deu yeu ca
                # 4 moc, tut hang oan (video that xep #24 thay vi top dau du frame gan dung GT
                # van co trong pool). "mean" (nhanh else, cong don roi chia n o duoi) cho 3 moc
                # manh vAN duoc TINH DIEM, khong bi 1 moc yeu xoa sach loi the.
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
    anchor (2026-08-15, canvas rieng/moc) + asr_text rieng (2026-08-20, xem docstring search())
    - anchor la string (khong filter rieng) hoac dict
    {"text", "must_have_labels", "min_count", "ocr_text", "asr_text", "spatial_boxes", "spatial_op"}.

    2026-08-21 (theo yeu cau nguoi dung, chi rieng ban app/: "quan hệ riêng cho từng box") -
    THEM "spatial_op" rieng cho tung anchor (None = dung chung spatial_op TOAN CUC, xem
    search()) - v3 KHONG co truong nay (chi 1 spatial_op DUY NHAT dung chung moi anchor)."""
    if isinstance(a, str):
        return {
            "text": a, "must_have_labels": None, "min_count": None,
            "ocr_text": None, "asr_text": None, "spatial_boxes": None, "spatial_op": None,
        }
    return {
        "text": a["text"],
        "must_have_labels": a.get("must_have_labels"),
        "min_count": a.get("min_count"),
        "ocr_text": a.get("ocr_text"),
        "asr_text": a.get("asr_text"),
        "spatial_boxes": a.get("spatial_boxes"),
        "spatial_op": a.get("spatial_op"),
    }


def search(
    anchors: list[str | dict],
    top_k: int = 10,
    *,
    dense_model: str = "rrf",
    must_have_labels: list[str] | None = None,
    min_count: dict[str, int] | None = None,
    ocr_text: str | None = None,
    asr_text: str | None = None,
    video_ids: list[str] | None = None,
    spatial_boxes: list[dict] | None = None,
    spatial_op: str = "and",
    ocr_algorithm: str = "flexible",
    score_algorithm: str = "cosine",
    distill_model: str | None = None,
    coarse_k: int = DEFAULT_COARSE_K,
    aggregate: str = "mean",
    max_gap_seconds: float = MAX_ANCHOR_GAP_SECONDS,
    log: StepLog | None = None,
    **_ignored,  # phong than: nhan them cac tham so BTC cu (authors/date_from/...) ma khong loi
    # neu app.py con truyen qua **common_filters - CHUA ho tro loc metadata rieng cho Temporal
    # (co the them sau, khong phai trong tam lan sua nay).
) -> pd.DataFrame:
    """Diem vao chinh Tang 3 tren BO DENSE - thay the hoan toan tier3_temporal.search() (BTC)
    khi dung bo dense. must_have_labels/min_count/ocr_text/asr_text/video_ids/spatial_boxes:
    filter CHUNG cho MOI anchor (khung ve tay tu canvas AP DUNG DONG LOAT cho tat ca moc, xem
    docstring dau file) - anchor.must_have_labels/min_count/ocr_text/asr_text (dict) la filter
    RIENG cho DUNG moc do, AND them vao filter chung. asr_text (2026-08-20): xem docstring day
    du o dense_search.py::search_dense - loc theo loi noi (ASR), khac audio_mentions (soft, tu
    LLM). video_ids (2026-08-20, theo yeu cau nguoi dung "thêm bộ lọc search theo video") - ap
    dung CHUNG cho CA CHUOI (khong co per-anchor rieng, video_id la thuoc tinh cua ca video,
    khong phai tung moc).

    Tra 1 dong/video, cot: video_id, score, anchor{i}_frame_id/pts_time/path."""
    if len(anchors) < 2:
        raise ValueError("Temporal cần >=2 anchor theo thứ tự thời gian (1 anchor thì dùng Tầng 2 thường)")

    anchors = [_normalize_anchor(a) for a in anchors]
    fps_map = _fps_by_video()

    def _build_pools(k: int, *, step_label: str) -> list[dict[tuple[str, int], tuple[float, str]]]:
        """Chay _run_anchor_pool cho TUNG anchor voi pool size k - tach thanh ham rieng (2026-08-18)
        de goi lai duoc voi k LON HON cho Level-2 backfill duoi day (xem "MO RONG POOL"), khong
        phai copy dan code.

        2026-08-21 (theo yeu cau nguoi dung: "xem lai thuat toan search cho temporal va trake...
        con cach nao toi uu them toc do khong") - TRUOC DAY chay TUAN TU (vong for) nen tong thoi
        gian = TONG thoi gian moi anchor. DO THAT (2 anchor, che do Modal remote): 1.96s + 1.95s
        = 3.91s cho pool goc, 2.49s + 2.31s = 4.83s cho pool mo rong -> 8.75s tong.
        Cac anchor HOAN TOAN DOC LAP nhau (moi anchor 1 lan search_dense rieng, khong anchor nao
        can ket qua cua anchor khac - viec GHEP CHUOI chi xay ra SAU DO o _temporal_join) nen
        chay SONG SONG duoc -> tong ~= MAX thay vi TONG (2 anchor: ~2x nhanh hon, 4 anchor: ~4x).
        Dung ThreadPoolExecutor (KHONG phai multiprocessing) vi _run_anchor_pool hoac CHO MANG
        (Modal .remote() - I/O-bound, nha GIL khi cho) hoac tinh numpy/faiss (cung nha GIL trong
        phan C) - y HET nguyen tac + ly do da ap dung cho dense_search.py::_rank_rrf (chay song
        song 2 model trong 1 lan RRF).

        LUU Y ve `log`: cac anchor gio chay CHONG LAN nhau ve thoi gian, nen KHONG duoc de moi
        anchor la 1 buoc TOP-LEVEL nhu truoc (StepLog.total_seconds() cong cac buoc top-level ->
        se cong don cac khoang thoi gian SONG SONG = dem nhieu lan cung 1 khoang dong ho thuc,
        bao tong LON HON thuc te - dung loi "dem 2 lan" ma co che `nested` sinh ra de tranh, xem
        steplog.py). Gio boc CA GIAI DOAN trong 1 buoc cha DUY NHAT (do dung wall-clock that),
        cac anchor la buoc CON (nested, van hien thi day du chi tiet nhung khong bi cong vao tong).
        """
        # Chuan bi tham so cho tung anchor TRUOC (thuan tuy, khong I/O) - de phan chay song song
        # ben duoi chi con viec goi _run_anchor_pool.
        per_anchor_kwargs = []
        for a in anchors:
            per_anchor_kwargs.append(dict(
                must_have_labels=list({*(must_have_labels or []), *(a["must_have_labels"] or [])}) or None,
                min_count={**(min_count or {}), **(a["min_count"] or {})} or None,
                ocr_text=a["ocr_text"] or ocr_text,
                asr_text=a["asr_text"] or asr_text,
                video_ids=video_ids,
                # 2026-08-15: MERGE khung ve tay GLOBAL (thuong None tu app.py, giu de tuong
                # thich nguoc) voi khung RIENG cua moc nay (canvas rieng/anchor) - khong loai cai nao.
                spatial_boxes=[*(spatial_boxes or []), *(a["spatial_boxes"] or [])] or None,
                # 2026-08-21 (theo yeu cau nguoi dung, chi rieng ban app/): moi anchor co the tu
                # chon AND/OR RIENG (a["spatial_op"]) - None = dung chung spatial_op toan cuc.
                spatial_op=a["spatial_op"] or spatial_op,
                ocr_algorithm=ocr_algorithm, score_algorithm=score_algorithm,
                distill_model=distill_model,
            ))

        def _one(i: int) -> dict[tuple[str, int], tuple[float, str]]:
            a = anchors[i]
            if log:
                with log.timed(f"{step_label} — anchor {i} ('{a['text']}')") as set_detail:
                    pool = _run_anchor_pool(a["text"], dense_model, k, log=log, **per_anchor_kwargs[i])
                    set_detail(f"{len(pool)} ứng viên (pool size={k})")
                return pool
            return _run_anchor_pool(a["text"], dense_model, k, log=None, **per_anchor_kwargs[i])

        if len(anchors) == 1:
            return [_one(0)]

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(len(anchors), 8)) as executor:
            # list(map(...)) giu DUNG THU TU anchor (thu tu la BAT BUOC - _temporal_join ghep
            # chuoi theo dung thu tu thoi gian cua cac moc, dao thu tu la sai ket qua).
            return list(executor.map(_one, range(len(anchors))))

    # Buoc cha DUY NHAT do wall-clock that cua ca giai doan (cac anchor ben trong chay SONG SONG
    # nen la buoc con/nested - xem ghi chu "LUU Y ve log" trong _build_pools).
    with (log.timed(f"Tầng 3 — dựng pool cho {len(anchors)} mốc (chạy song song)") if log
          else contextlib.nullcontext(lambda *_a, **_k: None)) as set_detail:
        anchor_pools = _build_pools(coarse_k, step_label="Tầng 3 — pool riêng cho")
        if log:
            set_detail(f"{coarse_k} ứng viên/mốc")

    if log:
        with log.timed("Tầng 3 — anchor-chain join") as set_detail:
            joined = _temporal_join(anchor_pools, fps_map, aggregate=aggregate, max_gap_seconds=max_gap_seconds)
            set_detail(f"{len(joined)} video khớp đủ cả {len(anchors)} anchor đúng thứ tự "
                       f"(cách nhau tối đa {max_gap_seconds:.0f}s/mốc)")
    else:
        joined = _temporal_join(anchor_pools, fps_map, aggregate=aggregate, max_gap_seconds=max_gap_seconds)

    joined = joined[:top_k]
    backfill_video_ids: set[str] = set()

    # BACKFILL (2026-08-18, theo yeu cau nguoi dung: "GT tac gia do den tan rank 80 ... nhu
    # KIS/QA truoc do, du ket qua khong dung van xep va render du top") - giong het pattern da
    # lam cho search_dense() (xem dense_search.py::search_dense docstring "BACKFILL"): neu chuoi
    # KHOP DUNG rang buoc (max_gap_seconds) khong du top_k, noi long HOAN TOAN gap constraint
    # (khong goi lai Modal - dung LAI anchor_pools da tinh) de lay them ung vien, danh dau
    # is_backfill=True cho UI biet day la goi y KHONG qua rang buoc khoang cach, khong phai
    # chuoi that su khop chat.
    n_strict = len(joined)  # so chuoi KHOP CHAT (khong phai backfill) - dung de biet dong nao
    # o Level-1 la BACKFILL (them o duoi, sau khi co ca Level-2) - xem "n_level1" ben duoi.

    if len(joined) < top_k:
        n_missing = top_k - len(joined)
        with (log.timed("Tầng 3 — anchor-chain — bù thêm (bỏ ràng buộc khoảng cách)") if log else contextlib.nullcontext(lambda *_a, **_k: None)) as set_detail:
            joined_relaxed = _temporal_join(anchor_pools, fps_map, aggregate=aggregate, max_gap_seconds=float("inf"))
            existing_videos = {video_id for video_id, _picks, _score in joined}
            extra = [j for j in joined_relaxed if j[0] not in existing_videos][:n_missing]
            backfill_video_ids |= {video_id for video_id, _picks, _score in extra}
            joined = joined + extra
            if log:
                set_detail(f"chỉ {top_k - n_missing}/{top_k} chuỗi khớp trong {max_gap_seconds:.0f}s/mốc — "
                           f"BỔ SUNG thêm {len(extra)} chuỗi bỏ qua ràng buộc khoảng cách cho đủ {top_k}, "
                           f"đánh dấu ở cột is_backfill")

    # LEVEL-2 BACKFILL/NANG CAP (2026-08-18, phat hien qua case that "cắt nấm/cắt củ năng/cắt
    # đậu hủ/lên bếp" - Level-1 o tren chi noi long RANG BUOC KHOANG CACH, VAN dung anchor_pools
    # GOC (coarse_k=1000/anchor) - neu ung vien DUNG cua 1 moc nam NGOAI top-1000 (vd rank ~1783
    # cho "cắt củ năng"), Level-1 buoc phai chon 1 ung vien TE HON trong pool de co du chuoi,
    # dan toi video DA duoc tim thay (dung nghia) nhung 1-2 moc bi chon SAI frame trong video do -
    # nguoi dung phat hien qua anh thuc te "frame thứ 2 thì không có củ năng". Fix 2 phan:
    #   (a) CHAY Level-2 (mo rong pool) bat cu khi nao CO backfill tu Level-1 (khong chi khi
    #       THIEU so voi top_k) - de co CO HOI nang cap cac dong da co, khong chi bu them dong moi.
    #   (b) VOI CAC VIDEO DA CO trong `joined` (tu Level-0/Level-1), neu ban pool-mo-rong cho
    #       diem CAO HON, THAY THE dong cu bang dong moi (nang cap), khong chi bo qua no.
    n_level1_backfill = len(backfill_video_ids)
    if len(joined) < top_k or n_level1_backfill > 0:
        widen_k = coarse_k * WIDEN_POOL_FACTOR
        with (log.timed(f"Tầng 3 — anchor-chain — bù/nâng cấp (mở rộng pool lên {widen_k}/mốc)") if log else contextlib.nullcontext(lambda *_a, **_k: None)) as set_detail:
            wide_pools = _build_pools(widen_k, step_label="Tầng 3 — pool mở rộng (backfill) cho")
            joined_wide = _temporal_join(wide_pools, fps_map, aggregate=aggregate, max_gap_seconds=float("inf"))
            wide_by_video = {video_id: (picks, score) for video_id, picks, score in joined_wide}

            n_upgraded = 0
            for idx in range(n_strict, len(joined)):  # CHI nang cap dong BACKFILL (Level-1), khong dong khop chat that
                video_id, picks, score = joined[idx]
                wide = wide_by_video.get(video_id)
                if wide is not None and wide[1] > score:
                    joined[idx] = (video_id, wide[0], wide[1])
                    n_upgraded += 1

            n_missing = top_k - len(joined)
            existing_videos = {video_id for video_id, _picks, _score in joined}
            extra2 = [j for j in joined_wide if j[0] not in existing_videos][:max(0, n_missing)]
            backfill_video_ids |= {video_id for video_id, _picks, _score in extra2}
            # sap xep lai PHAN BACKFILL (tu n_strict tro di) theo diem MOI (sau khi nang cap) -
            # phan KHOP CHAT (0..n_strict) GIU NGUYEN vi tri dau (uu tien hon backfill du diem
            # co the thap hon, giong nguyen tac KIS/QA: ket qua that su khop luon dung TRUOC).
            joined = joined[:n_strict] + sorted(joined[n_strict:] + extra2, key=lambda j: -j[2])
            if log:
                set_detail(f"nâng cấp {n_upgraded}/{n_level1_backfill} dòng backfill bằng pool mở rộng "
                           f"(chọn được frame khớp tốt hơn cho 1+ mốc) — bù thêm {len(extra2)} dòng mới")

    # LEVEL-3 BACKFILL (2026-08-21, theo yeu cau nguoi dung: "không đủ 100 kết quả, như thế này
    # khi nộp thì không đủ 100 kết quả bắt buộc" - phat hien qua case that TRAKE 4-moc "lân quay
    # vòng...4 chân chạm đất...chào ban giám khảo...chào 1 con rồng" - CHI ra 42/100 du da qua ca
    # Level-1 (noi rang buoc khoang cach) LAN Level-2 (mo rong pool 8000/moc)): 2 tang backfill o
    # tren VAN bi CHUNG 1 gioi han cung - _temporal_join() CHI tra ve DUNG 1 dong/video CO DU CA
    # N moc cung luc (giao cua N tap ung vien, xem docstring dau ham) - neu SO VIDEO co du ca 4
    # moc (du da noi long het co) < top_k, KHONG CO CACH NAO Level-1/2 vuot qua tran do, vi ban
    # chat van doi hoi giao ca N tap. Day la khac biet CO BAN voi search_dense() (KIS/QA) - backfill
    # o do lay THANG tu TOAN BO corpus (khong co rang buoc "phai co N thu tim thay trong 1 dong")
    # nen LUON du top_k; TRAKE thi KHONG THE, vi 1 dong TRAKE dinh nghia la 1 CHUOI ca N moc.
    #
    # Fix: neu SAU CA 2 tang tren VAN thieu, dem not bang XEP HANG RIENG cua 1 moc DUY NHAT (anchor
    # 0) cho cac VIDEO CHUA CO trong joined - LAP LAI frame cua moc 0 cho MOI vi tri mocc khac
    # (KHONG phai khop that su cac moc con lai, chi de CO DU FORMAT nop bai) - giong y het tinh
    # than "cac he thong khac tra ve DU top_k" da ap dung cho KIS/QA, is_backfill=True DE UI/nguoi
    # dung biet day la doan bu thap tin cay nhat (BTC chi can DU 100 dong, khong phat vi dong sai -
    # metric R@K chi quan tam GT co trong top-K hay khong, khong phat cac dong con lai).
    #
    # 2026-08-21 (bug thật LẦN 2, phát hiện qua chính case "lân...": lần fix đầu tái dùng
    # wide_pools[0]/anchor_pools[0] - CHỈ ra 48/100, tưởng đã đủ nhưng vẫn thiếu) - đo THẬT bằng
    # script debug: pool(1000) chỉ phủ 41 VIDEO DUY NHẤT, pool(8000) chỉ 48 - vì top-K bị "CHIẾM"
    # bởi 1 CỤM video liên quan nhất (mỗi video hàng trăm frame gần giống nhau đều lọt top), các
    # video CÒN LẠI hoàn toàn KHÔNG CÓ frame nào lọt nổi top-8000 dù muốn lấy "video khác" cũng
    # không có gì để lấy. Đo tiếp: pool(20000)=421 video, pool(40000)=867/873 (gần hết corpus) -
    # phải vượt hẳn ngưỡng đó mới "tràn" ra được các video ít liên quan hơn. Fix: Level-3 tự query
    # pool RIÊNG cho anchor 0 với kích thước ĐỦ LỚN (không tái dùng pool 8000 của Level-2 nữa) -
    # đủ để phủ gần hết 873 video trong corpus, đảm bảo LUÔN đủ top_k dòng bất kể top_k bao nhiêu.
    LEVEL3_POOL_SIZE = 45000
    if len(joined) < top_k:
        n_missing = top_k - len(joined)
        with (log.timed(f"Tầng 3 — bù nốt cho đủ {top_k} (xếp hạng riêng mốc 1, pool {LEVEL3_POOL_SIZE}, không khớp đủ chuỗi)") if log
              else contextlib.nullcontext(lambda *_a, **_k: None)) as set_detail:
            a0 = anchors[0]
            pool0 = _run_anchor_pool(
                a0["text"], dense_model, LEVEL3_POOL_SIZE,
                must_have_labels=list({*(must_have_labels or []), *(a0["must_have_labels"] or [])}) or None,
                min_count={**(min_count or {}), **(a0["min_count"] or {})} or None,
                ocr_text=a0["ocr_text"] or ocr_text, asr_text=a0["asr_text"] or asr_text,
                video_ids=video_ids,
                spatial_boxes=[*(spatial_boxes or []), *(a0["spatial_boxes"] or [])] or None,
                spatial_op=a0["spatial_op"] or spatial_op,
                ocr_algorithm=ocr_algorithm, score_algorithm=score_algorithm,
                distill_model=distill_model, log=log,
            )
            existing_videos = {video_id for video_id, _picks, _score in joined}
            # Moi video CHI lay 1 ung vien TOT NHAT (theo diem moc 0) - trung video thi giu ban
            # diem cao hon, dung dict thay vi list de khu trung truoc khi sap xep.
            best_per_video: dict[str, tuple[int, float, str]] = {}
            for (video_id, frame_id), (score, path) in pool0.items():
                if video_id in existing_videos:
                    continue
                cur = best_per_video.get(video_id)
                if cur is None or score > cur[1]:
                    best_per_video[video_id] = (frame_id, score, path)
            ranked0 = sorted(best_per_video.items(), key=lambda kv: -kv[1][1])[:n_missing]
            extra3 = []
            for video_id, (frame_id, score, path) in ranked0:
                fps = fps_map.get(video_id, 25.0)
                pts_time = frame_id / fps
                picks = [_Pick(frame_id=frame_id, pts_time=pts_time, score=score, path=path) for _ in range(len(anchors))]
                extra3.append((video_id, picks, score))
            backfill_video_ids |= {video_id for video_id, _picks, _score in extra3}
            joined = joined + extra3
            if log:
                set_detail(f"chỉ {top_k - n_missing}/{top_k} video có ĐỦ cả {len(anchors)} mốc (kể cả sau khi nới "
                           f"rộng pool) — bù nốt {len(extra3)} dòng theo xếp hạng RIÊNG mốc 1 (lặp lại "
                           f"frame mốc 1 cho các mốc còn lại - KHÔNG khớp thật, chỉ để đủ số dòng nộp bài)")

    rows = []
    for video_id, picks, score in joined:
        row: dict = {"video_id": video_id, "score": score, "is_backfill": video_id in backfill_video_ids}
        for i, p in enumerate(picks):
            row[f"anchor{i}_frame_id"] = p.frame_id
            row[f"anchor{i}_pts_time"] = p.pts_time
            row[f"anchor{i}_path"] = p.path
        rows.append(row)

    return pd.DataFrame(rows)
