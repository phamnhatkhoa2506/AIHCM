"""Batch driver Phase P1 — chạy `p1_extract.extract_frame` trên nhiều frame song song (Modal
tự autoscale container), ghi `semantic_edges.parquet`, checkpoint/resume được nếu ngắt giữa chừng.

MẶC ĐỊNH LÀ PILOT NHỎ (N_FRAMES=200) — đây là job tốn tiền thật (GPU-giây trên Modal), không
tự ý tăng lên full 156,965 frame khi chưa soát pilot này (đúng lộ trình đã thống nhất: pilot
5,000-10,000 trước, đo throughput+precision, rồi mới quyết full run).

Chạy: python build_semantic_edges.py [n_frames]
"""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from config import INDEX_DIR, OBJECTS_INDEX_PATH
from p1_extract import SpatialIndex, extract_frame
from tiers.pair_gate import gate_pairs

N_FRAMES_DEFAULT = 200  # PILOT — tăng dần sau khi soát, không nhảy thẳng lên hàng nghìn
N_WORKERS = 8  # request đồng thời — Modal tự scale container, không cần backoff kiểu rate-limit

RESULTS_PATH = INDEX_DIR / "semantic_edges.parquet"
PROGRESS_PATH = INDEX_DIR / "semantic_edges_progress.parquet"
# extra_observations: KHONG qua gate, khac ban chat voi RESULTS_PATH (relation da tin dung) —
# chi de soat/lam giau Registry sau nay, khong dung truc tiep cho truy van.
EXTRA_OBS_PATH = INDEX_DIR / "extra_observations.parquet"

# Relation hiếm (xem analyze_registry_coverage.py: feeding=3952, riding=24620 candidate trên
# toàn corpus) — ưu tiên đưa vào pilot để có mẫu hiệu chỉnh, tránh random sample bỏ sót hoàn toàn.
PRIORITY_RELATIONS = {"feeding", "riding"}


def _select_pilot_frames(n_frames: int, already_done: set[tuple[str, int]]) -> list[tuple[str, int]]:
    df = pd.read_parquet(OBJECTS_INDEX_PATH).reset_index(drop=False).rename(columns={"index": "detection_id"})
    groups = df.groupby(["video_id", "local_idx"], sort=False)

    priority: list[tuple[str, int]] = []
    others: list[tuple[str, int]] = []

    for (video_id, local_idx), g in groups:
        key = (video_id, local_idx)
        if key in already_done or len(g) < 2:
            continue
        cands = gate_pairs(g.to_dict("records"))
        if not cands:
            continue
        if any(r in PRIORITY_RELATIONS for c in cands for r in c.allowed_relations):
            priority.append(key)
        else:
            others.append(key)
        if len(priority) + len(others) >= n_frames * 20:  # quét đủ mẫu để chọn, khỏi quét hết corpus
            break

    import random

    random.seed(0)
    random.shuffle(others)

    selected = priority[: n_frames // 2] + others[: n_frames - len(priority[: n_frames // 2])]
    return selected[:n_frames]


def _load_progress() -> set[tuple[str, int]]:
    if not PROGRESS_PATH.exists():
        return set()
    df = pd.read_parquet(PROGRESS_PATH)
    return set(zip(df["video_id"], df["local_idx"]))


def main(n_frames: int) -> None:
    already_done = _load_progress()
    print(f"Da xong tu truoc: {len(already_done)} frame", file=sys.stderr)

    frames = _select_pilot_frames(n_frames, already_done)
    print(f"Pilot lan nay: {len(frames)} frame moi (uu tien relation hiem: {PRIORITY_RELATIONS})",
          file=sys.stderr)
    if not frames:
        print("Khong con frame nao moi de chay.", file=sys.stderr)
        return

    spatial_index = SpatialIndex.load()

    all_relations: list[dict] = []
    all_extra: list[dict] = []
    progress_rows: list[dict] = []
    n_ok, n_err, n_empty = 0, 0, 0

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(extract_frame, vid, li, spatial_index): (vid, li) for vid, li in frames}
        for i, fut in enumerate(as_completed(futures), 1):
            video_id, local_idx = futures[fut]
            try:
                relations, extra = fut.result()
                all_relations.extend(relations)
                for obs in extra:
                    all_extra.append({"video_id": video_id, "local_idx": local_idx, "observation": obs})
                if relations:
                    n_ok += 1
                else:
                    n_empty += 1
                progress_rows.append(
                    {"video_id": video_id, "local_idx": local_idx, "status": "ok", "n_relations": len(relations)}
                )
            except Exception as e:
                n_err += 1
                print(f"loi {video_id}/{local_idx}: {e}", file=sys.stderr)
                progress_rows.append(
                    {"video_id": video_id, "local_idx": local_idx, "status": "error", "n_relations": 0}
                )

            if i % 20 == 0 or i == len(frames):
                print(f"[{i}/{len(frames)}] ok={n_ok} rong={n_empty} loi={n_err}", file=sys.stderr)

    # ghi ket qua (append vao file cu neu co, giu checkpoint dung nghia)
    new_results = pd.DataFrame(all_relations)
    if RESULTS_PATH.exists():
        old_results = pd.read_parquet(RESULTS_PATH)
        new_results = pd.concat([old_results, new_results], ignore_index=True)
    new_results.to_parquet(RESULTS_PATH, index=False)

    if all_extra:
        new_extra = pd.DataFrame(all_extra)
        if EXTRA_OBS_PATH.exists():
            old_extra = pd.read_parquet(EXTRA_OBS_PATH)
            new_extra = pd.concat([old_extra, new_extra], ignore_index=True)
        new_extra.to_parquet(EXTRA_OBS_PATH, index=False)
        print(f"extra_observations: {len(new_extra)} dong -> {EXTRA_OBS_PATH} (soat tay sau)")

    new_progress = pd.DataFrame(progress_rows)
    if PROGRESS_PATH.exists():
        old_progress = pd.read_parquet(PROGRESS_PATH)
        new_progress = pd.concat([old_progress, new_progress], ignore_index=True)
    new_progress.to_parquet(PROGRESS_PATH, index=False)

    print(f"\nXong pilot: {len(frames)} frame ({n_ok} co relation, {n_empty} rong, {n_err} loi)")
    print(f"Tong relation da luu: {len(new_results)} -> {RESULTS_PATH}")
    print(f"Chay lai lenh nay se tu dong bo qua {len(already_done) + len(frames)} frame da xong.")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    n = int(sys.argv[1]) if len(sys.argv) > 1 else N_FRAMES_DEFAULT
    main(n)
