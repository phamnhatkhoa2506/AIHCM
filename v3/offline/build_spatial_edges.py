"""Cạnh SPATIAL (left_of/above/inside) — thuần hình học từ box, KHÔNG cần VLM, precompute
được cho cả corpus (khác cạnh ngữ nghĩa holding/riding — cái đó để lazy-online, xem
tiers/tier4_graph.py và lý do chi phí đã bàn).

Với mỗi frame có >=2 object (objects_index.parquet, đã lọc score>=0.3), xét từng cặp:
  1. Containment trước ("inside"): box A phần lớn nằm trong box B và A nhỏ hơn B rõ rệt.
  2. Nếu không, xét hướng tương đối (left_of/above) qua tâm box — CHỈ gán khi 2 box gần như
     không chồng lấn (IoU thấp) và độ lệch đủ lớn — chồng lấn nhiều hoặc lệch quá nhỏ thì
     KHÔNG gán gì (thà bỏ sót còn hơn gán quan hệ mơ hồ).

Không có ngưỡng nào ở đây được đo từ data thật — đây là điểm khởi đầu hợp lý (tương tự tinh
thần threshold ở build_object_stats.py), cần soát lại bằng mắt trên vài chục cặp khi có thời gian.
"""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/

import sys

import pandas as pd

from config import INDEX_DIR, OBJECTS_INDEX_PATH

SPATIAL_EDGES_PATH = INDEX_DIR / "spatial_edges.parquet"

CONTAIN_RATIO_THRESH = 0.75  # >=75% diện tích A nằm trong B -> A "inside" B
SIZE_RATIO_MAX = 0.85  # A phải nhỏ hơn B rõ rệt (area_A <= 0.85 * area_B) mới tính containment
IOU_DIRECTIONAL_MAX = 0.2  # chồng lấn nhiều hơn mức này -> không gán left_of/above (mơ hồ)
MIN_OFFSET = 0.03  # lệch tâm quá nhỏ (toạ độ đã normalize 0-1) -> không gán, tránh nhiễu

Box = tuple[float, float, float, float]  # (ymin, xmin, ymax, xmax), normalize 0-1


def _area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersection(a: Box, b: Box) -> float:
    iy1, ix1 = max(a[0], b[0]), max(a[1], b[1])
    iy2, ix2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, iy2 - iy1) * max(0.0, ix2 - ix1)


def classify_pair(a: Box, b: Box) -> tuple[str, bool] | None:
    """Trả (relation, subj_is_a) — subj_is_a=True nghĩa là quan hệ đọc theo chiều A->B.
    None nếu không đủ rõ ràng để gán (chủ động bỏ sót, không đoán)."""
    area_a, area_b = _area(a), _area(b)
    if area_a <= 0 or area_b <= 0:
        return None

    inter = _intersection(a, b)
    ratio_a_in_b = inter / area_a
    ratio_b_in_a = inter / area_b

    if ratio_a_in_b >= CONTAIN_RATIO_THRESH and area_a <= SIZE_RATIO_MAX * area_b:
        return "inside", True
    if ratio_b_in_a >= CONTAIN_RATIO_THRESH and area_b <= SIZE_RATIO_MAX * area_a:
        return "inside", False

    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    if iou > IOU_DIRECTIONAL_MAX:
        return None

    ca_y, ca_x = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    cb_y, cb_x = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    dy, dx = ca_y - cb_y, ca_x - cb_x

    if abs(dx) >= abs(dy):
        if abs(dx) < MIN_OFFSET:
            return None
        return "left_of", dx < 0  # A lệch trái hơn B (x nhỏ hơn) -> A left_of B
    else:
        if abs(dy) < MIN_OFFSET:
            return None
        return "above", dy < 0  # A lệch trên B (y nhỏ hơn, quy ước ymin=0 ở trên) -> A above B


def build() -> None:
    df = pd.read_parquet(OBJECTS_INDEX_PATH)
    df = df.reset_index(drop=False).rename(columns={"index": "detection_id"})

    n_total_frames = df[["video_id", "local_idx"]].drop_duplicates().shape[0]
    rows: list[dict] = []
    n_frames_scanned = 0
    n_frames_multi = 0

    for (video_id, local_idx), g in df.groupby(["video_id", "local_idx"], sort=False):
        n_frames_scanned += 1
        if len(g) < 2:
            continue
        n_frames_multi += 1

        recs = g.to_dict("records")
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                a, b = recs[i], recs[j]
                box_a: Box = (a["ymin"], a["xmin"], a["ymax"], a["xmax"])
                box_b: Box = (b["ymin"], b["xmin"], b["ymax"], b["xmax"])
                res = classify_pair(box_a, box_b)
                if res is None:
                    continue
                relation, subj_is_a = res
                subj, obj = (a, b) if subj_is_a else (b, a)
                rows.append(
                    {
                        "video_id": video_id,
                        "local_idx": local_idx,
                        "subj_detection_id": subj["detection_id"],
                        "subj_label": subj["label"],
                        "obj_detection_id": obj["detection_id"],
                        "obj_label": obj["label"],
                        "relation": relation,
                    }
                )

        if n_frames_scanned % 20000 == 0:
            print(f"[{n_frames_scanned}/{n_total_frames} frame quet, {len(rows)} canh]", file=sys.stderr)

    out = pd.DataFrame(rows)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(SPATIAL_EDGES_PATH, index=False)

    print(f"Xong: {len(out)} canh spatial, {n_frames_multi} frame co >=2 object "
          f"(tren {n_frames_scanned} frame quet) -> {SPATIAL_EDGES_PATH}")
    if len(out):
        print(out["relation"].value_counts().to_string())


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    build()
