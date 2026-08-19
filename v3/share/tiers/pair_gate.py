"""4 lớp gate chặn bùng nổ cặp — TRƯỚC khi gọi VLM (P1, chưa triển khai ở bản này). Thiết kế
đã bàn kỹ từ trước (xem artifact pair-gating.html): L0 type-filter -> L1 boundary-gap ->
L2 agency -> L3 budget cap. Toàn bộ thiên về RECALL (thà giữ dư — cặp bị loại ở đây mất
vĩnh viễn, không có cơ hội sửa ở pha sau).

Input: danh sách detection của 1 FRAME (video_id, local_idx cố định). Output: danh sách cặp
(subj, obj) đã qua gate, mỗi cặp kèm các relation còn khả dĩ (allowed_relations) — sẵn sàng
đưa vào P1 (VLM extract, CHƯA làm) để hỏi "cặp này có đúng là 1 trong các relation này không".
"""
from __future__ import annotations

from dataclasses import dataclass

from relation_registry import _effective_categories, find_relations_for

T_BOUNDARY_GAP = 0.5  # L1: normalized_gap <= T thì giữ (lỏng, ưu tiên recall)
AGENT_CATEGORIES = {"person", "animal"}  # L2
BUDGET_K = 25  # L3

Box = tuple[float, float, float, float]  # (ymin, xmin, ymax, xmax), normalize 0-1


@dataclass
class PairCandidate:
    subj_id: int
    subj_label: str
    obj_id: int
    obj_label: str
    allowed_relations: list[str]
    normalized_gap: float


def _box_dims(b: Box) -> tuple[float, float]:
    y0, x0, y1, x1 = b
    return max(0.0, y1 - y0), max(0.0, x1 - x0)  # (height, width)


def _edge_gap(a: Box, b: Box) -> float:
    """Khoảng cách nhỏ nhất giữa 2 biên box — 0 nếu chồng lấn/chạm nhau (khớp "gap âm nếu
    chồng lấn" trong thiết kế gốc, đơn giản hoá về 0 vì chỉ cần biết "đủ gần" hay không)."""
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    dx = max(bx0 - ax1, ax0 - bx1, 0.0)
    dy = max(by0 - ay1, ay0 - by1, 0.0)
    if dx == 0.0 and dy == 0.0:
        return 0.0
    return (dx**2 + dy**2) ** 0.5


def normalized_gap(a: Box, b: Box) -> float:
    """gap / min(cạnh ngắn hơn của 2 box) — đúng công thức đã thiết kế cho L1."""
    gap = _edge_gap(a, b)
    ah, aw = _box_dims(a)
    bh, bw = _box_dims(b)
    short_a = min(ah, aw) or 1e-6
    short_b = min(bh, bw) or 1e-6
    return gap / min(short_a, short_b)


def gate_pairs(detections: list[dict]) -> list[PairCandidate]:
    """detections: list[dict] của ĐÚNG 1 frame, mỗi dict có detection_id, label, ymin, xmin,
    ymax, xmax. Trả danh sách cặp đã qua L0-L3, budget K theo SỐ CẶP (không phải số relation —
    1 cặp có thể khớp nhiều relation, P1 sẽ hỏi VLM chọn trong allowed_relations của cặp đó)."""
    n = len(detections)
    pair_candidates: dict[tuple[int, int], PairCandidate] = {}

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            subj, obj = detections[i], detections[j]

            rels = find_relations_for(subj["label"], obj["label"])  # L0
            if not rels:
                continue

            box_s: Box = (subj["ymin"], subj["xmin"], subj["ymax"], subj["xmax"])
            box_o: Box = (obj["ymin"], obj["xmin"], obj["ymax"], obj["xmax"])
            gap = normalized_gap(box_s, box_o)
            subj_cats = _effective_categories(subj["label"])

            allowed: list[str] = []
            for r in rels:
                if r.gate_type != "orientation" and gap > T_BOUNDARY_GAP:  # L1 (ngoại lệ orientation)
                    continue
                if r.category == "interaction" and not (subj_cats & AGENT_CATEGORIES):  # L2
                    # Lưu ý: với REGISTRY hiện tại, mọi relation "interaction" đã có
                    # subject_categories giới hạn person/animal ngay từ lúc định nghĩa, nên L0
                    # thực ra đã tự động chặn trường hợp reversed-pair rồi — L2 ở đây hiện là
                    # lưới an toàn cho relation TƯƠNG LAI có subject_categories rộng hơn, không
                    # phải bước lọc thêm gì mới với Registry hiện có. Giữ lại để đúng thiết kế
                    # 4 lớp, không phải code thừa.
                    continue
                allowed.append(r.name)

            if not allowed:
                continue

            key = (subj["detection_id"], obj["detection_id"])
            pair_candidates[key] = PairCandidate(
                subj_id=subj["detection_id"],
                subj_label=subj["label"],
                obj_id=obj["detection_id"],
                obj_label=obj["label"],
                allowed_relations=allowed,
                normalized_gap=gap,
            )

    result = list(pair_candidates.values())
    if len(result) > BUDGET_K:  # L3
        result.sort(key=lambda c: c.normalized_gap)
        result = result[:BUDGET_K]
    return result


if __name__ == "__main__":
    import sys

    import pandas as pd

    from config import OBJECTS_INDEX_PATH

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    df = pd.read_parquet(OBJECTS_INDEX_PATH).reset_index(drop=False).rename(columns={"index": "detection_id"})
    counts = df.groupby(["video_id", "local_idx"]).size()

    # frame dong nguoi nhat (test L3 budget cap co kich hoat khong) + 1 frame trung binh
    busiest = counts.idxmax()
    typical = counts[counts == int(counts.median())].index[0]

    for label, (video_id, local_idx) in [("BUSIEST", busiest), ("TYPICAL", typical)]:
        g = df[(df.video_id == video_id) & (df.local_idx == local_idx)]
        detections = g.to_dict("records")
        cands = gate_pairs(detections)
        print(f"=== {label}: {video_id}/{local_idx} — {len(detections)} object ===")
        print(f"  -> {len(cands)} cap qua gate (BUDGET_K={BUDGET_K})")
        for c in cands[:8]:
            print(f"     {c.subj_label:20s} -> {c.obj_label:20s}  gap={c.normalized_gap:.3f}  rels={c.allowed_relations}")
        print()
