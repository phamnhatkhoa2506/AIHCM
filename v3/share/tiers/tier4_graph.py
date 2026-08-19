"""TẦNG 4 — Graph: xác minh cấu trúc/quan hệ object-object (holding/riding/inside...) trên
shortlist đã qua Tầng 1-3, theo thiết kế Relation Registry + gate 4 lớp (L0 type-filter ->
L1 boundary-gap -> L2 agency -> L3 budget cap) đã bàn trước đó.

Tiến độ (2026-08-05):
  1. ĐÃ XONG — `relation_registry.py`: Registry map vào 514 nhãn thật qua category đóng
     (`index/label_types.json`, phân loại bằng LLM + patch tay lỗi phát hiện được, vd "Hot dog"
     bị gắn nhầm animal vì có chữ "dog"). Đã fix 1 bug quan trọng: category "object" bị model
     gắn gần như universal (kể cả Person/Dog) — phải loại trừ person/animal khỏi "object" thì
     Person+Person mới đúng ra `[looking_at, talking_to]` như thiết kế lý thuyết ban đầu.
  2. ĐÃ XONG — `build_spatial_edges.py` -> `index/spatial_edges.parquet`: 4,454,643 cạnh
     (left_of/above/inside), thuần hình học từ box, không cần VLM.
  3. ĐÃ XONG — `pair_gate.py` (cùng thư mục): L0 (relation_registry.find_relations_for) ->
     L1 boundary-gap (ngoại lệ orientation, verify: 2 Person cách xa gap=5.0 vẫn qua vì
     looking_at/talking_to là orientation) -> L2 agency (hiện là lưới an toàn — REGISTRY hiện
     tại mọi relation interaction đã giới hạn subject person/animal từ L0 rồi, xem comment
     trong pair_gate.py) -> L3 budget cap K=25 (verify: frame 73 object -> đúng còn 25 cặp).
  4. CHƯA LÀM — P1 (VLM extract) đọc allowed_relations của mỗi cặp đã qua gate, hỏi model chọn
     đúng relation (hoặc none) — theo quyết định trước đó, đây là tầng LAZY, chỉ tính trên
     shortlist nhỏ tại query-time, KHÔNG precompute offline cho cả corpus (chi phí quá cao ở
     quy mô ~177k frame, xem thảo luận Colab timing).

CHƯA TRIỂN KHAI phần verify/query thật (apply() dưới đây) — hiện tại pass-through khi không có
relations, raise rõ ràng nếu có.
"""
from __future__ import annotations

import pandas as pd


def apply(candidates: pd.DataFrame, relations: list[dict] | None = None) -> pd.DataFrame:
    """relations: ràng buộc quan hệ cần verify, vd [{"subj": "...", "rel": "riding",
    "obj": "..."}]. None/rỗng -> bỏ qua tầng này, trả nguyên candidates."""
    if not relations:
        return candidates
    raise NotImplementedError("Tầng 4 (graph) chưa triển khai — xem docstring module này")
