"""Relation Registry — bảng tra cho Tầng 4 (graph). Đã map vào 514 nhãn OpenImages V4 THẬT
(qua category ở label_types.json, build_label_types.py) — thay placeholder "person"/"vehicle"
dùng lúc thiết kế lý thuyết ban đầu.

subject_categories/object_categories tham chiếu category ĐÓNG (7 loại, xem build_label_types.py),
không phải nhãn cụ thể — 1 category có thể ứng nhiều nhãn thật (vd "animal" -> Dog, Cat, Bird...).

gate_type thay cho "contact: yes/no" nhị phân cũ (đã nhận ra không đủ tổng quát khi bàn về
looking_at/talking_to — quan hệ đó cần "hướng nhìn" chứ không phải khoảng cách):
  - "proximity"   -> Tầng gate L1 sau này lọc theo khoảng cách box (holding, riding...).
  - "orientation" -> cần hướng cơ thể/ánh nhìn, KHÔNG lọc theo khoảng cách (looking_at...).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from config import INDEX_DIR

LABEL_TYPES_PATH = INDEX_DIR / "label_types.json"


@dataclass(frozen=True)
class Relation:
    name: str
    subject_categories: tuple[str, ...]
    object_categories: tuple[str, ...]
    gate_type: str  # "proximity" | "orientation"
    category: str  # "interaction" | "containment"


REGISTRY: list[Relation] = [
    Relation("holding", ("person",), ("object",), "proximity", "interaction"),
    Relation("riding", ("person",), ("vehicle", "animal"), "proximity", "interaction"),
    Relation("wearing", ("person",), ("clothing_accessory",), "proximity", "interaction"),
    Relation("carrying", ("person",), ("object",), "proximity", "interaction"),
    Relation("sitting_on", ("person", "animal"), ("surface", "object"), "proximity", "interaction"),
    Relation("standing_on", ("person", "animal"), ("surface", "object"), "proximity", "interaction"),
    Relation("pushing", ("person",), ("object", "vehicle"), "proximity", "interaction"),
    Relation("pulling", ("person",), ("object", "vehicle"), "proximity", "interaction"),
    Relation("feeding", ("person",), ("animal",), "proximity", "interaction"),
    Relation("looking_at", ("person",), ("person", "object"), "orientation", "interaction"),
    Relation("talking_to", ("person",), ("person", "object"), "orientation", "interaction"),
    Relation("inside", ("object",), ("object", "container"), "proximity", "containment"),
    Relation("on_top_of", ("object",), ("object", "container"), "proximity", "containment"),
    Relation("part_of", ("object",), ("object",), "proximity", "containment"),
]

_label_types: dict[str, list[str]] | None = None


def _load_label_types() -> dict[str, list[str]]:
    global _label_types
    if _label_types is None:
        with open(LABEL_TYPES_PATH, encoding="utf-8") as f:
            _label_types = json.load(f)
    return _label_types


def label_categories(label: str) -> list[str]:
    """Nhãn thật -> danh sách category THÔ (mặc định ["object"] nếu chưa phân loại)."""
    return _load_label_types().get(label, ["object"])


def _effective_categories(label: str) -> set[str]:
    """Category THÔ sau khi sửa lỗi "object" universal: model gắn "object" fallback gần như
    mọi nhãn (kể cả Person/Dog) — nếu để nguyên, "object" sẽ lấn vào person/animal, phá gate
    (vd Person+Person lại khớp "holding" vì Person cũng mang tag "object"). Coi "object" =
    generic KHÔNG PHẢI person/animal — bỏ tag "object" nếu nhãn đã có person hoặc animal.
    """
    cats = set(label_categories(label))
    if "object" in cats and ("person" in cats or "animal" in cats):
        cats.discard("object")
    return cats


def labels_for_category(category: str) -> set[str]:
    """category -> tập nhãn thật (vd "animal" -> {"Dog","Cat",...}), đã áp dụng
    _effective_categories (nên "object" ở đây KHÔNG gồm person/animal)."""
    lt = _load_label_types()
    return {lb for lb in lt if category in _effective_categories(lb)}


def find_relations_for(subj_label: str, obj_label: str) -> list[Relation]:
    """L0 type-filter: với 1 cặp nhãn thật cụ thể (subj, obj), trả relation nào trong
    Registry khớp type — đây là bước ĐẦU TIÊN của gate 4 lớp đã thiết kế (chưa gồm L1-L3)."""
    subj_cats = _effective_categories(subj_label)
    obj_cats = _effective_categories(obj_label)
    return [
        r
        for r in REGISTRY
        if set(r.subject_categories) & subj_cats and set(r.object_categories) & obj_cats
    ]


if __name__ == "__main__":
    import sys

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    # Đúng case "person-person" đã kiểm chứng lý thuyết trước đây (chỉ khớp looking_at/
    # talking_to, contact=no nên bỏ qua L1) — giờ test lại với 514 nhãn thật.
    tests = [
        ("Person", "Person"),
        ("Person", "Dog"),
        ("Person", "Bicycle"),
        ("Person", "Chair"),
        ("Person", "Shirt"),
        ("Person", "Waste container"),
        ("Man", "Woman"),
    ]
    for subj, obj in tests:
        rels = [r.name for r in find_relations_for(subj, obj)]
        print(f"{subj:20s} + {obj:20s} -> {rels}")
