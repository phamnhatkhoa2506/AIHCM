"""Regression test cho query_planner.plan_query() — gom moi case da PHAT HIEN THAT qua qua
trinh test thu cong (xem hoi thoai 2026-08-06/07), chay lai bo nay MOI KHI sua SYSTEM_PROMPT
de biet ngay co pha vo case cu khong, thay vi phai nho va test tay tung cai.

Nguyen tac (nguoi dung dong y 2026-08-07): khong the phong truoc HET moi truong hop bien cua
tieng Viet tu nhien - moi khi phat hien case moi thi THEM vao day, khong danh gia "het loi"
ma la "biet ngay khi co loi moi, khong am tham pha case cu".

Chay: python tests/test_query_planner_regression.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_ROOT / "share"))
_sys.path.insert(0, str(_ROOT / "online"))

from query_planner import plan_query


def _entity_terms(plan: dict) -> set[str]:
    return {e["term"].strip().lower() for e in plan.get("entities", [])}


def _entity_min_count(plan: dict, term: str) -> int | None:
    for e in plan.get("entities", []):
        if e["term"].strip().lower() == term.lower():
            return e["min_count"]
    return None


def _attribute_texts(plan: dict) -> list[str]:
    return [a["attribute_text"].lower() for a in plan.get("attributes", [])]


# Moi case: (ten, query, ham_kiem_tra(plan) -> str|None (None = pass, str = ly do fail))
def _c1(plan):
    # BTC vi du goc: "cây" phai co min_count>=2, KHONG duoc co entity la "áo đỏ"/"áo"
    terms = _entity_terms(plan)
    if "áo đỏ" in terms or "áo" in terms:
        return f"khong duoc trich 'áo'/'áo đỏ' lam entity rieng, thuc te: {terms}"
    cay_count = _entity_min_count(plan, "cây")
    if cay_count is None or cay_count < 2:
        return f"'cây' phai co min_count>=2, thuc te: {cay_count}"
    return None


def _c2(plan):
    # "người đàn ông mặc áo xanh" - khong duoc co entity "áo"/"áo xanh" rieng
    terms = _entity_terms(plan)
    if "áo" in terms or "áo xanh" in terms:
        return f"khong duoc trich 'áo'/'áo xanh' lam entity rieng, thuc te: {terms}"
    return None


def _c3(plan):
    # "2 con chó" - entity cho phai co min_count==2, KHONG co attribute nao (khong mo ta mau/thuoc tinh)
    terms = _entity_terms(plan)
    dog_terms = [t for t in terms if "chó" in t]
    if not dog_terms:
        return f"khong tim thay entity 'chó', thuc te: {terms}"
    cnt = _entity_min_count(plan, dog_terms[0])
    if cnt != 2:
        return f"entity cho phai min_count==2, thuc te: {cnt}"
    return None


def _c4(plan):
    # "hai người mặc áo trắng và xám đang nhìn vào bàn tay ở giữa"
    # KHONG duoc co entity "bàn" (tu ghep "bàn tay" bi tach sai)
    terms = _entity_terms(plan)
    if "bàn" in terms:
        return f"'bàn tay' bi tach sai thanh entity 'bàn', thuc te: {terms}"
    nguoi_count = _entity_min_count(plan, "người")
    if nguoi_count is None or nguoi_count < 2:
        return f"'người' phai co min_count>=2, thuc te: {nguoi_count}"
    return None


def _c5(plan):
    # "một vận động viên đang băng bó cổ tay bị thương" - khong duoc co entity "cổ" doc lap (tu ghep "cổ tay")
    terms = _entity_terms(plan)
    if "cổ" in terms:
        return f"'cổ tay' bi tach sai thanh entity 'cổ', thuc te: {terms}"
    return None


def _c6(plan):
    # "cận cảnh đầu gối của cầu thủ sau cú va chạm" - khong duoc co entity "đầu" doc lap (tu ghep "đầu gối")
    terms = _entity_terms(plan)
    if "đầu" in terms:
        return f"'đầu gối' bi tach sai thanh entity 'đầu', thuc te: {terms}"
    return None


def _secondary_terms(plan: dict) -> set[str]:
    return {e["term"].strip().lower() for e in plan.get("secondary_entities", [])}


def _c7(plan):
    # "hai người mặc áo trắng và xám đang nhìn vào bàn tay ở giữa" (2026-08-07, sau khi them
    # phan cap chinh/phu): "bàn tay" phai vao secondary_entities, KHONG vao entities (hard filter)
    terms = _entity_terms(plan)
    sec_terms = _secondary_terms(plan)
    if "bàn tay" in terms or "bàn" in terms:
        return f"'bàn tay' phai o secondary_entities, khong phai entities (hard filter), thuc te entities: {terms}"
    if "bàn tay" not in sec_terms:
        return f"'bàn tay' phai xuat hien trong secondary_entities, thuc te: {sec_terms}"
    return None


def _c8(plan):
    # BUG THAT phat hien qua benchmark 100 mau (2026-08-11): "một bát salad dưa leo trộn rau"
    # KHONG co tu chi so luong nao - truoc day LLM van trich CA 4 danh tu (bat/salad/dua leo/rau)
    # thanh entities voi min_count=1 mac dinh -> AND ca 4 nhan hiem gan nhu chac chan giao rong
    # (da do that: 0 frame co ca Egg+Wok cung luc). Sau khi sua prompt: KHONG co tu so luong ->
    # entities PHAI rong, de CLIP tu xu ly qua clip_text.
    terms = _entity_terms(plan)
    if terms:
        return f"khong co tu chi so luong nao trong cau -> entities phai RONG, thuc te: {terms}"
    return None


TEST_CASES: list[tuple[str, str, callable]] = [
    (
        "BTC-vi-du-goc",
        "Tìm video về một diễn giả mặc áo đỏ phát biểu tại một cuộc họp báo ngoài trời, phía sau có nhiều cây xanh.",
        _c1,
    ),
    ("ao-xanh-khong-phai-entity", "người đàn ông mặc áo xanh", _c2),
    ("so-luong-don-gian", "2 con chó", _c3),
    ("ban-tay-tu-ghep", "hai người mặc áo trắng và xám đang nhìn vào bàn tay ở giữa", _c4),
    ("co-tay-tu-ghep", "một vận động viên đang băng bó cổ tay bị thương", _c5),
    ("dau-goi-tu-ghep", "cận cảnh đầu gối của cầu thủ sau cú va chạm", _c6),
    ("ban-tay-la-secondary-entity", "hai người mặc áo trắng và xám đang nhìn vào bàn tay ở giữa", _c7),
    ("khong-so-luong-thi-entity-rong", "một bát salad dưa leo trộn rau", _c8),
]


def main() -> int:
    n_pass = 0
    n_fail = 0
    for name, query, check_fn in TEST_CASES:
        plan = plan_query(query)
        reason = check_fn(plan)
        if reason is None:
            print(f"[PASS] {name}")
            n_pass += 1
        else:
            print(f"[FAIL] {name}: {reason}")
            print(f"        query: {query}")
            print(f"        plan : entities={plan.get('entities')} attributes={plan.get('attributes')}")
            n_fail += 1

    print(f"\n{n_pass}/{len(TEST_CASES)} pass, {n_fail} fail")
    return 1 if n_fail else 0


if __name__ == "__main__":
    if _sys.stdout.encoding and _sys.stdout.encoding.lower() != "utf-8":
        try:
            _sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    raise SystemExit(main())
