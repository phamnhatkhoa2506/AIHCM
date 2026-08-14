"""So sanh: LLM (query_planner.plan_query) vs rule-based thuan tuy (pyvi/underthesea POS-tag +
tu dien so luong) cho task trich xuat entity+min_count - xem thu vien "thuong" co toi uu hon
LLM "to" khong (nguoi dung hoi 2026-08-07).
"""
import sys
import time

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import underthesea

# tu chi so luong -> gia tri min_count tuong ung (giong logic prompt LLM dang dung)
QUANTITY_WORDS = {
    "một": 1, "1": 1,
    "hai": 2, "2": 2, "đôi": 2, "cặp": 2,
    "ba": 3, "3": 3,
    "bốn": 4, "4": 4,
    "năm": 5, "5": 5,
    "nhiều": 2, "vài": 2, "một_vài": 2, "một vài": 2, "một số": 2,
}

# nhan POS coi la danh tu (co the la entity) - underthesea dung tagset kieu N/Np/Nc...
NOUN_TAGS = {"N", "Np", "Nc", "Ny", "Nu"}


def rule_based_extract(query: str) -> dict:
    """Tra ve dang gan giong output plan_query() - CHI phan entities/min_count (khong lam
    attributes/clip_text, vi do la quyet dinh ngu nghia LLM lam tot hon rule thuan)."""
    tagged = underthesea.pos_tag(query)  # list[(tu_da_gop, nhan_pos)]

    entities = []
    i = 0
    while i < len(tagged):
        word, tag = tagged[i]
        if tag in NOUN_TAGS:
            # tim tu so luong ngay TRUOC danh tu nay (neu co)
            min_count = 1
            if i > 0:
                prev_word = tagged[i - 1][0].lower()
                if prev_word in QUANTITY_WORDS:
                    min_count = QUANTITY_WORDS[prev_word]
            entities.append({"term": word.replace("_", " "), "min_count": min_count})
        i += 1
    return {"entities": entities}


TEST_QUERIES = [
    "Tìm video về một diễn giả mặc áo đỏ phát biểu tại một cuộc họp báo ngoài trời, phía sau có nhiều cây xanh.",
    "người đàn ông mặc áo xanh",
    "hai người mặc áo trắng và xám đang nhìn vào bàn tay ở giữa",
    "một vận động viên đang băng bó cổ tay bị thương",
    "cận cảnh đầu gối của cầu thủ sau cú va chạm",
]

print("=== RULE-BASED (pyvi/underthesea, khong goi LLM) ===\n")
for q in TEST_QUERIES:
    t0 = time.perf_counter()
    result = rule_based_extract(q)
    elapsed = time.perf_counter() - t0
    print(f"Q: {q}")
    print(f"  entities ({elapsed*1000:.1f}ms): {result['entities']}")
    print()
