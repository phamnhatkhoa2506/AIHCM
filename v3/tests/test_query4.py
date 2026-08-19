import sys, json
sys.path.insert(0, 'online')
from query_planner import plan_query

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

queries = [
    "một vận động viên đang băng bó cổ tay bị thương",
    "cận cảnh đầu gối của cầu thủ sau cú va chạm",
    "người phụ nữ đeo đồng hồ ở cổ tay đang cầm ly cà phê",
]
for q in queries:
    plan = plan_query(q)
    print(q)
    print(json.dumps(plan, ensure_ascii=False))
    print()
