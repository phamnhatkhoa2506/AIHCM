import sys, json
sys.path.insert(0, 'online')
from query_planner import plan_query

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

q = "hai người mặc áo trắng và xám đang nhìn vào bàn tay ở giữa"
plan = plan_query(q)
print(json.dumps(plan, ensure_ascii=False, indent=2))
