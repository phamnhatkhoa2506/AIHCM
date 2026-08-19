import sys
sys.path.insert(0, 'online')
from query_planner import planned_search
from steplog import StepLog

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

log = StepLog()
results, plan = planned_search("hai người mặc áo trắng và xám đang nhìn vào bàn tay ở giữa", top_k=12, log=log)
print("entities:", plan.get("entities"))
print("secondary_entities:", plan.get("secondary_entities"))
print("so ket qua:", len(results))
if "secondary_boost" in results.columns:
    print(results[["video_id", "local_idx", "score_before_secondary_boost", "secondary_boost", "score"]].to_string(index=False))
print()
for s in log.steps:
    print(f"  {s['step']}: {s['elapsed_s']}s - {s['detail']}")
