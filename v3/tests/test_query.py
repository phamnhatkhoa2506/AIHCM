import sys, time
sys.path.insert(0, 'online')
from query_planner import planned_search
from steplog import StepLog

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

log = StepLog()
t0 = time.perf_counter()
results, plan = planned_search("con lân màu vàng", top_k=12, log=log)
print("TONG:", round(time.perf_counter() - t0, 2), "s")
for s in log.steps:
    print(f"  {s['step']}: {s['elapsed_s']}s - {s['detail']}")
