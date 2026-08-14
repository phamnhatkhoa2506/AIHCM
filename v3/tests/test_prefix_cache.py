import sys, time
sys.path.insert(0, 'online')
from query_planner import plan_query

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# cac cau KHAC NHAU (khong trung lap) - loai tru kha nang do la full-response cache,
# chi con lai kha nang la PREFIX (SYSTEM_PROMPT chung) duoc cache o phia NIM.
queries = [
    "một người đàn ông đội mũ đỏ đang lái xe máy",
    "hai con chó chạy trên bãi cỏ xanh",
    "cô gái mặc váy vàng đứng cạnh xe đạp",
    "ba chiếc thuyền đậu ở bến cảng lúc hoàng hôn",
    "người đầu bếp đang thái rau trong bếp",
    "vận động viên nhảy cao qua xà ngang",
    "em bé cầm quả bóng màu cam",
    "ông cụ ngồi đọc báo trên ghế đá",
]

print("Do thoi gian goi plan_query() cho 8 cau KHAC NHAU, cung 1 SYSTEM_PROMPT dai co dinh:\n")
times = []
for i, q in enumerate(queries, 1):
    t0 = time.perf_counter()
    plan_query(q)
    elapsed = time.perf_counter() - t0
    times.append(elapsed)
    print(f"  [{i}] {elapsed:.2f}s")

print(f"\nTrung binh 3 cau DAU: {sum(times[:3])/3:.2f}s")
print(f"Trung binh 3 cau CUOI: {sum(times[-3:])/3:.2f}s")
