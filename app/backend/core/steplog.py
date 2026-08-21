"""Log từng bước xử lý (tên bước, chi tiết, thời gian) — truyền xuyên suốt pipeline
(search/tier3_temporal/query_planner/submission_pipeline) để hiển thị lên UI thay vì chỉ
thấy kết quả cuối. Không dùng logging module chuẩn vì cần hiển thị TRỰC TIẾP lên Streamlit
theo đúng thứ tự 1 lần chạy, không phải ghi file.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class StepLog:
    steps: list[dict] = field(default_factory=list)
    _depth: int = field(default=0, init=False, repr=False, compare=False)  # do sau long nhau HIEN TAI - xem timed()

    def add(self, step: str, detail: str, elapsed: float, nested: bool = False) -> None:
        self.steps.append({
            "step": step, "detail": detail, "elapsed_s": round(elapsed, 3), "nested": nested,
        })

    @contextmanager
    def timed(self, step: str):
        """with log.timed("Tầng 1 - lọc thô") as set_detail: ... set_detail(f"...")

        2026-08-20 (theo yeu cau nguoi dung, phat hien qua screenshot that: "Tổng thời gian xử
        lý: 4.65s" o dau trang nhung cong tay tung dong log lai ra ~6.5s) — BUG THAT: danh sach
        `steps` truoc day la PHANG, khong phan biet buoc LONG NHAU (vd search_dense() backfill
        goi lai _rank_single/_rank_rrf VOI log=log BEN TRONG chinh no, xem dense_search.py -
        buoc con "Encode query + xếp hạng" vua duoc ghi la 1 dong RIENG BIET, vua nam TRONG
        khoang thoi gian cua buoc cha "Bộ lọc cứng — bù thêm" -> cong don CA 2 = dem 2 LAN cung
        1 khoang thoi gian. "Tổng thời gian xử lý" o dau trang KHONG bi loi nay (do bang
        time.perf_counter() that o app.py, khong qua log) - loi CHI nam o cho nguoi dung TU
        CONG cac dong log lai se ra so KHAC voi tong that.

        Fix: theo doi DO SAU long nhau (_depth, tang khi VAO 1 timed() moi, giam khi RA) - buoc
        nao bat dau khi DA CO >=1 buoc khac dang mo (_depth>0 luc bat dau) duoc danh dau
        nested=True. total_seconds() (va UI, xem app.py) CHI cong buoc KHONG nested - buoc
        nested van HIEN THI DAY DU (de biet no ton bao nhieu) nhung khong tinh vao tong, tranh
        dem 2 lan. Khong doi ten/interface ham cong khai (add/timed/total_seconds) - CHI them
        tham so `nested` (mac dinh False, khong pha vo cho goi cu neu co)."""
        is_nested = self._depth > 0
        self._depth += 1
        t0 = time.perf_counter()
        detail_holder = {"detail": ""}

        def set_detail(text: str) -> None:
            detail_holder["detail"] = text

        try:
            yield set_detail
        finally:
            self._depth -= 1
            self.add(step, detail_holder["detail"], time.perf_counter() - t0, nested=is_nested)

    def total_seconds(self) -> float:
        """CHI cong cac buoc TOP-LEVEL (khong nested) - buoc nested da nam TRONG khoang thoi
        gian cua buoc cha roi (xem docstring timed()), cong them se dem 2 lan."""
        return sum(s["elapsed_s"] for s in self.steps if not s.get("nested"))
