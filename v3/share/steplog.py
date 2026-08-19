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

    def add(self, step: str, detail: str, elapsed: float) -> None:
        self.steps.append({"step": step, "detail": detail, "elapsed_s": round(elapsed, 3)})

    @contextmanager
    def timed(self, step: str):
        """with log.timed("Tầng 1 - lọc thô") as set_detail: ... set_detail(f"...")"""
        t0 = time.perf_counter()
        detail_holder = {"detail": ""}

        def set_detail(text: str) -> None:
            detail_holder["detail"] = text

        yield set_detail
        self.add(step, detail_holder["detail"], time.perf_counter() - t0)

    def total_seconds(self) -> float:
        return sum(s["elapsed_s"] for s in self.steps)
