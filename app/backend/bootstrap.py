"""Nối sys.path tới bản sao RIÊNG của app (`backend/core/`) — chứa search/submission logic đã
copy từ v3/share + v3/online rồi tách khỏi v3 hoàn toàn (2026-08-21, theo yêu cầu người dùng:
"tạo file mới trực tiếp vô hệ thống mới... đừng mount vào v3 nữa, hãy làm 2 cái đó tách biệt
nhau"). app/ và v3/ từ nay là 2 hệ thống ĐỘC LẬP — sửa search/submission logic phải sửa CẢ 2 nơi
nếu cần đồng bộ (v3/ vẫn chạy Streamlit song song trong giai đoạn chuyển tiếp, xem README).

Import module này ĐẦU TIÊN, trước bất kỳ import nào khác trong backend/."""
from __future__ import annotations

import sys
from pathlib import Path

_CORE_DIR = Path(__file__).resolve().parent / "core"
_sp = str(_CORE_DIR)
if _sp not in sys.path:
    sys.path.insert(0, _sp)
