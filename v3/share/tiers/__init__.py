"""4 tầng pipeline search, chạy tuần tự trong search.py::search():

  Tầng 1 (tier1_filter) — lọc thô: định nghĩa tập ứng viên (video_id, local_idx) TRƯỚC.
  Tầng 2 (tier2_vector)  — CLIP: xếp hạng theo ngữ nghĩa TRONG tập ứng viên. ĐANG DÙNG.
  Tầng 3 (tier3_temporal) — thứ tự sự kiện trong video (anchor-chain). CHƯA TRIỂN KHAI.
  Tầng 4 (tier4_graph)   — xác minh quan hệ object-object (Relation Registry). CHƯA TRIỂN KHAI.

Mỗi tầng là 1 module riêng, hàm `apply()` (hoặc `rank()` ở tầng 2) là điểm vào chuẩn —
thêm/bớt/đổi 1 tầng không ảnh hưởng tầng khác, miễn giữ đúng kiểu input/output.
"""
