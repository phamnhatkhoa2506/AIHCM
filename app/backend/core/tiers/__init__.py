"""Các tầng search (share/tiers/):

  filter — lọc thô: định nghĩa tập ứng viên (video_id, local_idx) TRƯỚC (object/OCR/
    metadata). ĐANG DÙNG — cả pipeline dense mới lẫn hard-filter đều gọi qua đây.
  dense_search — Tầng 2 (vector): SigLIP2/PE-Core/BEiT-3 + RRF fusion trên bộ dense tự trích.
    ĐANG DÙNG — pipeline chính app.py chạy live.
  dense_temporal — Tầng 3 (TRAKE/temporal, anchor-chain qua DP) trên bộ dense. ĐANG DÙNG.
  tier4_graph — xác minh quan hệ object-object (Relation Registry). CHƯA TRIỂN KHAI.
  pair_gate — hỗ trợ tier4_graph.

# 2026-08-20 (theo yeu cau nguoi dung: "dọn dẹp triệt để... code cũ") - da XOA HAN
# tier2_vector.py/tier3_temporal.py (pipeline CLIP-ViT-B-32 + keyframe BTC goc, khong con duoc
# app.py goi tu 2026-08-15/18, chi con lam baseline benchmark - da xac nhan dense pipeline
# thang ro nen bo baseline luon). Xem git history neu can xem lai pipeline cu.
"""
