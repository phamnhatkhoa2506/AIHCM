"""Đo Tier 1-3 hiện có bằng ĐÚNG công thức R-Score/Final Score của BTC
(`Thong tin vong So tuyen AIC2026.pdf` mục 2) — không tự bịa metric riêng.

R-Score:
  KIS:   I(video đúng ∧ frame ∈ [s,e])
  Q&A:   I(video đúng ∧ frame ∈ [s,e] ∧ answer khớp nghĩa)
  TRAKE: (số anchor khớp)/N nếu đúng video, else 0
Final Score = mean(R@{1,5,20,50,100})

Đo Q&A "khớp nghĩa" ở đây dùng normalize + containment (XẤP XỈ, không phải LLM-judge thật —
đủ cho việc soát Tier 1-3, không phải con số nộp bài chính thức).

Chạy: python evaluate.py [queries.jsonl]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_v3_root = Path(__file__).resolve().parent.parent.parent  # offline/benchmark/ -> v3/
sys.path.insert(0, str(_v3_root / "share"))  # module dung chung (config.py, tiers/...)
sys.path.insert(0, str(_v3_root / "online"))  # submission_pipeline.py chuyen sang online/

from submission_pipeline import answer_kis, answer_qa, answer_trake  # noqa: E402

R_AT_K = [1, 5, 20, 50, 100]


def _r_score_kis(row: pd.Series, gt_video_id: str, gt_range: list[int]) -> float:
    return float(row["video_id"] == gt_video_id and gt_range[0] <= row["frame_id"] <= gt_range[1])


def _r_score_qa(row: pd.Series, gt_video_id: str, gt_range: list[int], gt_answer: str) -> float:
    if row["video_id"] != gt_video_id or not (gt_range[0] <= row["frame_id"] <= gt_range[1]):
        return 0.0
    a = str(row["answer"]).strip().lower()
    gt = str(gt_answer).strip().lower()
    return float(gt in a or a in gt)


def _r_score_trake(row: pd.Series, gt_video_id: str, gt_frame_ranges: list[list[int]]) -> float:
    if row["video_id"] != gt_video_id:
        return 0.0
    n = len(gt_frame_ranges)
    hits = 0
    for j, (s, e) in enumerate(gt_frame_ranges):
        fid = row.get(f"frame_id_{j + 1}")
        if fid is not None and s <= fid <= e:
            hits += 1
    return hits / n


def final_score(r_scores: list[float]) -> dict:
    out = {}
    for k in R_AT_K:
        out[f"R@{k}"] = max(r_scores[:k]) if r_scores else 0.0
    out["final_score"] = sum(out.values()) / len(R_AT_K)
    return out


def evaluate_query(q: dict) -> dict:
    qtype = q["type"]
    if qtype == "KIS":
        results = answer_kis(q["query"], top_k=100)
        r_scores = [_r_score_kis(row, q["gt_video_id"], q["gt_frame_range"]) for _, row in results.iterrows()]
    elif qtype == "QA":
        results = answer_qa(q["event_query"], q["question"], top_k=100, vqa_top_n=5)
        r_scores = [
            _r_score_qa(row, q["gt_video_id"], q["gt_frame_range"], q["gt_answer"])
            for _, row in results.iterrows()
        ]
    elif qtype == "TRAKE":
        results = answer_trake(q["anchors"], top_k=100)
        r_scores = [_r_score_trake(row, q["gt_video_id"], q["gt_frame_ranges"]) for _, row in results.iterrows()]
    else:
        raise ValueError(f"Không biết type: {qtype}")

    scores = final_score(r_scores)
    return {"id": q["id"], "type": qtype, "tags": q.get("tags", []), "n_results": len(r_scores), **scores}


def main(queries_path: str) -> None:
    with open(queries_path, encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    rows = []
    for q in queries:
        print(f"Đang chấm: {q['id']} ({q['type']})...", file=sys.stderr)
        try:
            rows.append(evaluate_query(q))
        except Exception as e:
            print(f"  lỗi: {e}", file=sys.stderr)
            rows.append(
                {"id": q["id"], "type": q["type"], "tags": q.get("tags", []), "final_score": 0.0, "error": str(e)}
            )

    df = pd.DataFrame(rows)
    print()
    print(df.to_string(index=False))
    print()
    print(f"Final Score trung bình: {df['final_score'].mean():.3f}")
    print()
    print("Theo type:")
    print(df.groupby("type")["final_score"].mean().to_string())

    tag_scores: dict[str, list[float]] = {}
    for _, row in df.iterrows():
        for t in row.get("tags") or []:
            tag_scores.setdefault(t, []).append(row["final_score"])
    if tag_scores:
        print()
        print("Theo tag (bóc tách điểm yếu):")
        for t, scores in sorted(tag_scores.items()):
            print(f"  {t:20s} mean={sum(scores) / len(scores):.3f} (n={len(scores)})")

    out_path = Path(queries_path).with_suffix("")
    df.to_csv(f"{out_path}.results.csv", index=False)
    print(f"\nĐã lưu chi tiết -> {out_path}.results.csv")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "queries.jsonl")
    main(path)
