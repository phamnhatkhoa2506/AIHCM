"""Chạy gate L0-L3 (pair_gate.gate_pairs) trên TOÀN corpus thật để xem 14 relation trong
Registry có thực sự khớp với thành phần object của corpus này không — relation nào "chết"
(gần như không bao giờ có candidate) là tín hiệu nên xem lại/bỏ, relation nào phổ biến là
tín hiệu đáng ưu tiên khi làm P1 (VLM) sau này.

Rẻ, không cần VLM — chỉ hình học + tra bảng, chạy được cho cả corpus.
"""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/

import sys
from collections import Counter

import pandas as pd

from config import OBJECTS_INDEX_PATH
from tiers.pair_gate import gate_pairs


def main() -> None:
    df = pd.read_parquet(OBJECTS_INDEX_PATH).reset_index(drop=False).rename(columns={"index": "detection_id"})
    groups = df.groupby(["video_id", "local_idx"], sort=False)
    n_total_frames = groups.ngroups

    relation_counts: Counter[str] = Counter()
    pairs_per_frame: list[int] = []
    frames_with_candidates = 0
    n_scanned = 0

    for (video_id, local_idx), g in groups:
        n_scanned += 1
        if len(g) < 2:
            pairs_per_frame.append(0)
            continue
        cands = gate_pairs(g.to_dict("records"))
        pairs_per_frame.append(len(cands))
        if cands:
            frames_with_candidates += 1
        for c in cands:
            for r in c.allowed_relations:
                relation_counts[r] += 1

        if n_scanned % 20000 == 0:
            print(f"[{n_scanned}/{n_total_frames}]", file=sys.stderr)

    print(f"Frame quet: {n_scanned}, co >=1 cap qua gate: {frames_with_candidates} "
          f"({100*frames_with_candidates/n_scanned:.1f}%)")
    print()
    print("So candidate theo relation (1 cap co the tinh nhieu relation cung luc):")
    for rel, cnt in relation_counts.most_common():
        print(f"  {rel:14s} {cnt}")

    dead = [r.name for r in __import__("relation_registry").REGISTRY if r.name not in relation_counts]
    if dead:
        print()
        print("Relation KHONG BAO GIO xuat hien lam candidate (corpus nay):", dead)


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
