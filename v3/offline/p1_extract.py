"""Phase P1 hoàn chỉnh: 1 frame -> relation ngữ nghĩa đã verify (P1 VLM extract + P2 confidence
gate + P3 spatial cross-check). Dùng `tiers/pair_gate.py` (L0-L3) để biết cặp nào cần hỏi,
gọi Modal endpoint (Qwen2.5-VL-7B qua vLLM, xem modal_infra/), lọc kết quả trước khi trả về.

Khác `modal_infra/test_client.py` (chỉ test hạ tầng, dùng toạ độ box thô) ở chỗ: dùng
`box_to_position()` — mô tả vị trí tự nhiên, tránh model nhầm object cùng nhãn trong 1 frame
(bẫy đã nhận ra từ thiết kế lý thuyết, giờ mới code thật).
"""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/

import base64
import json
from dataclasses import dataclass

import pandas as pd
import requests

from config import MODAL_P1_URL, OBJECTS_INDEX_PATH
from keyframe_images import read_keyframe_bytes
from tiers.pair_gate import PairCandidate, gate_pairs

# P2 — ngưỡng confidence (đã chốt từ thiết kế trước)
C_DIRECT = 0.55
C_INFERRED = 0.7

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subj_id": {"type": "integer"},
                    "obj_id": {"type": "integer"},
                    "relation": {"type": ["string", "null"]},
                    "conf": {"type": "number"},
                    "inferred": {"type": "boolean"},
                    "evidence": {"type": "string"},
                },
                "required": ["subj_id", "obj_id", "relation", "conf", "inferred", "evidence"],
            },
        },
        # TRƯỜNG MỞ (2026-08-05) — KHÔNG qua gate, không giới hạn theo Registry. Không tin dùng
        # như "relations" ở trên (không có ngưỡng/loại để áp P2/P3) — chỉ để PHÁT HIỆN relation
        # thật ngoài 14 cái trong Registry, dùng làm bằng chứng làm giàu Registry sau này.
        "extra_observations": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["relations", "extra_observations"],
}

SYSTEM_PROMPT = """You are a visual relation extractor. You are given ONE image, a list of
objects (with rough position in the frame), and a list of candidate object PAIRS (already
pre-filtered geometrically — every pair is worth checking). For EACH pair, decide which
relation (if any) from that pair's allowed list applies, based on what is DIRECTLY VISIBLE.
Use position hints to tell apart objects sharing the same label. If unsure or nothing applies,
use relation=null. Answer EVERY pair listed, even with null — do not skip any. If a relation
can only be guessed from posture/context rather than seen directly, set inferred=true and
require higher confidence. Precision matters more than completeness.

IMPORTANT: "conf" MUST be a decimal number between 0.0 and 1.0 (e.g. 0.85) — NEVER a
percentage like 85 or 100.

ADDITIONALLY: after checking all listed pairs, look at the WHOLE image once more and note in
"extra_observations" (short strings, empty list if none) any OTHER visible interaction or
relation between objects that is NOT covered by the pairs/relations above — do not restrict
yourself to the allowed relation names for this part."""


def box_to_position(ymin: float, xmin: float, ymax: float, xmax: float) -> str:
    """Toạ độ box (normalize 0-1) -> mô tả vị trí tự nhiên (vd "top-left", "center").
    Lưới 3x3 theo TÂM box — đủ để phân biệt object cùng nhãn trong 1 frame, không cần chính xác
    tuyệt đối như toạ độ số (mà VLM đọc kém, xem lý do trong relation_extraction_prompt.md)."""
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2

    if cx < 1 / 3:
        h = "left"
    elif cx < 2 / 3:
        h = "center"
    else:
        h = "right"

    if cy < 1 / 3:
        v = "top"
    elif cy < 2 / 3:
        v = "middle"
    else:
        v = "bottom"

    if h == "center" and v == "middle":
        return "center"
    if v == "middle":
        return h
    if h == "center":
        return v
    return f"{v}-{h}"


def _load_frame_detections(video_id: str, local_idx: int) -> list[dict]:
    df = pd.read_parquet(OBJECTS_INDEX_PATH).reset_index(drop=False).rename(columns={"index": "detection_id"})
    g = df[(df.video_id == video_id) & (df.local_idx == local_idx)]
    return g.to_dict("records")


def build_request(video_id: str, local_idx: int) -> tuple[dict, list[PairCandidate]] | None:
    """None nếu frame không có cặp nào qua gate (không đáng gọi VLM)."""
    detections = _load_frame_detections(video_id, local_idx)
    cands = gate_pairs(detections)
    if not cands:
        return None

    by_id = {d["detection_id"]: d for d in detections}
    obj_ids = sorted({c.subj_id for c in cands} | {c.obj_id for c in cands})
    objects_text = "\n".join(
        f"  o{oid}: {by_id[oid]['label']} ({box_to_position(by_id[oid]['ymin'], by_id[oid]['xmin'], by_id[oid]['ymax'], by_id[oid]['xmax'])})"
        for oid in obj_ids
    )
    pairs_text = "\n".join(
        f'  ("o{c.subj_id}", "o{c.obj_id}"): allowed_relations={c.allowed_relations}' for c in cands
    )
    user_prompt = (
        f"OBJECTS (id: label (position)):\n{objects_text}\n\n"
        f"CANDIDATE PAIRS TO CHECK (subj_id, obj_id, allowed relations):\n{pairs_text}\n\n"
        f"Reply with JSON matching the given schema. subj_id/obj_id are the raw integer IDs "
        f"(strip the 'o' prefix)."
    )

    img_bytes = read_keyframe_bytes(video_id, local_idx)
    img_b64 = base64.b64encode(img_bytes).decode()

    payload = {
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ],
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "relations", "schema": OUTPUT_SCHEMA, "strict": True},
        },
        "max_tokens": 200 + 150 * len(cands),  # ~150 token/cap (schema + evidence) + đệm
        "temperature": 0.1,
    }
    return payload, cands


def call_model(payload: dict, base_url: str = MODAL_P1_URL, timeout: int = 300) -> dict:
    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=timeout)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return json.loads(content)


def _normalize_conf(conf: float) -> float:
    """Phòng thân: model đôi khi trả thang 0-100 dù prompt đã dặn 0-1 (đã gặp thật khi test,
    conf=100.0) — không tin prompt tuân thủ tuyệt đối, luôn chuẩn hoá lại trước khi so ngưỡng."""
    return conf / 100.0 if conf > 1.0 else conf


def p2_confidence_gate(raw_relations: list[dict]) -> list[dict]:
    """Giữ lại quan hệ (relation != null) đạt ngưỡng theo nhánh direct/inferred."""
    kept = []
    for r in raw_relations:
        if not r.get("relation"):
            continue
        r = dict(r)
        r["conf"] = _normalize_conf(r.get("conf", 0.0))
        threshold = C_INFERRED if r.get("inferred") else C_DIRECT
        if r["conf"] >= threshold:
            kept.append(r)
    return kept


@dataclass
class SpatialIndex:
    """Tra nhanh spatial_edges.parquet theo (subj_id, obj_id) — build 1 lần, dùng lại."""
    pairs: set[tuple[int, int]]

    @classmethod
    def load(cls, path: str = "index/spatial_edges.parquet") -> "SpatialIndex":
        df = pd.read_parquet(path, columns=["subj_detection_id", "obj_detection_id"])
        pairs = set(zip(df["subj_detection_id"], df["obj_detection_id"]))
        pairs |= set(zip(df["obj_detection_id"], df["subj_detection_id"]))  # đối xứng cho tra 2 chiều
        return cls(pairs=pairs)

    def has_geometric_link(self, subj_id: int, obj_id: int) -> bool:
        return (subj_id, obj_id) in self.pairs


# Relation nào BẮT BUỘC phải có cạnh hình học (inside/on_top_of) đi kèm mới hợp lý —
# claim mà không có box nào chồng/gần nhau chút nào thì đáng ngờ.
SPATIAL_REQUIRED_RELATIONS = {"riding", "sitting_on", "standing_on"}


def p3_spatial_crosscheck(relations: list[dict], spatial_index: SpatialIndex) -> list[dict]:
    """Hạ conf 1 nửa (không loại thẳng — vẫn có thể đúng, chỉ bớt tin) nếu claim thuộc
    SPATIAL_REQUIRED_RELATIONS mà không có cạnh hình học nào giữa đúng cặp đó."""
    out = []
    for r in relations:
        r = dict(r)
        if r["relation"] in SPATIAL_REQUIRED_RELATIONS and not spatial_index.has_geometric_link(
            r["subj_id"], r["obj_id"]
        ):
            r["conf"] = r["conf"] * 0.5
            r["spatial_checked"] = False
        else:
            r["spatial_checked"] = True
        out.append(r)
    return out


def extract_frame(
    video_id: str, local_idx: int, spatial_index: SpatialIndex, base_url: str = MODAL_P1_URL
) -> tuple[list[dict], list[str]]:
    """Điểm vào chính: 1 frame -> (relation đã qua P1+P2+P3 — tin dùng, ghi vào
    semantic_edges.parquet; extra_observations — KHÔNG qua gate, chỉ để soát/làm giàu Registry,
    không ghi chung bảng với relation đã verify)."""
    built = build_request(video_id, local_idx)
    if built is None:
        return [], []
    payload, _cands = built

    raw = call_model(payload, base_url)
    kept = p2_confidence_gate(raw.get("relations", []))
    final = p3_spatial_crosscheck(kept, spatial_index)

    for r in final:
        r["video_id"] = video_id
        r["local_idx"] = local_idx
    return final, raw.get("extra_observations", [])


if __name__ == "__main__":
    import sys

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    vid = sys.argv[1] if len(sys.argv) > 1 else "L24_V027"
    li = int(sys.argv[2]) if len(sys.argv) > 2 else 143

    sp = SpatialIndex.load()
    result, extra = extract_frame(vid, li, sp)
    print(f"Frame {vid}/{li}: {len(result)} relation qua P1+P2+P3 (tin dung)")
    for r in result:
        print(f"  o{r['subj_id']} -{r['relation']}-> o{r['obj_id']}  conf={r['conf']:.2f} "
              f"inferred={r['inferred']} spatial_checked={r['spatial_checked']}")

    print(f"\nextra_observations ({len(extra)} — KHONG qua gate, chi de soat/lam giau Registry):")
    for obs in extra:
        print(f"  - {obs}")
