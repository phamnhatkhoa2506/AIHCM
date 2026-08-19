"""Test kết nối end-to-end: đọc 1 frame thật từ v3 (ảnh + cặp đã qua gate L0-L3), gửi lên
endpoint Modal, kiểm structured output (JSON Schema) có đúng schema P1 không.

Đây là test HẠ TẦNG (đường ống ảnh vào -> JSON đúng schema ra) — CHƯA phải P1 hoàn chỉnh:
prompt ở đây dùng toạ độ box thô, chưa có box_to_position() (mô tả vị trí tự nhiên, tránh
nhầm object cùng nhãn) — việc đó để dành cho lúc build pipeline P1 thật trong v3/, không
thuộc phạm vi thư mục modal_infra/ (chỉ setup hạ tầng).

Chạy: python test_client.py <modal_server_url> [video_id] [local_idx]
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "share"))  # offline/modal_infra/ -> v3/share/

from config import OBJECTS_INDEX_PATH  # noqa: E402
from keyframe_images import read_keyframe_bytes  # noqa: E402
from tiers.pair_gate import gate_pairs  # noqa: E402

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
        }
    },
    "required": ["relations"],
}

SYSTEM_PROMPT = """You are a visual relation extractor. You are given ONE image and a list of
candidate object PAIRS (already pre-filtered geometrically — every pair is worth checking).
For EACH pair, decide which relation (if any) from that pair's allowed list applies, based on
what is DIRECTLY VISIBLE. If unsure or nothing applies, use relation=null. Answer EVERY pair
listed, even with null — do not skip any. Precision matters more than completeness."""


def _pick_frame() -> tuple[str, int]:
    import pandas as pd

    df = pd.read_parquet(OBJECTS_INDEX_PATH, columns=["video_id", "local_idx"])
    counts = df.groupby(["video_id", "local_idx"]).size()
    # frame trung bình, tránh case cực đoan (qua nhieu hoac qua it object) cho lan test dau
    video_id, local_idx = counts[counts == int(counts.median())].index[0]
    return video_id, int(local_idx)


def build_request(video_id: str, local_idx: int) -> tuple[dict, int]:
    import pandas as pd

    df = pd.read_parquet(OBJECTS_INDEX_PATH).reset_index(drop=False).rename(columns={"index": "detection_id"})
    g = df[(df.video_id == video_id) & (df.local_idx == local_idx)]
    detections = g.to_dict("records")
    cands = gate_pairs(detections)
    if not cands:
        raise SystemExit(f"Frame {video_id}/{local_idx} khong co cap nao qua gate, thu frame khac")

    objects_text = "\n".join(
        f"  o{c.subj_id}: {c.subj_label}  o{c.obj_id}: {c.obj_label}" for c in cands
    )
    pairs_text = "\n".join(
        f'  ("o{c.subj_id}", "o{c.obj_id}"): allowed_relations={c.allowed_relations}' for c in cands
    )
    user_prompt = (
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
        "max_tokens": 4000,  # ~25 cap * (schema + evidence text) can nhieu hon 800 (tung cat cut)
        "temperature": 0.1,
    }
    return payload, len(cands)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Dung: python test_client.py <modal_server_url> [video_id] [local_idx]")
    base_url = sys.argv[1].rstrip("/")

    if len(sys.argv) >= 4:
        video_id, local_idx = sys.argv[2], int(sys.argv[3])
    else:
        video_id, local_idx = _pick_frame()

    print(f"Frame test: {video_id}/{local_idx}")
    payload, n_pairs = build_request(video_id, local_idx)
    print(f"So cap gui len: {n_pairs}")

    print("Goi request (lan dau co the cold-start vai phut de tai model + khoi dong vLLM)...")
    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=600)
    resp.raise_for_status()
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    finish_reason = body["choices"][0].get("finish_reason")

    try:
        parsed = json.loads(content)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except json.JSONDecodeError as e:
        print(f"LOI PARSE JSON ({e}) — finish_reason={finish_reason}, do dai content={len(content)}")
        print("--- RAW CONTENT ---")
        print(content)


if __name__ == "__main__":
    main()
