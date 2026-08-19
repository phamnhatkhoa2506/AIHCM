"""Phân loại 514 nhãn OpenImages V4 vào category ngữ nghĩa đóng (7 loại) — dùng cho
Relation Registry (tiers/tier4_graph.py cần cái này để map subject_types/object_types
sang đúng nhãn thật, thay vì placeholder "person"/"vehicle" lúc thiết kế lý thuyết).

Không cần similarity/embedding ở đây (khác bài toán đồng nghĩa) — đây là CLASSIFICATION
vào 1 tập category CỐ ĐỊNH nhỏ, LLM làm tốt, chỉ cần validate output nằm trong tập cho phép
(không hallucinate category lạ) — rủi ro thấp hơn nhiều so với sinh từ đồng nghĩa tự do.
"""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

from config import INDEX_DIR, OBJECTS_INDEX_PATH

load_dotenv()

LABEL_TYPES_PATH = INDEX_DIR / "label_types.json"
MODEL = "meta/llama-3.1-8b-instruct"
N_WORKERS = 2
MAX_RETRIES = 6

# Category đóng — khớp đúng subject_types/object_types các relation đã thiết kế trước đó
# (holding/riding/wearing/carrying/sitting_on/pushing/feeding/looking_at/inside/part_of).
CATEGORIES = ["person", "animal", "vehicle", "clothing_accessory", "surface", "container", "object"]

# Patch tay — lỗi phát hiện khi soát toàn bộ output (2026-08-05): model bị đánh lừa bởi
# chữ trong tên nhãn ("Hot dog" -> animal vì có chữ "dog"; "Rays and skates" (cá đuối) ->
# clothing_accessory vì có chữ "skates") hoặc suy luận lỏng lẻo (Whisk/Horn -> clothing;
# Fire hydrant/Street light -> surface).
OVERRIDES: dict[str, list[str]] = {
    "Lily": ["object"],
    "Hot dog": ["object"],
    "Houseplant": ["object"],
    "Teddy bear": ["object"],
    "Dog bed": ["surface", "object"],
    "Whisk": ["object"],
    "Horn": ["object"],
    "Rays and skates": ["animal", "object"],
    "Toothbrush": ["object"],
    "Binoculars": ["object"],
    "Fire hydrant": ["object"],
    "Street light": ["object"],
}

_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.environ["NVIDIA_NIM_API_KEY"])

SYSTEM_PROMPT = f"""You are classifying object-detection labels (OpenImages V4) into semantic
categories for a scene-graph relation system. Given a label, output ALL applicable categories
from this FIXED list (use these exact strings, nothing else):
  person, animal, vehicle, clothing_accessory, surface, container, object

Rules:
- "person": human individuals (Person, Man, Woman, Boy, Girl...).
- "animal": living animals (Dog, Cat, Bird...) — NOT humans.
- "vehicle": things that can be ridden/driven (Car, Bicycle, Boat...).
- "clothing_accessory": wearable items (Shirt, Hat, Glasses, Necklace...).
- "surface": things a person/animal can sit/stand ON (Chair, Bench, Table, Stairs, ground-like surfaces...).
- "container": things that can hold other objects, or be "inside"/"part of" something (Box, Bag, Cabinet, Bottle, room/building parts...).
- "object": generic fallback — always include this UNLESS the label is clearly and ONLY one of the specific categories above.
A label CAN have multiple categories (e.g. "Bicycle" is vehicle; "Chair" is surface AND object).
Reply with ONLY a JSON array of category strings, no explanation, no markdown fences.
Example for "Bicycle": ["vehicle", "object"]
Example for "Chair": ["surface", "object"]
Example for "Dog": ["animal", "object"]"""


def _types_for(label: str) -> list[str]:
    delay = 2.0
    last_err: Exception | None = None
    for _ in range(MAX_RETRIES):
        try:
            resp = _client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f'Label: "{label}"'},
                ],
                temperature=0.0,
                max_tokens=80,
            )
            text = resp.choices[0].message.content.strip()
            text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            try:
                arr = json.loads(text)
            except json.JSONDecodeError:
                return ["object"]
            valid = [c for c in arr if c in CATEGORIES]
            return valid or ["object"]
        except Exception as e:
            last_err = e
            if "429" not in str(e) and "502" not in str(e) and "503" not in str(e):
                raise
            time.sleep(delay)
            delay = min(delay * 1.8, 30.0)
    raise last_err  # type: ignore[misc]


def _load_labels() -> list[str]:
    import pandas as pd

    df = pd.read_parquet(OBJECTS_INDEX_PATH, columns=["label"])
    return sorted(df["label"].unique().tolist())


def main() -> None:
    labels = _load_labels()

    result: dict[str, list[str]] = {}
    if LABEL_TYPES_PATH.exists():
        with open(LABEL_TYPES_PATH, encoding="utf-8") as f:
            result = json.load(f)
        print(f"Da co san {len(result)} nhan tu lan chay truoc, chi chay lai nhan con thieu", file=sys.stderr)

    todo = [lb for lb in labels if lb not in result]
    failed: list[str] = []

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_types_for, lb): lb for lb in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            lb = futures[fut]
            try:
                cats = fut.result()
            except Exception as e:
                print(f"loi '{lb}': {e}", file=sys.stderr)
                cats = ["object"]
                failed.append(lb)
            result[lb] = cats

            if i % 50 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}] (loi: {len(failed)})", file=sys.stderr)

    result.update(OVERRIDES)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(LABEL_TYPES_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    from collections import Counter

    cat_counts = Counter(c for cats in result.values() for c in cats)
    print(f"Xong: {len(result)} nhan -> {LABEL_TYPES_PATH}, {len(failed)} loi (fallback 'object')")
    print("Phan bo category:", dict(cat_counts))


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
