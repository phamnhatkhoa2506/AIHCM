"""Sinh bộ từ đồng nghĩa tiếng Việt cho TOÀN BỘ 514 nhãn OpenImages V4 — 1 lần, offline,
bằng LLM (không phải gõ tay từng case khi gặp — đó là lý do bản SYNONYMS cũ không tổng quát).

Dùng model text-only đã chọn sẵn cho việc này (agent_llm trong v1: meta/llama-3.1-8b-instruct,
qua NVIDIA NIM) — không phải model vision. Kết quả build 1 lần, cache ra file JSON, KHÔNG
gọi API lúc chạy search/app — giữ latency runtime nhanh và không phụ thuộc mạng khi thi.
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
from label_translate import _normalize

load_dotenv()

LABEL_SYNONYMS_PATH = INDEX_DIR / "label_synonyms.json"
MODEL = "meta/llama-3.1-8b-instruct"
N_WORKERS = 2
MAX_RETRIES = 6

_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.environ["NVIDIA_NIM_API_KEY"])

SYSTEM_PROMPT = """You are a Vietnamese lexicon assistant helping build a search synonym table.
Given an English OpenImages object-detection label, output 4-7 common Vietnamese words/phrases
that ordinary Vietnamese speakers would naturally use to refer to that object — including
colloquial forms, short forms, and common alternate names. Do NOT include the literal English
word. Reply with ONLY a JSON array of Vietnamese strings, no explanation, no markdown fences.
Example for "Dog": ["chó", "con chó", "cún", "cún con", "chó con"]
Example for "Waste container": ["thùng rác", "thùng chứa rác", "sọt rác", "giỏ rác"]"""


def _synonyms_for(label: str) -> list[str]:
    delay = 2.0
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = _client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f'English label: "{label}"'},
                ],
                temperature=0.3,
                max_tokens=200,
            )
            text = resp.choices[0].message.content.strip()
            text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            try:
                arr = json.loads(text)
                return [str(s).strip() for s in arr if str(s).strip()]
            except json.JSONDecodeError:
                return []
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


def _validate(raw: dict[str, list[str]], label_vi: dict[str, str]) -> dict[str, list[str]]:
    """Lọc xung đột — KHÔNG tin thẳng LLM (đã thấy model lẫn 'mèo' vào synonym của Dog,
    'cún' vào synonym của Cat). 2 quy tắc:
      1. Bỏ candidate nếu nó CHỨA từ chính (label_vi) của 1 nhãn KHÁC (vd "mèo cỏ" chứa "mèo").
      2. Nếu cùng 1 candidate xuất hiện ở >=2 nhãn khác nhau -> mơ hồ, bỏ khỏi TẤT CẢ (an toàn
         hơn đoán) — trừ khi nó trùng đúng từ chính (label_vi) của 1 trong các nhãn đó, giữ
         lại cho đúng nhãn đó thôi.
    """
    canon_norm = {_normalize(vi): lb for lb, vi in label_vi.items()}

    # bước 1: lọc theo "chứa từ chính của nhãn khác"
    step1: dict[str, list[str]] = {}
    for lb, cands in raw.items():
        kept = []
        for c in cands:
            cn = _normalize(c)
            bad = False
            for other_canon, other_lb in canon_norm.items():
                if other_lb != lb and other_canon and other_canon in cn:
                    bad = True
                    break
            if not bad:
                kept.append(c)
        step1[lb] = kept

    # bước 2: candidate xuất hiện ở nhiều nhãn khác nhau -> mơ hồ, bỏ (trừ khi là từ chính)
    owners: dict[str, set[str]] = {}
    for lb, cands in step1.items():
        for c in cands:
            owners.setdefault(_normalize(c), set()).add(lb)

    result: dict[str, list[str]] = {}
    for lb, cands in step1.items():
        final = []
        for c in cands:
            cn = _normalize(c)
            claimants = owners.get(cn, set())
            if len(claimants) <= 1:
                final.append(c)
            elif cn == _normalize(label_vi.get(lb, "")):
                final.append(c)  # trùng đúng từ chính của nhãn này -> giữ
        result[lb] = final
    return result


def main() -> None:
    labels = _load_labels()
    with open(INDEX_DIR / "label_vi.json", encoding="utf-8") as f:
        label_vi = json.load(f)

    raw: dict[str, list[str]] = {}
    if LABEL_SYNONYMS_PATH.exists():
        with open(LABEL_SYNONYMS_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        print(f"Da co san {sum(1 for v in raw.values() if v)} nhan tu lan chay truoc, "
              f"chi chay lai nhan con thieu", file=sys.stderr)

    todo = [lb for lb in labels if not raw.get(lb)]
    failed: list[str] = []

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_synonyms_for, lb): lb for lb in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            lb = futures[fut]
            try:
                syns = fut.result()
            except Exception as e:
                print(f"loi '{lb}': {e}", file=sys.stderr)
                syns = []
            if not syns:
                failed.append(lb)
            raw[lb] = syns

            if i % 50 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}] (loi: {len(failed)})", file=sys.stderr)

    validated = _validate(raw, label_vi)
    n_removed = sum(len(raw[lb]) - len(validated[lb]) for lb in raw)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(LABEL_SYNONYMS_PATH, "w", encoding="utf-8") as f:
        json.dump(validated, f, ensure_ascii=False, indent=2)

    print(f"Xong: {len(validated)} nhan -> {LABEL_SYNONYMS_PATH}, {len(failed)} nhan loi/rong, "
          f"{n_removed} candidate bi loai vi xung dot")
    if failed:
        print("Nhan loi:", failed[:20])


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
