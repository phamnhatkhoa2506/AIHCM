"""Dich top N nhan open-vocab (Grounding DINO, open_vocab_detections.parquet) sang tieng Viet
qua NIM LLM (cung model dung trong vocab_discovery.py/query_planner.py) - de label_translate.py
resolve() nhan duoc query tieng Viet (vd "lan mua") map sang dung nhan tieng Anh da luu.

CHI dich top N theo tan suat - long-tail nhan bi phan manh (do Grounding DINO noi cum tu khong
sach, xem hoi thoai 2026-08-06) qua nhieu de dich het, top 300 da phu ~62% tong detection.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))

import json
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from config import INDEX_DIR

load_dotenv()

TOP_N = 300
OUT_PATH = INDEX_DIR / "open_vocab_vi.json"
BATCH_SIZE = 40  # so nhan/lan goi LLM - vua du de output khong bi cat

_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.environ["NVIDIA_NIM_API_KEY"])
MODEL = "meta/llama-3.1-8b-instruct"

SYSTEM_PROMPT = """You translate short English object-detection labels into natural, SHORT
Vietnamese phrases (1-4 words) a user would type when searching a video. Keep the meaning
concrete and visual. Reply with ONLY a JSON object mapping each input label (as key, EXACT
string given) to its Vietnamese translation (as value). No explanation."""


def _translate_batch(labels: list[str]) -> dict[str, str]:
    user_prompt = "Translate these labels:\n" + "\n".join(f"- {lb}" for lb in labels)
    resp = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=60 * len(labels),
        temperature=0.1,
    )
    content = resp.choices[0].message.content.strip()
    content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(content)


def main() -> None:
    df = pd.read_parquet(INDEX_DIR / "open_vocab_detections.parquet")
    top_labels = df.label.value_counts().head(TOP_N).index.tolist()
    print(f"Dich {len(top_labels)} nhan (top theo tan suat)...", file=sys.stderr)

    result: dict[str, str] = {}
    for i in range(0, len(top_labels), BATCH_SIZE):
        batch = top_labels[i: i + BATCH_SIZE]
        try:
            translated = _translate_batch(batch)
            result.update(translated)
            print(f"[{i + len(batch)}/{len(top_labels)}] OK", file=sys.stderr)
        except Exception as e:
            print(f"[{i + len(batch)}/{len(top_labels)}] LOI: {type(e).__name__} {e}", file=sys.stderr)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nDa dich {len(result)}/{len(top_labels)} nhan -> {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
