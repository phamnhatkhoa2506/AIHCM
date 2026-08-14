"""Lay mau 100 (video_id, local_idx) rai theo dung phan bo kenh that (khong deu ngau nhien -
tranh thien lech ve 1 loai noi dung), sau do ghep thanh cac "contact sheet" (luoi anh) de xem
hang loat qua Read tool thay vi mo 100 anh rieng le - hieu qua hon nhieu, van dam bao XEM THAT
tung frame truoc khi viet query (khong doan mu tu metadata).

Chay: python offline/benchmark/sample_candidates.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "share"))

import io
import json
import random

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from config import INDEX_DIR
from keyframe_images import read_keyframe_bytes

N_SAMPLES = 100
GRID_COLS = 4
GRID_ROWS = 3
PER_SHEET = GRID_COLS * GRID_ROWS  # 12/sheet -> ~9 sheet cho 100 anh
CELL_W, CELL_H = 320, 180  # giu ti le 16:9, du nho de doc

OUT_DIR = _Path(__file__).resolve().parent / "candidate_sheets"
OUT_DIR.mkdir(exist_ok=True)
MANIFEST_PATH = _Path(__file__).resolve().parent / "candidate_manifest.json"

random.seed(42)  # deterministic - chay lai ra dung mau cu neu can


def sample_candidates() -> list[dict]:
    meta = pd.read_parquet(INDEX_DIR / "meta.parquet")
    video_meta = pd.read_parquet(INDEX_DIR / "video_metadata.parquet")

    # phan bo kenh THAT (da do truoc do trong hoi thoai) - lay mau theo dung ty le nay
    author_counts = video_meta["author"].value_counts()
    total_videos = len(video_meta)

    candidates = []
    for author, n_videos_author in author_counts.items():
        n_take = max(1, round(N_SAMPLES * n_videos_author / total_videos))
        vids = video_meta[video_meta["author"] == author]["video_id"].tolist()
        chosen_vids = random.sample(vids, min(n_take, len(vids)))
        for vid in chosen_vids:
            g = meta[meta["video_id"] == vid]
            if g.empty:
                continue
            # tranh 5 frame dau (dao dau) va 5 frame cuoi (outro) - giong SAMPLE_SKIP_INTRO
            usable = g.iloc[5:-5] if len(g) > 15 else g
            if usable.empty:
                usable = g
            row = usable.sample(1, random_state=hash(vid) % (2**31)).iloc[0]
            candidates.append({
                "video_id": vid,
                "local_idx": int(row["local_idx"]),
                "author": author,
            })

    random.shuffle(candidates)
    return candidates[:N_SAMPLES]


def build_sheets(candidates: list[dict]) -> None:
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    n_sheets = (len(candidates) + PER_SHEET - 1) // PER_SHEET
    for sheet_i in range(n_sheets):
        chunk = candidates[sheet_i * PER_SHEET: (sheet_i + 1) * PER_SHEET]
        sheet = Image.new("RGB", (GRID_COLS * CELL_W, GRID_ROWS * (CELL_H + 20)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, cand in enumerate(chunk):
            r, c = divmod(i, GRID_COLS)
            x0, y0 = c * CELL_W, r * (CELL_H + 20)
            try:
                img_bytes = read_keyframe_bytes(cand["video_id"], cand["local_idx"])
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                img.thumbnail((CELL_W, CELL_H))
                paste_x = x0 + (CELL_W - img.width) // 2
                sheet.paste(img, (paste_x, y0))
            except Exception as e:
                draw.text((x0 + 5, y0 + 5), f"LOI: {e}", fill="red", font=font)
            idx_global = sheet_i * PER_SHEET + i
            label = f"#{idx_global} {cand['video_id']} f{cand['local_idx']}"
            draw.rectangle([x0, y0 + CELL_H, x0 + CELL_W, y0 + CELL_H + 20], fill="black")
            draw.text((x0 + 4, y0 + CELL_H + 2), label, fill="white", font=font)
        sheet_path = OUT_DIR / f"sheet_{sheet_i:02d}.png"
        sheet.save(sheet_path)
        print(f"da luu {sheet_path} ({len(chunk)} anh)")


if __name__ == "__main__":
    cands = sample_candidates()
    print(f"da lay mau {len(cands)} candidate")
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(cands, f, ensure_ascii=False, indent=2)
    print(f"manifest -> {MANIFEST_PATH}")
    build_sheets(cands)
