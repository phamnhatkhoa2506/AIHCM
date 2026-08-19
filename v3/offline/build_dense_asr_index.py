"""Build ASR index cho bo dense tu index/asr_text_new.parquet (2026-08-15, ban ASR MOI - nguoi
dung cung cap) - GIAI QUYET gioi han da biet nhieu lan truoc do: ASR cu (index/asr_text.parquet)
map theo local_idx cua bo keyframe BTC, KHONG dung duoc cho bo dense (frame_idx khac han).

Ban MOI co "start"/"end" (GIAY THAT, tu ASR that, khong phai local_idx) - map THANG sang
frame_idx cua bo dense qua fps (index/meta.parquet, video-level, KHONG doi theo mat do
keyframe - dung an toan cho ca BTC lan dense): frame_idx = round(pts_time * fps).

Cot "text_refined" (LLM da sua loi nhan dang + dau cau/hoa dung) - dung lam text CHINH thay vi
"text_raw" (nhieu loi nhan dang tho, vd "trong trọngộng" thay vi "trầm trọng" - da kiem tra
that qua 1 dong mau).

Output: index/dense/asr_text.parquet - schema:
  video_id, frame_idx_start, frame_idx_end, start, end, text_raw (= text_refined goc), text_norm
  (bo dau, dung cho _strip_accents-based search giong OCR/keywords).

Chay: python offline/build_dense_asr_index.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/

import sys
import unicodedata

import pandas as pd

from config import DENSE_DIR, INDEX_DIR

ASR_NEW_PATH = INDEX_DIR / "asr_text_new.parquet"
META_PATH = INDEX_DIR / "meta.parquet"  # video-level fps, dung chung cho ca BTC lan dense
OUT_PATH = DENSE_DIR / "asr_text.parquet"


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d")


def main() -> None:
    if not ASR_NEW_PATH.exists():
        print(f"LOI: khong tim thay {ASR_NEW_PATH}", file=sys.stderr)
        sys.exit(1)

    asr = pd.read_parquet(ASR_NEW_PATH)
    meta = pd.read_parquet(META_PATH)
    fps_by_video = meta.groupby("video_id")["fps"].first()

    missing = set(asr["video_id"]) - set(fps_by_video.index)
    if missing:
        print(f"CANH BAO: {len(missing)} video trong asr_text_new khong co fps trong meta.parquet "
              f"- se BI LOAI khoi output (vd: {list(missing)[:5]})", file=sys.stderr)
    asr = asr[asr["video_id"].isin(fps_by_video.index)].copy()

    fps = asr["video_id"].map(fps_by_video)
    asr["frame_idx_start"] = (asr["start"] * fps).round().astype(int)
    asr["frame_idx_end"] = (asr["end"] * fps).round().astype(int)
    asr["text_raw"] = asr["text_refined"].fillna(asr["text_itn"]).fillna(asr["text_raw"])
    asr["text_norm"] = asr["text_raw"].apply(_strip_accents)

    out = asr[["video_id", "frame_idx_start", "frame_idx_end", "start", "end", "text_raw", "text_norm"]]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"Da luu {len(out)} dong ASR ({out['video_id'].nunique()} video) -> {OUT_PATH}")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
