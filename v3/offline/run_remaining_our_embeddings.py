"""Chay embedding (siglip, pe_core, beit3) cho cac bo con lai trong D:\\...\\data\\Our\\
(L22-L30, L21 da chay rieng truoc do) - lap qua tung bo, tung model, goi lai
build_dense_embeddings.py::run() truc tiep (khong spawn subprocess) de log gon 1 cho.

Chay: python offline/run_remaining_our_embeddings.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dense_embeddings import run  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/
from config import OUR_DATA_ROOT as OUR_ROOT  # noqa: E402

# (ten thu muc _extracted, ten thu muc con ben trong) - da kiem tra thuc te (2026-08-13)
FOLDERS = [
    ("L22_extracted", "L22"),
    ("L23_extracted", "L23"),
    ("L24_extracted", "L24"),
    ("L25_extracted", "L25"),
    ("L26_a-b_extracted", "L26"),
    ("L27_extracted", "L27"),
    ("L28_extracted", "L28"),
    ("L29_extracted", "L29"),
    ("L30_extracted", "L30"),
]
MODELS = ["siglip", "pe_core", "beit3"]

if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    for extracted_dir, subdir in FOLDERS:
        src = OUR_ROOT / extracted_dir / subdir
        out_dir = OUR_ROOT / "embeddings" / subdir  # rieng tung bo - tranh de len siglip_matrix.npy chung
        for model in MODELS:
            print(f"\n===== {subdir} / {model} =====", file=sys.stderr)
            run(model, src, out_dir)
