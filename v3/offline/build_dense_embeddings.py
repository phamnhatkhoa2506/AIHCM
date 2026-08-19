"""Encode toan bo keyframe TU TRICH (repo rieng D:\\Programming\\AIHCM\\keyframe, ban ban
da chay local cho L21, giai nen tu L21.rar) bang SigLIP2 va/hoac PE-Core qua Modal (2 app da
co san: siglip_app.py, pe_core_app.py) - dung chung interface encode_images(images_bytes).

Anh khong nam trong Keyframes_*.zip cua BTC nen CLIP feature BTC cap KHONG dung duoc (xem
hoi thoai 2026-08-11, phan "Tang mat do keyframe") - phai encode lai tu dau bang model moi.

Ten file anh dang shotNNNN_fFFFFFFFF.jpg (NNNN=shot index, FFFFFFFF=frame_idx tren truc video
goc) - parse thang tu ten file, khong can doc lai keyframes.jsonl.

Chay: python offline/build_dense_embeddings.py --model siglip --src "D:\...\L21_extracted\L21"
      python offline/build_dense_embeddings.py --model pe_core --src "D:\...\L21_extracted\L21"
"""
from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/
from config import OUR_DATA_ROOT  # noqa: E402

BATCH_SIZE = 32
N_READ_WORKERS = 16

MODEL_APP = {
    "siglip": ("aic2026-siglip", "SigLIPEncoder"),
    "pe_core": ("aic2026-pe-core", "PECoreEncoder"),
    "beit3": ("aic2026-beit3", "BEiT3Encoder"),
}

_NAME_RE = re.compile(r"shot(\d+)_f(\d+)\.jpg$", re.IGNORECASE)


def _parse_filename(fp: Path) -> tuple[int, int] | None:
    m = _NAME_RE.search(fp.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _scan(src: Path) -> pd.DataFrame:
    rows = []
    for video_dir in sorted(src.iterdir()):
        if not video_dir.is_dir():
            continue
        for fp in sorted(video_dir.glob("*.jpg")):
            parsed = _parse_filename(fp)
            if parsed is None:
                print(f"  bo qua ten file khong khop pattern: {fp}", file=sys.stderr)
                continue
            shot_idx, frame_idx = parsed
            rows.append({
                "video_id": video_dir.name,
                "shot_idx": shot_idx,
                "frame_idx": frame_idx,
                "path": str(fp),
            })
    return pd.DataFrame(rows)


def run(model: str, src: Path, out_dir: Path, limit: int | None = None) -> None:
    app_name, cls_name = MODEL_APP[model]
    Encoder = modal.Cls.from_name(app_name, cls_name)
    encoder = Encoder()

    manifest = _scan(src)
    if limit:
        manifest = manifest.iloc[:limit].reset_index(drop=True)
    n = len(manifest)
    print(f"[{model}] {n} anh tu {manifest['video_id'].nunique()} video", file=sys.stderr)

    read_pool = ThreadPoolExecutor(max_workers=N_READ_WORKERS)
    paths = manifest["path"].tolist()
    batches = [paths[i:i + BATCH_SIZE] for i in range(0, n, BATCH_SIZE)]

    def _read_batch(batch_paths: list[str]) -> list[bytes]:
        return [Path(p).read_bytes() for p in batch_paths]

    # BUG THAT (2026-08-13): doc HET tat ca ~100k+ anh vao RAM truoc khi goi .map() -> MemoryError
    # khi chay NHIEU job song song (siglip+pe_core+beit3 cung luc, moi job tu doc rieng het
    # corpus). SUA: generator LAZY qua .map() cua Modal (no tu iterate qua tung phan tu cua
    # all_batches_bytes) + doc tung batch bang generator thay vi list() ep het vao bo nho
    # ngay - giu bo nho O(so luong doc truoc, khong phai O(toan bo corpus)).
    def _batches_gen():
        for b in batches:
            yield read_pool.submit(_read_batch, b)

    # prefetch nong: giu toi da (N_READ_WORKERS+2) future cung luc, khong nap het 1 luc.
    futures_iter = iter(_batches_gen())
    pending: list = []
    for _ in range(min(N_READ_WORKERS + 2, len(batches))):
        try:
            pending.append(next(futures_iter))
        except StopIteration:
            break

    def _all_batches_bytes():
        nonlocal pending
        while pending:
            fut = pending.pop(0)
            try:
                pending.append(next(futures_iter))
            except StopIteration:
                pass
            yield fut.result()

    print(f"[{model}] dang doc anh (song song {N_READ_WORKERS} luong, prefetch nho)...", file=sys.stderr)
    all_vecs: list[list[float]] = []
    done = 0
    for vecs in encoder.encode_images.map(_all_batches_bytes(), order_outputs=True):
        all_vecs.extend(vecs)
        done += len(vecs)
        print(f"[{model}] [{done}/{n}]", file=sys.stderr)
    read_pool.shutdown(wait=False)

    matrix = np.array(all_vecs, dtype=np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{model}_matrix.npy", matrix)
    manifest.to_parquet(out_dir / f"{model}_manifest.parquet", index=False)
    print(f"[{model}] xong: matrix {matrix.shape} -> {out_dir / f'{model}_matrix.npy'}", file=sys.stderr)


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_APP))
    ap.add_argument("--src", required=True, help="thu muc chua <video_id>/*.jpg")
    ap.add_argument("--out", default=str(OUR_DATA_ROOT / "embeddings"))
    ap.add_argument("--limit", type=int, default=None, help="chi encode N anh dau (test nhanh)")
    args = ap.parse_args()
    run(args.model, Path(args.src), Path(args.out), limit=args.limit)
