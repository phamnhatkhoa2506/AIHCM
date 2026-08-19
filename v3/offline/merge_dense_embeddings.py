"""Gop tat ca embedding SigLIP2/PE-Core/BEiT-3 da tinh rieng le (build_dense_embeddings.py,
chay cho L21-L30 tu D:\\...\\data\\Our\\embeddings\\<bo>\\ VA L26_c/d/e tu
D:\\...\\keyframe\\data\\output\\embeddings\\) thanh 1 bo THONG NHAT cho moi model:
  index/dense/<model>_matrix.npy   - (N, dim) float32, DA L2-normalize
  index/dense/<model>_faiss.index  - IndexFlatIP tren matrix tren
  index/dense/dense_meta.parquet   - CHUNG cho ca 3 model (thu tu dong khop voi row cua matrix,
                                      vi ca 3 model deu encode CUNG 1 danh sach anh - xem
                                      _scan() trong build_dense_embeddings.py, thu tu deterministic)

meta schema: video_id, shot_idx, frame_idx, path
  frame_idx = frame_idx tren truc VIDEO GOC (parse thang tu ten file shotNNNN_fFFFFFFFF.jpg)
  -> DUNG DUOC LUON lam frame_id nop bai (xem PDF muc 3 - "vi tri keyframe duoc ghi trong file
  metadata" tren truc video goc, khong phu thuoc keyframe BTC chon).

QUAN TRONG: dense_meta phai THONG NHAT thu tu giua 3 nguon (L21-L30 rieng le + keyframe L26_c/d/e)
-> gop THEO DUNG THU TU da chay (L21..L30 roi keyframe/L26cde), kiem tra so dong matrix khop
voi manifest truoc khi ghep, khong tin suong.

Chay: python offline/merge_dense_embeddings.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/

import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from config import INDEX_DIR

OUR_EMB_ROOT = Path(r"D:\Programming\AIHCM\data\Our\embeddings")
KEYFRAME_EMB_ROOT = Path(r"D:\Programming\AIHCM\keyframe\data\output\embeddings")

OUR_FOLDERS = ["L21", "L22", "L23", "L24", "L25", "L26", "L27", "L28", "L29", "L30"]
MODELS = ["siglip", "pe_core", "beit3"]

DENSE_DIR = INDEX_DIR / "dense"


def _load_source(emb_dir: Path, model: str) -> tuple[np.ndarray, pd.DataFrame]:
    matrix = np.load(emb_dir / f"{model}_matrix.npy")
    manifest = pd.read_parquet(emb_dir / f"{model}_manifest.parquet")
    if len(matrix) != len(manifest):
        raise ValueError(f"LECH: {emb_dir}/{model} matrix={len(matrix)} dong nhung manifest={len(manifest)} dong")
    return matrix, manifest


def main() -> None:
    DENSE_DIR.mkdir(parents=True, exist_ok=True)

    sources = [(OUR_EMB_ROOT / f, f) for f in OUR_FOLDERS] + [(KEYFRAME_EMB_ROOT, "keyframe_L26cde")]

    for model in MODELS:
        print(f"\n=== gop {model} ===", file=sys.stderr)
        matrices, manifests = [], []
        for emb_dir, tag in sources:
            if not (emb_dir / f"{model}_matrix.npy").exists():
                print(f"  [BO QUA] {tag}: khong co {model}_matrix.npy", file=sys.stderr)
                continue
            m, mf = _load_source(emb_dir, model)
            matrices.append(m)
            manifests.append(mf)
            print(f"  {tag}: {len(mf)} anh", file=sys.stderr)

        full_matrix = np.concatenate(matrices, axis=0).astype(np.float32)
        full_meta = pd.concat(manifests, axis=0, ignore_index=True)
        assert len(full_matrix) == len(full_meta), "matrix/meta lech dong sau gop"

        # L2-normalize (embeddings tra ve tu Modal apps DA normalize san luc encode, nhung
        # normalize lai o day cho CHAC - re, khong hai gi neu da normalize).
        norms = np.linalg.norm(full_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        full_matrix = full_matrix / norms

        np.save(DENSE_DIR / f"{model}_matrix.npy", full_matrix)
        index = faiss.IndexFlatIP(full_matrix.shape[1])
        index.add(full_matrix)
        faiss.write_index(index, str(DENSE_DIR / f"{model}_faiss.index"))
        print(f"  TONG: {full_matrix.shape} -> {DENSE_DIR / f'{model}_matrix.npy'}", file=sys.stderr)

        if model == MODELS[0]:
            full_meta.to_parquet(DENSE_DIR / "dense_meta.parquet", index=False)
            print(f"  meta: {len(full_meta)} dong -> {DENSE_DIR / 'dense_meta.parquet'}", file=sys.stderr)
        else:
            # XAC MINH thu tu giong het model dau (cung nguon, cung _scan() deterministic) -
            # KHONG tin suong, so sanh video_id+frame_idx tung dong.
            ref = pd.read_parquet(DENSE_DIR / "dense_meta.parquet")
            if not (ref["video_id"].values == full_meta["video_id"].values).all() or \
               not (ref["frame_idx"].values == full_meta["frame_idx"].values).all():
                raise ValueError(
                    f"THU TU {model} KHONG KHOP voi meta da luu tu {MODELS[0]} - "
                    "khong the dung chung 1 dense_meta.parquet cho ca 3 model!"
                )
            print(f"  thu tu khop voi {MODELS[0]} - dung chung dense_meta.parquet", file=sys.stderr)

    print(f"\nXONG. {len(pd.read_parquet(DENSE_DIR / 'dense_meta.parquet'))} anh tong cong.", file=sys.stderr)


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
