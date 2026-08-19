"""Anh xa duong dan LOCAL (dense_meta.parquet cot "path") sang duong dan tuong doi trong Modal
Volume "aic2026-dense-keyframes" - dung chung giua script upload (offline/upload_dense_to_volume.py)
va Modal app (owlv2_app.py) de ca 2 phia hieu CUNG mot quy uoc duong dan.

Boi canh (2026-08-14): da chan doan nghen co chai la LOCAL CPU (doc/giai nen + gui bytes anh
qua mang cho Modal) - khong phai Modal GPU container. Giai phap: upload anh len Modal Volume 1
lan, sau do container GPU tu doc thang tu Volume (khong qua CPU/mang local nua) - loai bo hoan
toan nghen co chai nay cho MOI lan chay sau (khong chi OWLv2, ca OCR/DINO/embeddings dense sau
nay neu can).

11 thu muc goc (root2 = parents[2] cua path) chua toan bo 369.589 anh dense - moi thu muc anh
xa sang 1 subfolder co ten ro rang tren Volume (khong dung nguyen ten "output" trung lap)."""
from __future__ import annotations

import sys
from pathlib import Path, PureWindowsPath

_share_dir = str(Path(__file__).resolve().parent)
if _share_dir not in sys.path:
    sys.path.insert(0, _share_dir)  # phong than neu module nay duoc import truc tiep (khong
    # qua app.py, vd tu offline/*.py) - dam bao "from config import ..." duoi day chay duoc.
from config import KEYFRAME_OUTPUT_ROOT, OUR_DATA_ROOT

VOLUME_NAME = "aic2026-dense-keyframes"
VOLUME_MOUNT = "/dense_keyframes"  # duong dan mount trong container Modal

# local root2 (parents[2] cua moi anh) -> ten subfolder tren Volume. THU TU: nho -> lon (de
# script upload validate voi thu muc nho truoc, xem offline/upload_dense_to_volume.py).
# 2026-08-19: goc duong dan (OUR_DATA_ROOT/KEYFRAME_OUTPUT_ROOT) chuyen vao config.py, KHONG
# con hardcode "D:\..." o day - doi may/o dia chi sua config.py, khong can sua file nay.
ROOT_MAP: dict[str, str] = {
    str(OUR_DATA_ROOT / "L27_extracted"): "L27_extracted",       # 9.334 anh
    str(OUR_DATA_ROOT / "L23_extracted"): "L23_extracted",       # 10.743 anh
    str(OUR_DATA_ROOT / "L24_extracted"): "L24_extracted",       # 13.553 anh
    str(OUR_DATA_ROOT / "L30_extracted"): "L30_extracted",       # 15.036 anh
    str(OUR_DATA_ROOT / "L28_extracted"): "L28_extracted",       # 23.833 anh
    str(OUR_DATA_ROOT / "L29_extracted"): "L29_extracted",       # 23.964 anh
    str(OUR_DATA_ROOT / "L21_extracted"): "L21_extracted",       # 28.536 anh
    str(OUR_DATA_ROOT / "L22_extracted"): "L22_extracted",       # 33.881 anh
    str(OUR_DATA_ROOT / "L25_extracted"): "L25_extracted",       # 38.366 anh
    str(OUR_DATA_ROOT / "L26_a-b_extracted"): "L26_a-b_extracted",  # 64.281 anh
    str(KEYFRAME_OUTPUT_ROOT / "output"): "keyframe_output",        # 108.062 anh
}


def local_root2(path: str) -> str:
    """Tra ve root2 (parents[2]) cua 1 duong dan local - PHAI khop 1 trong cac key ROOT_MAP."""
    p = Path(path)
    parts = p.parts
    if len(parts) > 3:
        return str(Path(*parts[:-3]))
    return str(p.parent)


def to_volume_rel_path(local_path: str) -> str:
    """local path (Windows, vd D:\\...\\L21_extracted\\L21\\L21_V009\\shot0214_f0020910.jpg)
    -> duong dan tuong doi tren Volume (POSIX, vd L21_extracted/L21/L21_V009/shot0214_f0020910.jpg)."""
    root = local_root2(local_path)
    subfolder = ROOT_MAP.get(root)
    if subfolder is None:
        raise ValueError(f"Khong tim thay root2 '{root}' trong ROOT_MAP - can bo sung.")
    rel = PureWindowsPath(local_path).relative_to(PureWindowsPath(root))
    return f"{subfolder}/" + "/".join(rel.parts)
