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

from pathlib import Path, PureWindowsPath

VOLUME_NAME = "aic2026-dense-keyframes"
VOLUME_MOUNT = "/dense_keyframes"  # duong dan mount trong container Modal

# local root2 (parents[2] cua moi anh) -> ten subfolder tren Volume. THU TU: nho -> lon (de
# script upload validate voi thu muc nho truoc, xem offline/upload_dense_to_volume.py).
ROOT_MAP: dict[str, str] = {
    r"D:\Programming\AIHCM\data\Our\L27_extracted": "L27_extracted",       # 9.334 anh
    r"D:\Programming\AIHCM\data\Our\L23_extracted": "L23_extracted",       # 10.743 anh
    r"D:\Programming\AIHCM\data\Our\L24_extracted": "L24_extracted",       # 13.553 anh
    r"D:\Programming\AIHCM\data\Our\L30_extracted": "L30_extracted",       # 15.036 anh
    r"D:\Programming\AIHCM\data\Our\L28_extracted": "L28_extracted",       # 23.833 anh
    r"D:\Programming\AIHCM\data\Our\L29_extracted": "L29_extracted",       # 23.964 anh
    r"D:\Programming\AIHCM\data\Our\L21_extracted": "L21_extracted",       # 28.536 anh
    r"D:\Programming\AIHCM\data\Our\L22_extracted": "L22_extracted",       # 33.881 anh
    r"D:\Programming\AIHCM\data\Our\L25_extracted": "L25_extracted",       # 38.366 anh
    r"D:\Programming\AIHCM\data\Our\L26_a-b_extracted": "L26_a-b_extracted",  # 64.281 anh
    r"D:\Programming\AIHCM\keyframe\data\output\output": "keyframe_output",   # 108.062 anh
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
