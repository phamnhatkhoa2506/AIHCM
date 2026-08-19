"""Driver: upload video 3 bo L26_c/d/e (chua ai chay, xem hoi thoai 2026-08-11) vao Modal
Volume, goi dense_keyframe_app.run_folder() tren GPU cho tung bo, tai data/output/ ve local
merge chung voi output cac bo khac ban da chay san (D:\\Programming\\AIHCM\\keyframe\\data\\output\\).

Chay: python offline/run_dense_keyframes_l26_cde.py
      python offline/run_dense_keyframes_l26_cde.py --only Videos_L26_c   # 1 bo de test truoc
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/
from config import DATA_ROOT, KEYFRAME_OUTPUT_ROOT  # noqa: E402

import modal

LOCAL_OUTPUT_DIR = KEYFRAME_OUTPUT_ROOT

TARGET_ZIPS = {
    "Videos_L26_c": DATA_ROOT / "Videos_L26_c.zip",
    "Videos_L26_d": DATA_ROOT / "Videos_L26_d.zip",
    "Videos_L26_e": DATA_ROOT / "Videos_L26_e.zip",
}


def upload_zip_to_volume(folder_name: str, zip_path: Path) -> None:
    """Doc tung video trong zip, ghi tam ra file, upload vao Volume qua batch_upload - tranh
    giai nen ~7GB/zip ra dia local truoc (khong can thiet, chi ton dia/thoi gian)."""
    # BUG THAT (2026-08-11): batch.put_file() KHONG upload ngay - no lazy, chi thuc su doc
    # file luc __aexit__ cua "with vol.batch_upload()". Xoa file tam TRONG vong lap (truoc khi
    # block with dong) -> FileNotFoundError vi luc do file da bi xoa roi. SUA: giu het file tam
    # toi khi block with dong xong moi xoa.
    vol = modal.Volume.from_name("aic2026-dense-keyframe-data", create_if_missing=True)
    tmp_paths: list[str] = []
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".mp4")]
        print(f"[{folder_name}] {len(names)} video trong {zip_path.name}, dang upload...", file=sys.stderr)
        try:
            with vol.batch_upload(force=True) as batch:
                for i, name in enumerate(names, 1):
                    vid_filename = Path(name).name  # vd L26_V200.mp4
                    data = z.read(name)
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                        tmp.write(data)
                        tmp_paths.append(tmp.name)
                    # Volume mount o /root/keyframe/data (xem dense_keyframe_app.py) - path
                    # BEN TRONG volume KHONG duoc co tien to "data/" nua, khong thi thanh
                    # /root/keyframe/data/data/... (BUG THAT 2026-08-11, da gap luc smoke test).
                    batch.put_file(tmp.name, f"video/{folder_name}/{vid_filename}")
                    if i % 20 == 0 or i == len(names):
                        print(f"[{folder_name}]   {i}/{len(names)} da xep hang upload", file=sys.stderr)
        finally:
            for p in tmp_paths:
                Path(p).unlink(missing_ok=True)
    print(f"[{folder_name}] upload xong.", file=sys.stderr)


def download_output_from_volume() -> None:
    """Tai toan bo data/output/ tu Volume ve dung vi tri local repo keyframe/ de gop chung
    voi output cac bo da chay truoc do (group_keyframes_by_collection.py/package_btc_format.py
    chay sau se thay du lieu day du)."""
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Dang tai data/output/ ve {LOCAL_OUTPUT_DIR} ...", file=sys.stderr)
    subprocess.run(
        ["modal", "volume", "get", "aic2026-dense-keyframe-data", "output",
         str(LOCAL_OUTPUT_DIR), "--force"],
        check=True,
    )
    print("Tai xong.", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="vd Videos_L26_c - chi chay 1 bo de test truoc")
    ap.add_argument("--skip-upload", action="store_true", help="da upload roi, chi chay + tai ve")
    args = ap.parse_args()

    targets = {args.only: TARGET_ZIPS[args.only]} if args.only else TARGET_ZIPS

    if not args.skip_upload:
        for folder_name, zip_path in targets.items():
            upload_zip_to_volume(folder_name, zip_path)

    # 3 bo doc lap hoan toan (thu muc rieng) - goi song song qua starmap thay vi remote()
    # tuan tu, moi bo 1 container GPU rieng -> ~47 phut thay vi ~2.3 gio (do smoke test 2
    # video ~28s/video tren T4, ~100 video/bo).
    run_folder = modal.Function.from_name("aic2026-dense-keyframe", "run_folder")
    print(f"dang chay song song {len(targets)} bo tren Modal GPU...", file=sys.stderr)
    for folder_name, log in zip(targets, run_folder.map(list(targets), order_outputs=True)):
        print(f"[{folder_name}] XONG:\n{log[-1500:]}", file=sys.stderr)

    download_output_from_volume()


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
