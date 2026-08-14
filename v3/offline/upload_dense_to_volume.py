"""Upload TUNG thu muc goc (11 thu muc, xem share/dense_volume_map.py) len Modal Volume
"aic2026-dense-keyframes" - upload TUAN TU tung thu muc theo yeu cau nguoi dung (2026-08-14,
"upload tung thu muc cho do nghen").

QUAN TRONG (2026-08-14, da do thuc te): KHONG dung `modal volume put <thu_muc>` truc tiep -
voi 108k+ file nho/thu muc, toc do chi ~22KB/s (overhead round-trip TUNG file rieng le), 108k
file se mat ~87 GIO. Da kiem chung 1 file 50MB rieng le upload trong ~14s (~3.6MB/s, nhanh gap
~160 lan). Giai phap: NEN tung thu muc thanh 1 file zip -> upload 1 file lon duy nhat (nhanh) ->
goi Modal function giai nen NGAY TREN VOLUME (volume_extract_app.py, khong qua mang local nua).

Idempotent: log rieng (upload_volume_done.log) danh dau thu muc da xong - chay lai script AN
TOAN neu bi ngat giua chung (bo qua thu muc da xong, lam lai thu muc dang do tu dau - zip lai
khong dang ke thoi gian so voi upload).

Chay: python offline/upload_dense_to_volume.py
"""
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "share"))
import modal
from dense_volume_map import ROOT_MAP, VOLUME_NAME

DONE_LOG = Path(__file__).resolve().parent.parent / "index" / "dense" / "upload_volume_done.log"
ZIP_STAGING_DIR = Path(__file__).resolve().parent.parent / "index" / "dense" / "_upload_zips"


def _done_folders() -> set[str]:
    if not DONE_LOG.exists():
        return set()
    return set(DONE_LOG.read_text(encoding="utf-8").splitlines())


def _zip_folder(local_root: str, zip_path: Path) -> int:
    """Nen local_root thanh zip_path - entries la duong dan tuong doi TU BEN TRONG local_root
    (vd "L21\\L21_V009\\xxx.jpg" -> luu la "L21/L21_V009/xxx.jpg" trong zip)."""
    root = Path(local_root)
    n = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        # ZIP_STORED (khong nen) - anh JPEG da nen san, nen lai chi ton CPU khong giam dung
        # luong dang ke, uu tien toc do zip (may local CPU yeu).
        for f in root.rglob("*"):
            if f.is_file():
                zf.write(f, arcname=str(f.relative_to(root)).replace("\\", "/"))
                n += 1
    return n


def run() -> None:
    done = _done_folders()
    items = list(ROOT_MAP.items())
    print(f"Tong {len(items)} thu muc can upload, da xong {len(done)}.", file=sys.stderr)

    ZIP_STAGING_DIR.mkdir(parents=True, exist_ok=True)
    extract_fn = modal.Function.from_name("aic2026-volume-extract", "extract_zip")

    with open(DONE_LOG, "a", encoding="utf-8") as log_f:
        for i, (local_root, subfolder) in enumerate(items, 1):
            if subfolder in done:
                print(f"[{i}/{len(items)}] BO QUA (da xong): {subfolder}", file=sys.stderr)
                continue
            if not Path(local_root).exists():
                print(f"[{i}/{len(items)}] LOI: khong tim thay {local_root}", file=sys.stderr)
                continue

            zip_path = ZIP_STAGING_DIR / f"{subfolder}.zip"
            print(f"[{i}/{len(items)}] Nen {subfolder} <- {local_root} ...", file=sys.stderr)
            n_zipped = _zip_folder(local_root, zip_path)
            zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
            print(f"[{i}/{len(items)}] Da nen {n_zipped} anh, {zip_size_mb:.1f} MB -> {zip_path}",
                  file=sys.stderr)

            volume_zip_rel = f"_zips/{subfolder}.zip"
            print(f"[{i}/{len(items)}] Upload zip len Volume ...", file=sys.stderr)
            result = subprocess.run(
                ["modal", "volume", "put", VOLUME_NAME, str(zip_path), volume_zip_rel, "-f"],
                capture_output=False,
            )
            if result.returncode != 0:
                print(f"[{i}/{len(items)}] LOI upload zip {subfolder} (exit {result.returncode}) "
                      f"- chay lai script se tu resume tu day.", file=sys.stderr)
                sys.exit(1)

            print(f"[{i}/{len(items)}] Giai nen TREN Modal (Volume->Volume, nhanh) ...", file=sys.stderr)
            res = extract_fn.remote(zip_rel_path=volume_zip_rel, dest_subfolder=f"{subfolder}/")
            print(f"[{i}/{len(items)}] XONG: {subfolder} ({res['n_files']} anh tren Volume)",
                  file=sys.stderr)

            zip_path.unlink(missing_ok=True)  # xoa zip local, da co tren Volume roi
            log_f.write(subfolder + "\n")
            log_f.flush()

    print("\nDa upload + giai nen xong TAT CA thu muc.", file=sys.stderr)


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    run()
