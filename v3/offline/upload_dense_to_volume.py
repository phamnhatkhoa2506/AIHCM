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

# 2026-08-14: nguoi dung da tu chuan bi san 2 file zip cho 2 thu muc LON NHAT (khoi phai tu
# nen, tiet kiem thoi gian dang ke) - NHUNG cau truc BEN TRONG khac quy uoc script tu tao (co
# 1 lop thu muc boc ngoai KHAC ten muc tieu) -> can strip_levels=1 khi giai nen (xem
# volume_extract_app.py::extract_zip). subfolder -> (duong dan zip co san, strip_levels).
PREMADE_ZIPS: dict[str, str] = {
    "L26_a-b_extracted": r"D:\Programming\AIHCM\data\Our\L26_a-b_extracted.zip",
    "keyframe_output": r"D:\Programming\AIHCM\keyframe\data\output\output.zip",
}


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

    # PIPELINE (2026-08-14, nguoi dung yeu cau): giai nen chay TREN Modal (khong dung tai
    # nguyen local) - trong luc cho giai nen xong, may local co the ZIP+UPLOAD thu muc TIEP
    # THEO song song. Dung extract_fn.spawn() (khong block) thay .remote() (block) - chi CHO
    # ket qua cua lan giai nen TRUOC ngay truoc khi bat dau upload thu muc HIEN TAI (do 1 lan
    # tre - "pending" o duoi).
    pending: tuple | None = None  # (FunctionCall, subfolder, i)

    def _wait_pending(log_f) -> None:
        nonlocal pending
        if pending is None:
            return
        call, sf, idx = pending
        res = call.get()
        print(f"[{idx}] XONG: {sf} ({res['n_files']} anh tren Volume)", file=sys.stderr)
        log_f.write(sf + "\n")
        log_f.flush()
        pending = None

    with open(DONE_LOG, "a", encoding="utf-8") as log_f:
        for i, (local_root, subfolder) in enumerate(items, 1):
            if subfolder in done:
                print(f"[{i}/{len(items)}] BO QUA (da xong): {subfolder}", file=sys.stderr)
                continue
            if not Path(local_root).exists():
                print(f"[{i}/{len(items)}] LOI: khong tim thay {local_root}", file=sys.stderr)
                continue

            premade = PREMADE_ZIPS.get(subfolder)
            if premade:
                zip_path = Path(premade)
                strip_levels = 1
                own_zip = False
                if not zip_path.exists():
                    print(f"[{i}/{len(items)}] LOI: khong tim thay zip co san {zip_path}",
                          file=sys.stderr)
                    sys.exit(1)
                print(f"[{i}/{len(items)}] Dung zip co san: {zip_path}", file=sys.stderr)
            else:
                zip_path = ZIP_STAGING_DIR / f"{subfolder}.zip"
                strip_levels = 0
                own_zip = True
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

            if own_zip:
                zip_path.unlink(missing_ok=True)  # chi xoa zip TU TAO (staging), KHONG dung
                # zip nguoi dung tu chuan bi san o duong dan goc cua ho

            # cho lan giai nen TRUOC xong (thuong da xong tu lau vi upload/zip thu muc nay
            # vua chay song song voi no) TRUOC KHI ghi DONE_LOG cho no.
            _wait_pending(log_f)

            print(f"[{i}/{len(items)}] Gui yeu cau giai nen TREN Modal (khong cho, chay ngam) ...",
                  file=sys.stderr)
            call = extract_fn.spawn(zip_rel_path=volume_zip_rel, dest_subfolder=f"{subfolder}/",
                                     strip_levels=strip_levels)
            pending = (call, subfolder, i)

        _wait_pending(log_f)  # cho lan giai nen CUOI CUNG (khong con thu muc nao de chay song song)

    print("\nDa upload + giai nen xong TAT CA thu muc.", file=sys.stderr)


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    run()
