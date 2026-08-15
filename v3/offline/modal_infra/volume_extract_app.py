"""App phu tro CHI de giai nen file zip da upload len Volume "aic2026-dense-keyframes" NGAY
TREN MODAL (khong qua mang local) - xem hoi thoai 2026-08-14: upload 108k file nho rieng le qua
`modal volume put` cuc cham (~22KB/s, overhead round-trip TUNG file) vi 1 file 50MB upload
nhanh gap ~160 lan (~3.6MB/s). Giai phap: nen tung thu muc thanh 1 zip -> upload 1 file lon
(nhanh) -> giai nen O DAY (giua Volume voi Volume, khong qua mang local nua, cuc nhanh).

Deploy: modal deploy offline/modal_infra/volume_extract_app.py
"""
import modal

app = modal.App("aic2026-volume-extract")
dense_kf_vol = modal.Volume.from_name("aic2026-dense-keyframes", create_if_missing=True)
MOUNT = "/dense_keyframes"

image = modal.Image.debian_slim(python_version="3.11")


@app.function(image=image, volumes={MOUNT: dense_kf_vol}, timeout=1800, cpu=2)
def extract_zip(zip_rel_path: str, dest_subfolder: str, strip_levels: int = 0) -> dict:
    """zip_rel_path: duong dan zip tren Volume (vd "_zips/L21_extracted.zip").
    strip_levels=0 (mac dinh, zip tu offline/upload_dense_to_volume.py tu tao): entries la
    duong dan tuong doi TU BEN TRONG thu muc goc (vd "L21/L21_V009/xxx.jpg") - giai nen
    truc tiep vao MOUNT/dest_subfolder/.
    strip_levels=1 (zip nguoi dung tu chuan bi san, co 1 lop thu muc boc ngoai khac ten -
    2026-08-14, vd "L26_a-b_extracted/L26/..." hoac "output/keyframes/...") - BO cap thu muc
    dau tien cua moi entry truoc khi ghi vao MOUNT/dest_subfolder/."""
    import zipfile
    from pathlib import Path, PurePosixPath

    zip_path = Path(MOUNT) / zip_rel_path
    dest = Path(MOUNT) / dest_subfolder
    dest.mkdir(parents=True, exist_ok=True)

    n_files = 0
    with zipfile.ZipFile(zip_path) as zf:
        if strip_levels == 0:
            zf.extractall(dest)
            n_files = len(zf.namelist())
        else:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue  # entry thu muc, bo qua (mkdir tu dong khi ghi file)
                parts = PurePosixPath(name).parts
                if len(parts) <= strip_levels:
                    continue  # entry chinh la thu muc boc ngoai, khong con gi sau khi strip
                rel = PurePosixPath(*parts[strip_levels:])
                out_path = dest / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
                n_files += 1

    zip_path.unlink()  # xoa zip staging sau khi giai nen xong, tiet kiem dung luong Volume
    dense_kf_vol.commit()
    return {"dest_subfolder": dest_subfolder, "n_files": n_files}
