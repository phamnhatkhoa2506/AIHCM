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
def extract_zip(zip_rel_path: str, dest_subfolder: str) -> dict:
    """zip_rel_path: duong dan zip tren Volume (vd "_zips/L21_extracted.zip"), entries ben
    trong zip la duong dan tuong doi TU BEN TRONG thu muc goc (vd "L21/L21_V009/xxx.jpg").
    Giai nen vao MOUNT/dest_subfolder/ - ket qua khop dung quy uoc share/dense_volume_map.py."""
    import zipfile
    from pathlib import Path

    zip_path = Path(MOUNT) / zip_rel_path
    dest = Path(MOUNT) / dest_subfolder
    dest.mkdir(parents=True, exist_ok=True)

    n_files = 0
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
        n_files = len(zf.namelist())

    zip_path.unlink()  # xoa zip staging sau khi giai nen xong, tiet kiem dung luong Volume
    dense_kf_vol.commit()
    return {"dest_subfolder": dest_subfolder, "n_files": n_files}
