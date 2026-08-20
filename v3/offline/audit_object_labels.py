"""Audit chất lượng nhãn của object detector gốc (objects_index.parquet, 514 nhãn OpenImages
V4) — KHÔNG chạy lại detector, chỉ lấy mẫu N crop/nhãn rồi đối chiếu bằng CLIP zero-shot
(ảnh crop vs text "a photo of a {label}") để tìm nhãn nào hay bị GÁN SAI (vd "Dog" gán cho
đầu lân múa lân — phát hiện thật, xem hội thoại 2026-08-06).

Vì sao KHÔNG dùng ngưỡng tuyệt đối để tự động loại nhãn: đã kiểm chứng (xem region_clip
docstring + benchmark/) điểm CLIP ảnh-text bị "score clustering" — frame đúng và frame sai có
thể chỉ cách nhau 0.02-0.03 dù thứ hạng cách xa hàng nghìn bậc. Script này chỉ XẾP HẠNG nhãn
theo điểm trung bình (thấp nhất = nghi ngờ nhiều nhất) để người dùng tự soát bằng mắt qua ảnh
mẫu đã lưu, không tự động kết luận đúng/sai.

Tận dụng lại hạ tầng đã có, KHÔNG tính toán trên CPU local:
  - Object Person/Animal đã có embedding precompute sẵn (region_embeddings.npy) -> tra thẳng.
  - Object nhãn khác (phần lớn) -> crop local (I/O, ThreadPoolExecutor) rồi gửi Modal GPU
    (aic2026-region-clip/Encoder, app đã deploy cho build_region_embeddings.py) để encode,
    KHÔNG encode CPU (đã đo: CPU ~4.5 box/s, quá chậm).
"""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/

import io
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import modal
import numpy as np
import pandas as pd

from config import CLIP_TEXT_MODEL_NAME, INDEX_DIR, MODEL_CACHE_DIR, OBJECTS_INDEX_PATH
from region_clip import crop_region

# 2026-08-20 (theo yeu cau nguoi dung: "dọn dẹp triệt để") - TRUOC DAY dung tiers.tier2_vector.
# encode_query() (pipeline CLIP-32 cu, da xoa) - encode text CHI can 1 SentenceTransformer nhe,
# khong can ca bo may resources.get() (model+faiss+matrix) cua pipeline do. Load truc tiep tai
# day, lazy singleton giong pattern region_clip.py::_get_image_model.
_text_model = None


def encode_query(text: str):
    global _text_model
    if _text_model is None:
        from sentence_transformers import SentenceTransformer

        _text_model = SentenceTransformer(CLIP_TEXT_MODEL_NAME, cache_folder=str(MODEL_CACHE_DIR))
    return _text_model.encode([text], convert_to_numpy=True, normalize_embeddings=True)

N_PER_LABEL = 25
SEED = 42
N_SAMPLE_IMAGES_PER_WORST_LABEL = 5
N_WORST_LABELS_TO_SAVE = 15

EMBEDDINGS_PATH = INDEX_DIR / "region_embeddings.npy"
DETECTION_IDS_PATH = INDEX_DIR / "region_embeddings_detection_ids.npy"
AUDIT_OUT_CSV = INDEX_DIR / "label_audit_scores.csv"
AUDIT_SAMPLES_DIR = INDEX_DIR / "label_audit_samples"


def _load_precomputed() -> tuple[dict[int, int], np.ndarray]:
    if EMBEDDINGS_PATH.exists() and DETECTION_IDS_PATH.exists():
        vecs = np.load(EMBEDDINGS_PATH)
        ids = np.load(DETECTION_IDS_PATH)
        return {int(d): i for i, d in enumerate(ids)}, vecs
    return {}, np.zeros((0, 512), dtype=np.float32)


def _crop_bytes(video_id: str, local_idx: int, box: tuple) -> bytes:
    crop = crop_region(video_id, local_idx, box)
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def main() -> None:
    df = pd.read_parquet(OBJECTS_INDEX_PATH).reset_index(drop=False).rename(columns={"index": "detection_id"})
    print(f"Tong {len(df)} object, {df.label.nunique()} nhan", file=sys.stderr)

    parts = [
        g.sample(n=min(N_PER_LABEL, len(g)), random_state=SEED)
        for _, g in df.groupby("label", group_keys=False, sort=False)
    ]
    sampled = pd.concat(parts, ignore_index=True)
    print(f"Da lay mau {len(sampled)} object ({N_PER_LABEL}/nhan, {sampled.label.nunique()} nhan)", file=sys.stderr)

    id_to_row, precomputed_vecs = _load_precomputed()

    # tach object da co embedding san vs can crop+encode
    need_encode_idx = [i for i, did in enumerate(sampled.detection_id) if did not in id_to_row]
    print(f"{len(sampled) - len(need_encode_idx)} object dung embedding precomputed, "
          f"{len(need_encode_idx)} object can crop+encode qua Modal", file=sys.stderr)

    fresh_vecs_by_pos: dict[int, np.ndarray] = {}
    if need_encode_idx:
        rows = [sampled.iloc[i] for i in need_encode_idx]
        crops_bytes: list[bytes | None] = [None] * len(rows)
        with ThreadPoolExecutor(max_workers=32) as pool:
            futures = {
                pool.submit(_crop_bytes, r.video_id, int(r.local_idx), (r.ymin, r.xmin, r.ymax, r.xmax)): i
                for i, r in enumerate(rows)
            }
            done = 0
            for fut in as_completed(futures):
                crops_bytes[futures[fut]] = fut.result()
                done += 1
                if done % 2000 == 0 or done == len(rows):
                    print(f"[crop {done}/{len(rows)}]", file=sys.stderr)

        Encoder = modal.Cls.from_name("aic2026-region-clip", "Encoder")
        encoder = Encoder()
        batch_size = 64
        batches = [crops_bytes[i : i + batch_size] for i in range(0, len(crops_bytes), batch_size)]
        print(f"Goi Modal encode: {len(batches)} batch...", file=sys.stderr)
        all_vecs: list[list[float]] = []
        for bi, vecs in enumerate(encoder.encode_batch.map(batches), 1):
            all_vecs.extend(vecs)
            if bi % 20 == 0 or bi == len(batches):
                print(f"  [Modal batch {bi}/{len(batches)}] -> {len(all_vecs)}/{len(rows)}", file=sys.stderr)
        for pos, vec in zip(need_encode_idx, all_vecs):
            fresh_vecs_by_pos[pos] = np.array(vec, dtype=np.float32)

    # gop vector theo dung thu tu sampled
    region_vecs = np.zeros((len(sampled), 512), dtype=np.float32)
    for i, did in enumerate(sampled.detection_id):
        if did in id_to_row:
            region_vecs[i] = precomputed_vecs[id_to_row[did]]
        else:
            region_vecs[i] = fresh_vecs_by_pos[i]

    # encode text 1 lan / nhan (khong lap lai cho tung crop cung nhan)
    labels = sorted(sampled.label.unique())
    print(f"Encode text cho {len(labels)} nhan...", file=sys.stderr)
    label_text_vec = {lb: encode_query(f"a photo of a {lb.lower()}")[0] for lb in labels}

    sims = np.array([
        float(region_vecs[i] @ label_text_vec[lb])
        for i, lb in enumerate(sampled.label)
    ])
    sampled = sampled.copy()
    sampled["clip_sim"] = sims

    agg = sampled.groupby("label")["clip_sim"].agg(["mean", "std", "count"]).sort_values("mean")
    agg.to_csv(AUDIT_OUT_CSV, encoding="utf-8-sig")
    print(f"\nDa luu bang day du -> {AUDIT_OUT_CSV}", file=sys.stderr)

    print("\n" + "=" * 70)
    print(f"{'label':<25} {'mean_sim':>10} {'std':>8} {'n':>5}")
    print("=" * 70)
    for label, row in agg.head(30).iterrows():
        print(f"{label:<25} {row['mean']:>10.4f} {row['std']:>8.4f} {int(row['count']):>5}")

    # luu vai anh mau cho cac nhan te nhat de soat bang mat
    AUDIT_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    worst_labels = agg.head(N_WORST_LABELS_TO_SAVE).index.tolist()
    print(f"\nDang luu anh mau cho {len(worst_labels)} nhan te nhat -> {AUDIT_SAMPLES_DIR}", file=sys.stderr)
    for label in worst_labels:
        sub = sampled[sampled.label == label].sort_values("clip_sim").head(N_SAMPLE_IMAGES_PER_WORST_LABEL)
        safe_label = label.replace("/", "_").replace(" ", "_")
        for j, r in enumerate(sub.itertuples(index=False)):
            crop = crop_region(r.video_id, int(r.local_idx), (r.ymin, r.xmin, r.ymax, r.xmax))
            fname = AUDIT_SAMPLES_DIR / f"{safe_label}__{j}__{r.video_id}_{r.local_idx}__sim{r.clip_sim:.3f}.jpg"
            crop.save(fname, format="JPEG", quality=90)

    print("Xong.", file=sys.stderr)


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    main()
