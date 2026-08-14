"""Giai đoạn 3: đọc frame đã bị VLM gắn cờ (vocab_discovery.py), gọi Grounding DINO
(`modal_infra/grounding_dino_app.py`) định vị đúng vùng vật thể mới, lưu thành detection MỚI
(KHÔNG ghi đè objects_index.parquet gốc — xem lý do trong thiết kế: giữ cả nhãn cũ SAI để soát
lại sau, gắn cờ nguồn gốc `source` để phân biệt).

TỐI ƯU: gom các frame CÙNG tập phrase (cùng prompt) thành 1 lượt gọi Modal (model batch nhiều
ảnh 1 lần) thay vì gọi riêng từng frame — nhiều frame cùng video hay lặp lại đúng phrase (vd
"news studio graphic" ở rất nhiều frame bản tin), gom nhóm tiết kiệm được nhiều lượt forward.
"""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/

import io
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import modal
import pandas as pd

from atomic_io import atomic_write_parquet
from PIL import Image

from config import INDEX_DIR
from keyframe_images import read_keyframe_bytes
from vocab_discovery import _sampled_frames

FLAGGED_FRAMES_PATH = INDEX_DIR / "vocab_discovery_flagged_frames.jsonl"
OUT_PATH = INDEX_DIR / "open_vocab_detections.parquet"
DONE_PATH = INDEX_DIR / "open_vocab_dino_done.jsonl"  # resume - track (video_id,local_idx) DA
# xu ly (ke ca khi khong co detection nao - phai track rieng, khong the dua vao co dong trong
# OUT_PATH hay khong vi frame 0-detection va frame CHUA chay trong giong het nhau trong parquet)

THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20

# GIOI HAN batch/luot goi Modal - BUG THAT (2026-08-06): nhom prompt trung nhau co the co
# HANG NGAN anh (vd "news studio graphic" lap lai rat nhieu video), don het vao 1 lan forward
# -> CUDA OOM tren A10G (22GB), sap toan bo tien trinh, MAT HET progress vi truoc do chi ghi
# parquet 1 lan luc cuoi. Gio: (1) cat nho nhom lon thanh nhieu luot <= MAX_IMAGES_PER_CALL,
# (2) try/except quanh moi luot goi - 1 luot loi khong sap ca job, (3) checkpoint parquet
# dinh ky thay vi doi den cuoi.
MAX_IMAGES_PER_CALL = 8
CHECKPOINT_EVERY_GROUPS = 20

# Sau thay doi chien thuat (2026-08-06): flagged_frames.jsonl gom ca frame tu lan chay FULL
# CORPUS ban dau (exhaustive, ~10k frame dau, moi (video_id,local_idx) co the co) LAN frame
# tu lan chay SAMPLE THUA sau nay (chi cac vi tri dung theo _sampled_frames()). Theo yeu cau
# nguoi dung: frame thuoc doan EXHAUSTIVE cu -> dung HET (khong loc them). Frame thuoc doan
# SAMPLE moi -> loc THEM 1 lan nua.
#
# CACH LOC (v2, 2026-08-06): ban dau dung stride tho "1/10 theo thu tu chung ca video" - phat
# hien van de: neu 1 vat the (vd "con lan") xuat hien lien tuc qua nhieu frame gan nhau, stride
# co the xoa gan het cac lan xuat hien do vi no khong biet frame nao thuoc CUNG 1 lan xuat hien.
# Sua: gom theo (video_id, tap phrase giong nhau) TRUOC, trong tung nhom tim cac "doan lien tuc"
# (local_idx cach nhau <= RUN_GAP_THRESHOLD -> cung 1 lan vat the xuat hien, vi cac frame sample
# von da cach nhau dung SAMPLE_STRIDE=15 khi lien tiep trong sample), lay 1 dai dien/doan (frame
# giua doan) - dam bao MOI lan vat the xuat hien deu co it nhat 1 dai dien, khong bi stride tho
# "an nham" ca cum. Chi la group/sort tren du lieu co san, KHONG goi model nao -> khong doi thoi
# gian chay (deterministic, O(n log n)).
RUN_GAP_THRESHOLD = 20  # hoi lon hon SAMPLE_STRIDE (15) - 2 frame sample lien tiep = cung 1 doan

# GHI LAI (chua lam, chi lam khi can do chinh xac hon): cach 2 - phan cum theo DO TUONG DONG
# ANH THAT (crop+encode CLIP roi cosine similarity giua cac frame ung vien) thay vi chi dua vao
# vi tri thoi gian nhu tren. Chinh xac hon (bat duoc vat the xuat hien lai sau khoang cach xa,
# hoac phan biet 2 vat the khac nhau dung gan nhau ve thoi gian) nhung TON THEM CHI PHI: cac
# object open-vocab nay KHONG nam trong scope Person/Animal da co embedding precompute
# (build_region_embeddings.py) nen phai tinh moi hoan toan - uoc ~10-30 phut cho 17-26k frame
# qua Modal GPU (region_clip_app.py da co san, tai su dung duoc).


def _build_prompt(phrases: list[str]) -> str:
    """Grounding DINO: cum tu cach nhau boi dau cham, lowercase - chuan hoa + sort de nhom
    frame CUNG tap phrase (khac thu tu) vao chung 1 prompt, tang co hoi batch."""
    norm = sorted({p.strip().lower() for p in phrases if p.strip()})
    return ". ".join(norm) + "."


def _load_flagged() -> list[dict]:
    if not FLAGGED_FRAMES_PATH.exists():
        print(f"Khong thay {FLAGGED_FRAMES_PATH} - chay vocab_discovery.py truoc.", file=sys.stderr)
        return []
    recs = []
    with open(FLAGGED_FRAMES_PATH, encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    return recs


def _split_and_thin(flagged: list[dict]) -> list[dict]:
    """Tach flagged thanh 2 nhom theo (video_id, local_idx) co nam trong tap sample thua hien
    tai khong. Ngoai tap sample -> chi co the den tu lan chay EXHAUSTIVE cu -> giu HET. Trong
    tap sample -> co the den tu lan chay SAMPLE moi -> gom theo doan lien tuc (xem RUN_GAP_THRESHOLD),
    lay 1 dai dien/doan."""
    _sf = _sampled_frames()
    sample_scope = set(zip(_sf["video_id"], _sf["local_idx"].astype(int)))

    exhaustive_group = [r for r in flagged if (r["video_id"], r["local_idx"]) not in sample_scope]
    sampled_group = [r for r in flagged if (r["video_id"], r["local_idx"]) in sample_scope]

    # gom theo (video_id, tap phrase giong nhau) - chi coi la "cung 1 lan vat the xuat hien"
    # neu CA video LAN loai vat the deu trung, khong gop nham 2 vat the khac nhau dung gan nhau.
    by_video_phrase: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in sampled_group:
        key = (r["video_id"], _build_prompt(r["phrases"]))
        by_video_phrase[key].append(r)

    thinned_sampled = []
    n_runs = 0
    for (vid, _phrase_key), recs in by_video_phrase.items():
        recs.sort(key=lambda r: r["local_idx"])
        # tach thanh cac doan lien tuc (gap <= RUN_GAP_THRESHOLD)
        run: list[dict] = [recs[0]]
        for r in recs[1:]:
            if r["local_idx"] - run[-1]["local_idx"] <= RUN_GAP_THRESHOLD:
                run.append(r)
            else:
                thinned_sampled.append(run[len(run) // 2])  # dai dien = frame giua doan
                n_runs += 1
                run = [r]
        thinned_sampled.append(run[len(run) // 2])
        n_runs += 1

    print(
        f"Tach: {len(exhaustive_group)} frame tu lan chay exhaustive cu (dung het) + "
        f"{len(sampled_group)} frame tu lan chay sample moi -> gom thanh {n_runs} doan lien tuc "
        f"(theo video+loai vat the), lay 1 dai dien/doan -> con {len(thinned_sampled)} frame",
        file=sys.stderr,
    )
    return exhaustive_group + thinned_sampled


N_CROP_WORKERS = 32


def _prepare_chunks(flagged: list[dict]) -> list[tuple[str, list[dict], list[bytes], list[tuple[int, int]]]]:
    """PHA 1 (CPU, I/O-bound) - doc HET anh can dung truoc, KHONG xen ke voi goi Modal (GPU).
    Ly do: xen ke CPU (doc/crop) va GPU (goi remote) moi vong lap tao do tre lap lai (round-
    trip latency) khong can thiet - tach han 2 pha giong pattern build_region_embeddings.py
    (crop truoc, encode sau) de pha GPU chay lien tuc khong bi cho CPU giua chung."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in flagged:
        groups[_build_prompt(rec["phrases"])].append(rec)
    print(f"Gom thanh {len(groups)} nhom prompt (frame cung nhom -> 1 luot batch)", file=sys.stderr)

    # tach nhom lon thanh chunk <= MAX_IMAGES_PER_CALL truoc (chua doc anh)
    plan: list[tuple[str, list[dict]]] = []
    for prompt, recs in groups.items():
        for ci in range(0, len(recs), MAX_IMAGES_PER_CALL):
            plan.append((prompt, recs[ci: ci + MAX_IMAGES_PER_CALL]))

    def _read_one(rec: dict) -> tuple[bytes, tuple[int, int]]:
        img_bytes = read_keyframe_bytes(rec["video_id"], rec["local_idx"])
        with Image.open(io.BytesIO(img_bytes)) as im:
            return img_bytes, im.size

    # doc song song TOAN BO anh can dung (khong phan biet chunk nao) - ThreadPoolExecutor vi
    # day la I/O-bound (doc zip + decode JPEG), giong pattern cac script build_*.py khac.
    all_recs_flat = [rec for _, chunk in plan for rec in chunk]
    print(f"Dang doc {len(all_recs_flat)} anh (song song {N_CROP_WORKERS} luong)...", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=N_CROP_WORKERS) as pool:
        read_results = list(pool.map(_read_one, all_recs_flat))
    print(f"Da doc xong {len(read_results)} anh.", file=sys.stderr)

    # ghep lai dung theo chunk ban dau
    idx = 0
    prepared = []
    for prompt, chunk in plan:
        n = len(chunk)
        chunk_bytes = [r[0] for r in read_results[idx: idx + n]]
        chunk_sizes = [r[1] for r in read_results[idx: idx + n]]
        idx += n
        prepared.append((prompt, chunk, chunk_bytes, chunk_sizes))
    return prepared


def _load_done_keys() -> set[tuple[str, int]]:
    if not DONE_PATH.exists():
        return set()
    done = set()
    with open(DONE_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                done.add((rec["video_id"], rec["local_idx"]))
            except json.JSONDecodeError:
                continue
    return done


def run() -> None:
    flagged = _load_flagged()
    print(f"Tong {len(flagged)} frame bi gan co", file=sys.stderr)
    if not flagged:
        return
    flagged = _split_and_thin(flagged)
    print(f"Sau khi loc: {len(flagged)} frame se goi Grounding DINO", file=sys.stderr)

    done_keys = _load_done_keys()
    if done_keys:
        before = len(flagged)
        flagged = [r for r in flagged if (r["video_id"], r["local_idx"]) not in done_keys]
        print(f"Resume: da xong {len(done_keys)} frame tu lan chay truoc, "
              f"con lai {len(flagged)}/{before} can xu ly", file=sys.stderr)
    if not flagged:
        print("Khong con gi de lam.", file=sys.stderr)
        return

    prepared = _prepare_chunks(flagged)  # PHA 1: CPU xong het o day
    print(f"Chuan bi xong {len(prepared)} chunk, bat dau PHA 2 (GPU, goi Modal.map() song song "
          f"nhieu container)...", file=sys.stderr)

    # PHA 2 (GPU) - dung .map() (khong phai .remote() tuan tu) de Modal tu dong scale nhieu
    # container song song - .remote() tung cai 1 chi bao gio co 1 request dang bay, Modal
    # KHONG CO LY DO gi de mo them container du MAX_CONTAINERS dat cao (bug hieu nang phat hien
    # 2026-08-06, nguoi dung hoi vi sao chi thay 1 container chay).
    Detector = modal.Cls.from_name("aic2026-grounding-dino", "Detector")
    detector = Detector()

    images_iter = [p[2] for p in prepared]
    prompts_iter = [p[0] for p in prepared]

    # load lai ket qua da co (neu resume) de khong ghi de mat detection cu
    all_rows: list[dict] = pd.read_parquet(OUT_PATH).to_dict("records") if OUT_PATH.exists() else []

    n_errors = 0
    done_f = open(DONE_PATH, "a", encoding="utf-8")
    try:
        map_results = detector.detect_batch.map(
            images_iter, prompts_iter,
            kwargs={"threshold": THRESHOLD, "text_threshold": TEXT_THRESHOLD},
            order_outputs=True, return_exceptions=True,
        )
        chunks_iter = [p[1] for p in prepared]
        sizes_iter = [p[3] for p in prepared]
        for ci, (results, chunk, sizes) in enumerate(zip(map_results, chunks_iter, sizes_iter), 1):
            if isinstance(results, Exception):
                n_errors += 1
                print(f"  [LOI] chunk {ci}/{len(prepared)}: {type(results).__name__} {str(results)[:120]}", file=sys.stderr)
                continue

            for rec, (w, h), dets in zip(chunk, sizes, results):
                for d in dets:
                    xmin, ymin, xmax, ymax = d["box"]
                    all_rows.append({
                        "video_id": rec["video_id"],
                        "local_idx": rec["local_idx"],
                        "label": d["label"],
                        "score": d["score"],
                        "ymin": max(0.0, ymin / h),
                        "xmin": max(0.0, xmin / w),
                        "ymax": min(1.0, ymax / h),
                        "xmax": min(1.0, xmax / w),
                        "source": "open_vocab_dino",
                    })
                done_f.write(json.dumps({"video_id": rec["video_id"], "local_idx": rec["local_idx"]}, ensure_ascii=False) + "\n")
            done_f.flush()
            os.fsync(done_f.fileno())

            if ci % CHECKPOINT_EVERY_GROUPS == 0 or ci == len(prepared):
                atomic_write_parquet(pd.DataFrame(all_rows), OUT_PATH, index=False)
                print(f"[chunk {ci}/{len(prepared)}] checkpoint {len(all_rows)} detection, "
                      f"{n_errors} chunk loi -> {OUT_PATH}", file=sys.stderr)
    finally:
        done_f.close()

    df = pd.DataFrame(all_rows)
    atomic_write_parquet(df, OUT_PATH, index=False)
    print(f"\nDa luu {len(df)} detection (open-vocab), {n_errors} chunk loi -> {OUT_PATH}", file=sys.stderr)
    if not df.empty:
        print(df.label.value_counts().to_string(), file=sys.stderr)


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    run()
