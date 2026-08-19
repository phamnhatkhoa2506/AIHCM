"""Giai đoạn 1+2 của thiết kế "VLM khám phá vocab mở -> Grounding DINO định vị" (chốt hướng
2026-08-06, xem hội thoại — phát hiện gốc: detector closed-set 514 nhãn OpenImages V4 ép nhãn
SAI khi gặp vật thể ngoài tập, vd "Dog" cho đầu lân múa lân, "Camel" cho người múa rồng,
"Watercraft" cho background trường quay tin tức — audit_object_labels.py + test_grounding_dino.py
đã xác nhận bằng ảnh thật).

GIAI ĐOẠN 1 (hàm run_discovery): mỗi frame -> gọi VLM (Qwen2.5-VL qua Modal, app RIÊNG
`modal_infra/vocab_discovery_app.py`, KHÁC app P1) hỏi "có vật thể/trang phục nào KHÔNG thuộc
514 nhãn chuẩn không". Output CỐ Ý ngắn (mảng string, tối đa vài cụm) — khác hẳn P1 (JSON quan
hệ từng cặp object, output dài tỉ lệ thuận số object/frame) nên rẻ hơn nhiều, chạy được TOÀN BỘ
corpus (177,321 frame) chứ không chỉ lấy mẫu.

GIAI ĐOẠN 2 (hàm aggregate_vocab): gộp toàn bộ cụm đã đề xuất, chuẩn hoá, đếm tần suất, lọc bỏ
cụm hiếm (khả năng VLM hallucinate) — MIN_FREQ do người dùng chốt (2026-08-06): "giới hạn từ
2-5", để mặc định 3 (giữa khoảng), CHỈNH TRỰC TIẾP Ở ĐÂY khi tune.

GIAI ĐOẠN 3 (Grounding DINO định vị bằng vocab đã gộp) CHƯA code ở đây — làm sau khi xem kết
quả giai đoạn 1+2 (đúng yêu cầu "viết xong để kiểm tra đã" trước khi chạy tiếp).
"""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/

import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from config import INDEX_DIR, INDEX_META_PATH, MODAL_VOCAB_URL
from keyframe_images import read_keyframe_bytes

RAW_OUTPUT_PATH = INDEX_DIR / "vocab_discovery_raw.jsonl"
VOCAB_OUT_PATH = INDEX_DIR / "discovered_vocab.json"
FLAGGED_FRAMES_OUT_PATH = INDEX_DIR / "vocab_discovery_flagged_frames.jsonl"

N_WORKERS = 80  # MAX_CONTAINERS thuc te bi gioi han 10 (tai khoan Modal, 2026-08-06, khong
# phai 16 nhu tuong truoc) - giu ty le ~8 worker/container da xac nhan bao hoa o muc 8 container.
CHECKPOINT_EVERY = 200  # in progress thuong xuyen hon (2000 qua thua, tuong nhu dung yen)

# Nguong tan suat giu lai cum tu (loc nhieu hallucination) - nguoi dung chot (2026-08-06):
# "gioi han tu 2-5", de mac dinh 3 (giua khoang), CHINH TAI DAY khi tune.
MIN_FREQ = 3

with open(INDEX_DIR / "label_vi.json", encoding="utf-8") as _f:
    _KNOWN_LABELS = sorted(json.load(_f).keys())

SYSTEM_PROMPT = f"""You are given ONE image and a fixed list of {len(_KNOWN_LABELS)} known
object categories already detected by an existing system:
{", ".join(_KNOWN_LABELS)}

Look at the image and identify any DISTINCT visible subject, costume, mascot, prop, or object
that does NOT fit naturally into any of the categories above (e.g. a lion-dance costume, a
dragon-dance costume, a news-studio graphic backdrop, a mascot suit, a specific tool or
instrument not in the list). Do NOT propose something just because it could be phrased
differently from a list entry that already covers it (e.g. do not propose "human being" — an
entry for it already exists).

Reply with a short JSON array of English noun phrases (max 5 items, each 1-4 words, suitable
as an object-detection query — e.g. "lion dance costume"). Return an empty array [] if
everything visible is already covered by the known categories. Do NOT explain, just the array."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "novel_objects": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["novel_objects"],
}


def build_request(video_id: str, local_idx: int) -> dict:
    img_bytes = read_keyframe_bytes(video_id, local_idx)
    img_b64 = base64.b64encode(img_bytes).decode()
    return {
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "List novel objects (if any) per the instructions."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ],
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "novel_objects", "schema": OUTPUT_SCHEMA, "strict": True},
        },
        "max_tokens": 200,  # output co dinh ngan (mang toi da 5 cum, khong phu thuoc so object)
        "temperature": 0.1,
    }


def call_model(payload: dict, base_url: str = MODAL_VOCAB_URL, timeout: int = 120) -> list[str]:
    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=timeout)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return json.loads(content).get("novel_objects", [])


def _process_one(video_id: str, local_idx: int) -> dict:
    try:
        payload = build_request(video_id, local_idx)
        novel = call_model(payload)
        return {"video_id": video_id, "local_idx": int(local_idx), "novel_objects": novel, "error": None}
    except Exception as e:  # phong than: 1 frame loi khong duoc lam chet ca batch
        return {"video_id": video_id, "local_idx": int(local_idx), "novel_objects": [], "error": str(e)}


def _load_done_keys() -> set[tuple[str, int]]:
    """BUG DA SUA (2026-08-06): truoc day tinh CA record loi la "da xong" -> frame loi (vd
    404 luc server bi dung/deploy lai giua chung) khong bao gio duoc thu lai. Gio chi tinh
    record THANH CONG (khong co "error") la da xong - record loi bi bo qua, se tu dong nam
    trong todo cua lan chay sau va duoc goi lai."""
    if not RAW_OUTPUT_PATH.exists():
        return set()
    done = set()
    with open(RAW_OUTPUT_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("error"):
                continue
            done.add((rec["video_id"], rec["local_idx"]))
    return done


# Sample thua thay vi quet TOAN BO 177,321 frame - phat hien 2026-08-06 (khi da chay ~26k
# frame): vocab lap lai RAT NHIEU trong cung 1 video (vd "news studio graphic" xuat hien o
# hang chuc frame lien tiep cung video) -> quet exhaustive lang phi, khong doi lai thong tin
# moi tuong xung. SKIP_INTRO bo qua vai keyframe dau (thuong la doan gioi thieu/dao dau chuong
# trinh, khong dai dien noi dung chinh). STRIDE lay 1 frame moi N frame trong PHAN CON LAI cua
# video (1/10-1/20 tong so, nguoi dung chon).
SAMPLE_SKIP_INTRO = 5
SAMPLE_STRIDE = 15


def _sampled_frames(skip_intro: int = SAMPLE_SKIP_INTRO, stride: int = SAMPLE_STRIDE) -> pd.DataFrame:
    meta = pd.read_parquet(INDEX_META_PATH).sort_values(["video_id", "local_idx"])
    parts = []
    for _, g in meta.groupby("video_id", sort=False):
        g = g.iloc[skip_intro:]  # bo qua N keyframe dau (doan gioi thieu/dao dau)
        parts.append(g.iloc[::stride])  # lay 1 frame moi `stride` frame
    return pd.concat(parts, ignore_index=True) if parts else meta.iloc[0:0]


def run_discovery(limit: int | None = None, sampled: bool = True) -> None:
    """Chay Giai doan 1. Mac dinh `sampled=True`: chi quet tap con thua (xem _sampled_frames)
    thay vi TOAN BO 177,321 frame - re hon nhieu, van du de kham pha vocab lap lai trong video.
    Resume: bo qua (video_id, local_idx) da co trong RAW_OUTPUT_PATH tu lan chay truoc (kem ca
    ket qua tu lan chay FULL truoc do, khong mat gi)."""
    meta = _sampled_frames() if sampled else pd.read_parquet(INDEX_META_PATH)
    if limit:
        meta = meta.head(limit)

    done = _load_done_keys()
    todo = [(r.video_id, int(r.local_idx)) for r in meta.itertuples(index=False) if (r.video_id, int(r.local_idx)) not in done]
    print(f"Tong {len(meta)} frame, da xong {len(done)}, con lai {len(todo)}", file=sys.stderr)
    if not todo:
        print("Khong con gi de lam.", file=sys.stderr)
        return

    n_done = 0
    t0 = time.time()
    with open(RAW_OUTPUT_PATH, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {pool.submit(_process_one, vid, li): (vid, li) for vid, li in todo}
            for fut in as_completed(futures):
                rec = fut.result()
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                # flush+fsync MOI dong, khong doi CHECKPOINT_EVERY - job chay nhieu gio, tien
                # trinh co the bi kill bat cu luc nao (da gap that voi build_region_embeddings.py,
                # SIGKILL khong kip flush buffer OS -> mat het viec chua flush). Chi phi flush
                # 1 dong JSON nho la khong dang ke so voi rui ro mat vai nghin request da tra tien.
                out_f.flush()
                os.fsync(out_f.fileno())
                n_done += 1
                if n_done % CHECKPOINT_EVERY == 0 or n_done == len(todo):
                    elapsed = time.time() - t0
                    rate = n_done / elapsed if elapsed > 0 else 0
                    eta_min = (len(todo) - n_done) / rate / 60 if rate > 0 else float("nan")
                    print(f"[{n_done}/{len(todo)}] {rate:.2f} frame/s, ETA {eta_min:.1f} phut", file=sys.stderr)

    n_errors = sum(1 for line in open(RAW_OUTPUT_PATH, encoding="utf-8") if json.loads(line).get("error"))
    print(f"Xong Giai doan 1. Loi: {n_errors}/{len(meta)}", file=sys.stderr)


def _normalize_phrase(p: str) -> str:
    return " ".join(p.strip().lower().split())


def aggregate_vocab(min_freq: int = MIN_FREQ) -> None:
    """Giai doan 2: doc RAW_OUTPUT_PATH, chuan hoa + dem tan suat cum tu, loc theo min_freq,
    luu vocab cuoi + danh sach frame da bi gan co (dung cho Giai doan 3 - DINO)."""
    if not RAW_OUTPUT_PATH.exists():
        print("Chua co du lieu Giai doan 1, chay run_discovery() truoc.", file=sys.stderr)
        return

    freq: dict[str, int] = {}
    frame_phrases: dict[tuple[str, int], list[str]] = {}
    with open(RAW_OUTPUT_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("error") or not rec.get("novel_objects"):
                continue
            norm_phrases = [_normalize_phrase(p) for p in rec["novel_objects"] if p.strip()]
            if not norm_phrases:
                continue
            frame_phrases[(rec["video_id"], rec["local_idx"])] = norm_phrases
            for p in norm_phrases:
                freq[p] = freq.get(p, 0) + 1

    kept_vocab = {p: c for p, c in freq.items() if c >= min_freq}
    print(f"Tong {len(freq)} cum khac nhau, giu lai {len(kept_vocab)} cum (freq>={min_freq})", file=sys.stderr)

    with open(VOCAB_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(kept_vocab.items(), key=lambda x: -x[1])), f, ensure_ascii=False, indent=2)
    print(f"Da luu vocab -> {VOCAB_OUT_PATH}", file=sys.stderr)

    kept_set = set(kept_vocab)
    with open(FLAGGED_FRAMES_OUT_PATH, "w", encoding="utf-8") as f:
        n_flagged = 0
        for (vid, li), phrases in frame_phrases.items():
            kept_phrases = [p for p in phrases if p in kept_set]
            if kept_phrases:
                f.write(json.dumps({"video_id": vid, "local_idx": li, "phrases": kept_phrases}, ensure_ascii=False) + "\n")
                n_flagged += 1
    print(f"Da luu {n_flagged} frame bi gan co (co >=1 cum qua nguong) -> {FLAGGED_FRAMES_OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    # Doi chien thuat (2026-08-06): thay vi FULL 177,321 frame, chi sample thua (skip 5 frame
    # dau/video + lay 1/15 frame con lai) - da co san ~26k frame ket qua tu lan chay FULL truoc,
    # resume se tan dung lai, khong mat gi.
    run_discovery(limit=None, sampled=True)
    aggregate_vocab()
