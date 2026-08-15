"""TANG TIM KIEM CHINH (2026-08-15, cutover) - dung tren bo keyframe TU TRICH mat do cao hon
(AutoShot shot-detection, xem hoi thoai 2026-08-11/13), encode bang 3 model rieng
(SigLIP2/PE-Core/BEiT-3, Modal apps: aic2026-siglip/aic2026-pe-core/aic2026-beit3) + 1 che do
fusion RRF (Reciprocal Rank Fusion). BO KEYFRAME BTC GOC (tier1_filter/tier2_vector/resources.py
o thu muc nay) KHONG con la nguon chinh nua - theo yeu cau nguoi dung (2026-08-15) "khong su
dung keyframe cua BTC nhe" - CHI con giu lai cho Tier3 (TRAKE)/Region-CLIP rerank/ASR boost
(xem gioi han duoi) cho toi khi migrate xong.

Object-filter (2026-08-15): OWLv2 closed-set (514 nhan) DA chay xong toan bo 369,589 anh dense
(xem index/dense/objects_index.parquet) - _object_candidates() duoi day. CHUA co open-vocab
(Grounding DINO) rieng cho bo dense (se chay sau) - "source" hien chi co "owlv2", coi nhu
tuong duong "closed_set" cua bo BTC cu, KHONG co suppression (chua co open-vocab de suppress).

GIOI HAN CON LAI (2026-08-15, chua lam trong lan cutover nay - xem hoi thoai):
  - Tier3 (TRAKE/temporal): dense_meta.parquet KHONG co pts_time/fps -> tier3_temporal.py van
    tro ve du lieu BTC cu (search.py nhanh anchors), KHONG dung duoc tren bo dense nay.

ASR boost (2026-08-15, MOI wire): dung index/dense/asr_text.parquet (xem
offline/build_dense_asr_index.py - nguon tu index/asr_text_new.parquet, ban ASR MOI co
timestamp GIAY THAT + text da qua LLM sua loi/dau cau, KHAC han ban ASR cu map theo local_idx
cua BTC). Frame_idx_start/end da tinh san qua fps (index/meta.parquet). Soft-boost (KHONG loai
frame nao) theo "audio_mentions" LLM trich duoc (query_planner.py) - vd "MC nhắc tới Pháp" ->
kiem tra "Pháp" co duoc noi GAN frame ung vien khong (co dem bien do +-3s vi loi noi khong dong
bo chinh xac voi hinh, giong ASR_CONTEXT_WINDOW cu).

Region-CLIP rerank thuoc tinh (2026-08-15, MOI wire): dung region_embeddings_siglip.npy +
region_embeddings_siglip_detection_ids.npy (offline/build_dense_region_embeddings_shard.py,
scope person+animal+vehicle+clothing_accessory+surface, score>=0.25) - CHI SigLIP2 (theo yeu
cau nguoi dung 2026-08-15: khong chay het 3 model, rerank dung 1 model duy nhat bat ke search
mode nao dang dung - rerank la buoc CONG DIEM bo sung, khong bat buoc cung khong gian voi
model xep hang ban dau). Job build embedding CHUA chay het (~1.72M/2.6M object da co embedding
tai thoi diem tich hop - object thieu embedding cho diem 0 cho box do, KHONG loi/crash) - chay
tiep phan con lai sau se tu dong cai thien do chinh xac rerank ma khong can sua code them.

CO OCR rieng (2026-08-14, build_dense_ocr_index.py, OCR THANG tren bo dense - xem OCR_TEXT_PATH
duoi) dung nhu HARD FILTER truoc khi rank (giong tier1_filter.by_text() nhung scope o bo dense,
khac cho KHONG co "run" gom cum - moi dong da la 1 frame don, xem _object_candidates duoi).

frame_id nop bai = cot "frame_idx" trong dense_meta.parquet (frame tren truc VIDEO GOC, xem
merge_dense_embeddings.py va PDF muc 3) - dung THANG duoc, khong can quy doi.

4 che do (DENSE_MODES): "siglip" | "pe_core" | "beit3" | "rrf".
"""
from __future__ import annotations

import contextlib
from functools import lru_cache

import faiss
import numpy as np
import pandas as pd

from config import INDEX_DIR
from label_translate import resolve as resolve_label_vi
from local_text_encoders import ENCODERS
from tiers.tier1_filter import _ordered_words_match, by_metadata

DENSE_DIR = INDEX_DIR / "dense"
OCR_TEXT_PATH = DENSE_DIR / "ocr_text.parquet"  # xem build_dense_ocr_index.py - schema:
# video_id, frame_idx, text_raw, text_norm, ymin/xmin/ymax/xmax, score. KHAC ocr_text.parquet
# sparse (co local_idx_start/end, da gom "run") - o day 1 dong/dong chu/frame THANG, khong gom,
# vi frame dense khong deu theo shot (xem docstring build_dense_ocr_index.py).
OBJECTS_INDEX_PATH = DENSE_DIR / "objects_index.parquet"  # xem build_dense_objects_shard.py -
# schema: video_id, frame_idx, label, score, ymin/xmin/ymax/xmax, source ("owlv2", closed-set
# 514 nhan, TOAN BO 369,589 anh - xem hoi thoai 2026-08-14/15).
DENSE_MODES = ["siglip", "pe_core", "beit3", "rrf"]

RRF_K = 60  # hang so chuan trong cong thuc RRF: 1/(k+rank) - k=60 la gia tri pho bien trong
# literature (Cormack et al. 2009), lam mem anh huong cua rank thap ma khong can chuan hoa diem.


@lru_cache(maxsize=None)
def _load_dense_meta() -> pd.DataFrame:
    return pd.read_parquet(DENSE_DIR / "dense_meta.parquet")


@lru_cache(maxsize=None)
def _load_dense_row_pos() -> dict[tuple, int]:
    meta = _load_dense_meta()
    return {(vid, int(fid)): i for i, (vid, fid) in enumerate(zip(meta["video_id"], meta["frame_idx"]))}


@lru_cache(maxsize=None)
def _load_dense_index(model: str):
    matrix = np.load(DENSE_DIR / f"{model}_matrix.npy")
    index = faiss.read_index(str(DENSE_DIR / f"{model}_faiss.index"))
    return matrix, index


@lru_cache(maxsize=1)
def _load_objects_index() -> pd.DataFrame | None:
    if not OBJECTS_INDEX_PATH.exists():
        return None
    return pd.read_parquet(OBJECTS_INDEX_PATH)


# ============================================================ Region-CLIP rerank (2026-08-15)
# SUA (theo de xuat nguoi dung "region-CLIP cung lam 1 server rieng nhu embedding model") -
# KHONG con nap region_embeddings_siglip.npy (5.3GB) + objects_index.parquet (14.5 trieu dong)
# TRUC TIEP tren may local nua (truoc day gay cham/an RAM lan dau moi phien Streamlit). Toan bo
# du lieu + tinh toan chuyen sang server Modal rieng (aic2026-region-rerank, CPU, luon giu am -
# xem offline/modal_infra/region_rerank_app.py + offline/upload_region_index_to_volume.py de
# dong bo du lieu len Volume "aic2026-region-index"). May local CHI gui (video_id, frame_id) +
# nhan + cau thuoc tinh (rat nhe), nhan ve diem so - khong con giu du lieu nang trong RAM local.
REGION_RERANK_APP_NAME = "aic2026-region-rerank"
REGION_RERANK_CLASS_NAME = "RegionRerankServer"
REGION_CLIP_WEIGHT = 0.5  # cung gia tri voi ban BTC cu (query_planner.py) - da tune truoc do


@lru_cache(maxsize=1)
def _region_rerank_server():
    import modal

    Server = modal.Cls.from_name(REGION_RERANK_APP_NAME, REGION_RERANK_CLASS_NAME)
    return Server()


def apply_region_clip_rerank(
    results: pd.DataFrame, attributes: list[dict], top_k: int, log=None
) -> pd.DataFrame:
    """Rerank theo thuoc tinh (vd "nguoi mac ao dai mau tim") - CHI dung SigLIP2 (theo yeu cau
    nguoi dung 2026-08-15: khong chay Region-CLIP cho ca 3 model, bat ke search mode dang dung
    la gi). Tinh toan THAT SU chay tren server Modal (aic2026-region-rerank) - ham nay chi goi
    .remote() va gop ket qua, khong tu tinh cosine/loc data nua."""
    frame_keys = [[vid, int(fid)] for vid, fid in zip(results["video_id"], results["frame_id"])]
    per_attr_scores: list[list[float]] = []

    for attr in attributes:
        entity_term, attribute_text = attr["entity_term"], attr["attribute_text"]
        step_name = f"Region-CLIP (SigLIP2, server rieng) — '{attribute_text}' (entity: {entity_term})"
        with (log.timed(step_name) if log else contextlib.nullcontext(lambda *_a, **_k: None)) as set_detail:
            entity_labels = resolve_label_vi(entity_term)
            if not entity_labels:
                scores = [0.0] * len(results)
                if log:
                    set_detail("BỎ QUA — không resolve được nhãn")
                per_attr_scores.append(scores)
                continue

            try:
                scores = _region_rerank_server().rerank.remote(frame_keys, entity_labels, attribute_text)
            except Exception as e:
                scores = [0.0] * len(results)
                if log:
                    set_detail(f"LỖI gọi server rerank: {type(e).__name__} {str(e)[:120]} — coi như không khớp")
                per_attr_scores.append(scores)
                continue

            per_attr_scores.append(scores)
            n_match = sum(1 for s in scores if s > 0)
            if log:
                set_detail(f"nhãn={entity_labels}, {n_match}/{len(results)} frame có object khớp")

    region_score = np.mean(per_attr_scores, axis=0) if per_attr_scores else np.zeros(len(results))
    results = results.copy()
    results["region_score"] = region_score
    results["score_before_rerank"] = results["score"]
    results["score"] = results["score"] + REGION_CLIP_WEIGHT * results["region_score"]
    return results.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)


def _object_candidates(
    must_have_labels: list[str] | None, min_count: dict[str, int] | None
) -> set[tuple[str, int]] | None:
    """Giong tier1_filter.by_objects() nhung key la (video_id, frame_idx) va scope LUON la
    toan bo objects_index dense (chi 1 nguon "owlv2" closed-set, chua co open-vocab/suppression
    - xem gioi han o docstring dau file)."""
    if not must_have_labels and not min_count:
        return None
    objects_index = _load_objects_index()
    if objects_index is None:
        raise RuntimeError("index/dense/objects_index.parquet chua build")

    need: dict[str, int] = {}
    for lb in must_have_labels or []:
        need[lb] = max(need.get(lb, 1), 1)
    for lb, c in (min_count or {}).items():
        need[lb] = max(need.get(lb, 1), c)

    allowed: set[tuple[str, int]] | None = None
    for lb, min_c in need.items():
        sub = objects_index[objects_index["label"] == lb]
        cnt = sub.groupby(["video_id", "frame_idx"]).size()
        keys = set(cnt[cnt >= min_c].index)
        allowed = keys if allowed is None else (allowed & keys)
        if not allowed:
            return set()
    return allowed


def _ocr_candidates(ocr_text: str) -> set[tuple[str, int]] | None:
    """Tra ve set (video_id, frame_idx) co chu KHOP CHINH XAC ocr_text (word-boundary, khong
    phan biet dau) - giong het tier1_filter.by_text(), scope o OCR_TEXT_PATH cua bo dense.
    None neu chua co du lieu OCR dense (chua chay xong build_dense_ocr_index.py)."""
    if not OCR_TEXT_PATH.exists():
        return None
    df = pd.read_parquet(OCR_TEXT_PATH)
    hit = df[df["text_norm"].apply(lambda t: _ordered_words_match(ocr_text, t))]
    return set(zip(hit["video_id"], hit["frame_idx"].astype(int)))


# ============================================================ ASR boost (2026-08-15)
ASR_TEXT_PATH = DENSE_DIR / "asr_text.parquet"  # xem offline/build_dense_asr_index.py - schema:
# video_id, frame_idx_start, frame_idx_end, start, end (giay), text_raw (= text_refined da qua
# LLM sua), text_norm.
META_PATH = INDEX_DIR / "meta.parquet"  # video-level fps, dung chung BTC/dense (khong doi theo
# mat do keyframe).
ASR_CONTEXT_WINDOW_SECONDS = 3.0  # loi noi KHONG dong bo chinh xac voi hinh (giong ASR_CONTEXT_
# WINDOW cu trong submission_pipeline.py, o day tinh theo GIAY roi quy doi ra frame qua fps
# rieng tung video, thay vi 1 hang so local_idx co dinh).
AUDIO_MENTION_BOOST_WEIGHT = 0.08  # cung thang do voi ASR_BOOST_WEIGHT (query_planner.py cu)


@lru_cache(maxsize=1)
def _load_dense_asr() -> pd.DataFrame | None:
    if not ASR_TEXT_PATH.exists():
        return None
    return pd.read_parquet(ASR_TEXT_PATH)


@lru_cache(maxsize=1)
def _fps_by_video() -> dict[str, float]:
    meta = pd.read_parquet(META_PATH)[["video_id", "fps"]].drop_duplicates("video_id")
    return dict(zip(meta["video_id"], meta["fps"]))


def _audio_mention_boost(results: pd.DataFrame, audio_mentions: list[dict]) -> np.ndarray:
    """Diem CONG THEM (KHONG loai frame nao) neu tu/cum "audio_mentions" (LLM trich, vd "MC
    nhắc tới Pháp" -> term="Pháp") duoc NOI GAN frame ung vien (dem bien do +-ASR_CONTEXT_
    WINDOW_SECONDS vi loi noi khong dong bo chinh xac voi hinh, giong pattern ASR boost cu o
    query_planner.py). Nhi phan (co/khong noi, khong co diem tin cay per-doan) - nhieu mention
    cung khop 1 frame duoc CLIP ve 1.0 (khong cong don), giong het pattern cu."""
    asr = _load_dense_asr()
    if asr is None or not audio_mentions or results.empty:
        return np.zeros(len(results))
    fps_map = _fps_by_video()

    boost = np.zeros(len(results))
    for mention in audio_mentions:
        term = (mention.get("term") or "").strip()
        if not term:
            continue
        hit = asr[asr["text_norm"].apply(lambda t: _ordered_words_match(term, t))]
        if hit.empty:
            continue
        match = np.zeros(len(results))
        for i, r in enumerate(results.itertuples(index=False)):
            sub = hit[hit["video_id"] == r.video_id]
            if sub.empty:
                continue
            fps = fps_map.get(r.video_id, 25.0)
            window = int(round(ASR_CONTEXT_WINDOW_SECONDS * fps))
            frame_id = int(r.frame_id)
            in_range = ((sub["frame_idx_start"] - window <= frame_id) & (sub["frame_idx_end"] + window >= frame_id)).any()
            if in_range:
                match[i] = 1.0
        boost += match

    return np.clip(boost, 0, 1)


# ============================================================ Khung loc theo vi tri (2026-08-15)
# UI (xem app.py): 1 canvas duy nhat, ve nhieu khung, moi khung gan loai OCR (chu) hoac Object
# (nhan) + vi tri (ymin,xmin,ymax,xmax chuan hoa [0,1]).
#
# SUA (2026-08-15, theo yeu cau nguoi dung: "nguoi dung khong the ve chinh xac") - TACH LAM 2,
# khong con dung IoU lam HARD CUTOFF nua (truoc day: khung lech vi tri -> LOAI HAN frame dung,
# du nguong da rat thap 0.05):
#   1. HARD FILTER = NOI DUNG (chu gi/nhan gi) - dang tin cay, GIU NGUYEN co che loai frame
#      khong co chu/nhan do o DAU DO trong frame (khong xet vi tri) - xem _ocr_box_candidates/
#      _object_box_candidates duoi.
#   2. SOFT BOOST = VI TRI (khung ve khop bao nhieu voi detection that) - khong con loai frame
#      nao, chi CONG DIEM theo IoU (giong nguyen tac da co: OCR text=hard, ASR/secondary=soft) -
#      xem _ocr_box_position_scores/_object_box_position_scores + _spatial_position_boost duoi.
SPATIAL_POSITION_BOOST_WEIGHT = 0.15  # cung thang do voi SECONDARY_ENTITY_BOOST_WEIGHT (query_planner.py)


def _box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    ay0, ax0, ay1, ax1 = boxes_a[:, 0:1], boxes_a[:, 1:2], boxes_a[:, 2:3], boxes_a[:, 3:4]
    by0, bx0, by1, bx1 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]
    iy0, ix0 = np.maximum(ay0, by0), np.maximum(ax0, bx0)
    iy1, ix1 = np.minimum(ay1, by1), np.minimum(ax1, bx1)
    inter = np.clip(iy1 - iy0, 0, None) * np.clip(ix1 - ix0, 0, None)
    area_a = np.clip(ay1 - ay0, 0, None) * np.clip(ax1 - ax0, 0, None)
    area_b = np.clip(by1 - by0, 0, None) * np.clip(bx1 - bx0, 0, None)
    union = area_a + area_b - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _ocr_box_candidates(text: str) -> set[tuple[str, int]]:
    """1 khung loai OCR (chu, khong dau) -> set (video_id, frame_idx) co chu do O DAU DO trong
    frame (KHONG xet vi tri nua - xem docstring o tren). Tra set RONG (khong phai None) vi
    khung nguoi dung ve la rang buoc THAT, khong duoc coi nhu "khong loc gi" khi thieu du lieu."""
    if not OCR_TEXT_PATH.exists():
        return set()
    df = pd.read_parquet(OCR_TEXT_PATH)
    if text:
        df = df[df["text_norm"].apply(lambda t: _ordered_words_match(text, t))]
    return set(zip(df["video_id"], df["frame_idx"].astype(int)))


def _object_box_candidates(labels: list[str], min_count: int = 1) -> set[tuple[str, int]]:
    """1 khung loai Object (nhan, co the nhieu nhan dong nghia tu resolve_label_vi - OR giua
    cac nhan dong nghia) -> set (video_id, frame_idx) co DU so luong object khop nhan O DAU DO
    trong frame (KHONG xet vi tri nua)."""
    objects_index = _load_objects_index()
    if objects_index is None or not labels:
        return set()
    df = objects_index[objects_index["label"].isin(labels)]
    cnt = df.groupby(["video_id", "frame_idx"]).size()
    return set(cnt[cnt >= min_count].index)


def _ocr_box_position_scores(text: str, region: tuple[float, float, float, float]) -> dict[tuple[str, int], float]:
    """(video_id, frame_idx) -> IoU CAO NHAT giua khung nguoi dung ve va cac dong OCR khop
    `text` trong frame do - dung lam SOFT BOOST (xem _spatial_position_boost), KHONG loai frame
    nao ca."""
    if not OCR_TEXT_PATH.exists():
        return {}
    df = pd.read_parquet(OCR_TEXT_PATH)
    if text:
        df = df[df["text_norm"].apply(lambda t: _ordered_words_match(text, t))]
    if df.empty:
        return {}
    boxes = df[["ymin", "xmin", "ymax", "xmax"]].to_numpy(dtype=float)
    iou = _box_iou_matrix(boxes, np.array([region], dtype=float)).flatten()
    df = df.assign(_iou=iou)
    return {(vid, int(fid)): score for (vid, fid), score in df.groupby(["video_id", "frame_idx"])["_iou"].max().items()}


def _object_box_position_scores(
    labels: list[str], region: tuple[float, float, float, float]
) -> dict[tuple[str, int], float]:
    """(video_id, frame_idx) -> IoU CAO NHAT giua khung nguoi dung ve va cac object khop
    `labels` trong frame do - dung lam SOFT BOOST, KHONG loai frame nao ca."""
    objects_index = _load_objects_index()
    if objects_index is None or not labels:
        return {}
    df = objects_index[objects_index["label"].isin(labels)]
    if df.empty:
        return {}
    boxes = df[["ymin", "xmin", "ymax", "xmax"]].to_numpy(dtype=float)
    iou = _box_iou_matrix(boxes, np.array([region], dtype=float)).flatten()
    df = df.assign(_iou=iou)
    return {(vid, int(fid)): score for (vid, fid), score in df.groupby(["video_id", "frame_idx"])["_iou"].max().items()}


def _spatial_position_boost(results: pd.DataFrame, spatial_boxes: list[dict]) -> np.ndarray:
    """Diem CONG THEM theo do khop VI TRI (IoU) cho tung khung CO ve vi tri - khung nguoi dung
    ve LECH van duoc GIU frame (0 diem, khong bi loai), chi khung khop tot moi duoc uu tien
    hon - xem hoi thoai 2026-08-15 (nguoi dung: "họ không thể vẽ chính xác")."""
    boxes_with_region = [b for b in spatial_boxes if b.get("region") is not None]
    if not boxes_with_region or results.empty:
        return np.zeros(len(results))
    boost = np.zeros(len(results))
    for box in boxes_with_region:
        if box["type"] == "ocr":
            scores_map = _ocr_box_position_scores(box.get("text", ""), box["region"])
        else:
            scores_map = _object_box_position_scores(box.get("labels") or [], box["region"])
        match = np.array([
            scores_map.get((r.video_id, int(r.frame_id)), 0.0) for r in results.itertuples(index=False)
        ])
        boost += match
    return boost / len(boxes_with_region)


def _encode_query(query: str, model: str) -> np.ndarray:
    # SUA (2026-08-14, theo yeu cau nguoi dung): encode QUERY TEXT chay LOCAL (giong het
    # pattern tier2_vector.py::encode_query() dung cho CLIP hien tai) thay vi goi Modal
    # remote() moi lan - tranh phu thuoc Modal app con song/da deploy cho duong hoi/dap ONLINE
    # (chi anh CORPUS moi thuc su can Modal GPU, da lam xong o build_dense_embeddings.py).
    v = ENCODERS[model](query)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    return v


def _rank_single(
    query: str, model: str, top_k: int, candidates: set[tuple[str, int]] | None = None, log=None
) -> pd.DataFrame:
    """1 model - tra ve DataFrame [video_id, frame_id (=frame_idx), shot_idx, path, score].
    candidates=None -> tim tren TOAN BO index qua FAISS (giong tier2_vector.rank()). candidates
    cu the (vd tu OCR hard-filter) -> tinh cosine TRUC TIEP tren dung tap do, KHONG dung
    FAISS-pool-roi-loc (cung nguyen tac voi tier2_vector.rank()). log: StepLog tuy chon (xem
    steplog.py) - ghi rieng thoi gian encode query + xep hang cho MODEL nay (2026-08-15, theo
    yeu cau nguoi dung: log thoi gian encode cho ca 4 che do siglip/pe_core/beit3/rrf - rrf goi
    lai chinh ham nay 3 lan, xem _rank_rrf duoi, nen tu no da duoc log rieng tung model)."""
    matrix, index = _load_dense_index(model)
    meta = _load_dense_meta()

    with (log.timed(f"Encode query + xếp hạng — {model}") if log else contextlib.nullcontext(lambda *_a, **_k: None)) as set_detail:
        qvec = _encode_query(query, model)

        if candidates is None:
            scores, idx = index.search(qvec, top_k)
            out = meta.iloc[idx[0]].copy()
            out["score"] = scores[0]
            result = out.rename(columns={"frame_idx": "frame_id"}).reset_index(drop=True)
        elif not candidates:
            result = meta.iloc[0:0].copy().rename(columns={"frame_idx": "frame_id"}).assign(score=[])
        else:
            row_pos = _load_dense_row_pos()
            positions = np.array([row_pos[k] for k in candidates if k in row_pos], dtype=np.int64)
            if len(positions) == 0:
                result = meta.iloc[0:0].copy().rename(columns={"frame_idx": "frame_id"}).assign(score=[])
            else:
                sub_matrix = matrix[positions]
                scores = sub_matrix @ qvec[0]
                order = np.argsort(-scores)[:top_k]
                out = meta.iloc[positions[order]].copy()
                out["score"] = scores[order]
                result = out.rename(columns={"frame_idx": "frame_id"}).reset_index(drop=True)

        set_detail(f"{len(result)} kết quả (candidates="
                    f"{'toàn corpus' if candidates is None else len(candidates)})")
    return result


def _rank_rrf(
    query: str, top_k: int, candidates: set[tuple[str, int]] | None = None, pool_k: int = 200, log=None
) -> pd.DataFrame:
    """Fusion RRF: lay top pool_k tu MOI model rieng le, tinh RRF-score = sum(1/(RRF_K+rank_i))
    qua cac model co xuat hien (khong xuat hien trong top pool_k cua 1 model nao do coi nhu
    rank vo cung, dong gop 0 tu model do - KHONG loai anh, chi khong duoc cong tu nguon do).
    log: truyen xuong _rank_single() cho TUNG model - xem timing rieng cua tung model trong 1
    lan chay rrf (thuong la model cham nhat quyet dinh tong thoi gian, vi chay TUAN TU)."""
    rrf_scores: dict[tuple, float] = {}
    per_model_row: dict[tuple, pd.Series] = {}
    for model in ("siglip", "pe_core", "beit3"):
        ranked = _rank_single(query, model, pool_k, candidates=candidates, log=log)
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            key = (row["video_id"], int(row["frame_id"]))
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            per_model_row.setdefault(key, row)

    order = sorted(rrf_scores.items(), key=lambda kv: -kv[1])[:top_k]
    rows = []
    for key, score in order:
        row = per_model_row[key].copy()
        row["score"] = score
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame(
        columns=["video_id", "frame_id", "shot_idx", "path", "score"]
    )


def _spatial_box_candidates(spatial_boxes: list[dict], op: str = "and") -> set[tuple[str, int]]:
    """spatial_boxes: [{"type": "ocr"|"object", "text": str, "labels": list[str],
    "min_count": int, "region": (ymin,xmin,ymax,xmax)|None}, ...] tu UI (xem app.py).
    op="and": frame phai khop DU MOI khung (giao). op="or": frame khop BAT KY khung nao
    (hop) - theo yeu cau nguoi dung 2026-08-15 (giong toggle "Objects in AND/OR" o UI tham
    khao)."""
    per_box = [
        _ocr_box_candidates(box.get("text", "")) if box["type"] == "ocr"
        else _object_box_candidates(box.get("labels") or [], box.get("min_count", 1))
        for box in spatial_boxes
    ]
    if not per_box:
        return set()
    if op == "or":
        result: set[tuple[str, int]] = set()
        for keys in per_box:
            result |= keys
        return result
    allowed = per_box[0]
    for keys in per_box[1:]:
        allowed = allowed & keys
    return allowed


def _combine_candidates(
    *,
    authors: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    keywords_any: list[str] | None = None,
    must_have_labels: list[str] | None = None,
    min_count: dict[str, int] | None = None,
    ocr_text: str | None = None,
    spatial_boxes: list[dict] | None = None,
    spatial_op: str = "and",
) -> set[tuple[str, int]] | None:
    """Gop metadata (video-level) + object + OCR (frame-level) + khung vi tri (spatial_boxes)
    thanh 1 tap (video_id, frame_idx). Cac khung vi tri gop VOI NHAU theo spatial_op ("and"/
    "or"), ROI ket qua do lai AND voi metadata/object/OCR toan cuc (giong tier1_filter.apply()
    nhung key la frame_idx)."""
    video_allowed = by_metadata(authors, date_from, date_to, keywords_any)
    object_allowed = _object_candidates(must_have_labels, min_count)
    text_allowed = _ocr_candidates(ocr_text) if ocr_text else None
    spatial_allowed = _spatial_box_candidates(spatial_boxes, spatial_op) if spatial_boxes else None

    frame_sets = [s for s in (object_allowed, text_allowed, spatial_allowed) if s is not None]
    if video_allowed is None and not frame_sets:
        return None

    if frame_sets:
        combined = frame_sets[0]
        for s in frame_sets[1:]:
            combined = combined & s
        if video_allowed is None:
            return combined
        return {(v, f) for v, f in combined if v in video_allowed}

    row_pos = _load_dense_row_pos()
    assert video_allowed is not None
    return {(vid, fi) for vid, fi in row_pos if vid in video_allowed}


def search_dense(
    query: str,
    mode: str,
    top_k: int = 100,
    ocr_text: str | None = None,
    *,
    authors: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    keywords_any: list[str] | None = None,
    must_have_labels: list[str] | None = None,
    min_count: dict[str, int] | None = None,
    spatial_boxes: list[dict] | None = None,
    spatial_op: str = "and",
    audio_mentions: list[dict] | None = None,
    log=None,
) -> pd.DataFrame:
    """mode in DENSE_MODES ("siglip"/"pe_core"/"beit3"/"rrf"). ocr_text: hard-filter chu tren
    man hinh (giong tier1_filter.by_text) - None = khong loc, "" cung coi nhu None.
    authors/date_from/date_to/keywords_any: loc video (tier1_filter.by_metadata, dung chung
    voi bo BTC vi la du lieu video-level). must_have_labels/min_count: loc theo Object toan cuc
    (khong xet vi tri - OWLv2 closed-set, xem _object_candidates). spatial_boxes: khung OCR/
    Object CO vi tri tu canvas (xem _spatial_box_candidates + app.py). spatial_op: "and" (frame
    phai khop DU moi khung) hoac "or" (khop BAT KY khung nao). audio_mentions: [{"term": str}]
    tu LLM (query_planner.py) - soft-boost theo loi noi GAN frame (xem _audio_mention_boost).
    log: StepLog tuy chon (xem steplog.py) - ghi lai buoc chung cat query (query_distill.py) de debug."""
    if mode not in DENSE_MODES:
        raise ValueError(f"mode phai la 1 trong {DENSE_MODES}, nhan '{mode}'")
    candidates = _combine_candidates(
        authors=authors, date_from=date_from, date_to=date_to, keywords_any=keywords_any,
        must_have_labels=must_have_labels, min_count=min_count, ocr_text=ocr_text,
        spatial_boxes=spatial_boxes, spatial_op=spatial_op,
    )

    # 2026-08-15 (theo yeu cau nguoi dung: "cac he thong khac tra ve DU top_k, du co loc hay
    # khong" - trc do da lam ban "all-or-nothing" cho truong hop 0 ket qua, GIO MO RONG thanh
    # BACKFILL TUNG PHAN: bo loc cung ra IT HON top_k (ke ca 0) -> LUON tra ve DU top_k bang
    # cach bu them ket qua KHONG qua loc (xep hang CLIP thuan) cho phan con thieu. Danh dau
    # tung DONG bang cot "is_backfill" (KHONG phai ca ket qua) de UI biet dong nao la khop loc
    # THAT, dong nao la bu them - gap-honesty, khong am tham danh lua nguoi dung.
    hard_filter_active = candidates is not None

    # 2026-08-15 (theo yeu cau nguoi dung): chung cat/dich query 1 LAN duy nhat truoc khi encode
    # - dung CHUNG cho ca 3 model (siglip/pe_core/beit3) thay vi tung model tu dich rieng. Xem
    # query_distill.py de biet ly do (PE-Core/BEiT-3 bat buoc tieng Anh, SigLIP2 dung chung
    # duong tieng Anh de don gian hoa + cau goc hay "van ve" lam loang tin hieu thi giac).
    from query_distill import distill_query

    if log:
        with log.timed("Chưng cất query (dịch + rút gọn cho embedding)") as set_detail:
            distilled = distill_query(query)
            set_detail(f'Gốc: "{query}"  \nChưng cất: "{distilled}"')
    else:
        distilled = distill_query(query)

    # 2026-08-15 (theo yeu cau nguoi dung: khung ve tay khong the chinh xac) - khung CO vi tri
    # gio la SOFT BOOST (khong loai frame), can pool RONG HON top_k de con "cua" ma sap xep lai
    # sau khi cong diem, giong pattern Region-CLIP rerank. ASR audio_mentions cung la soft boost
    # - dung CHUNG 1 lan mo rong pool (khong mo rong 2 lan rieng) neu CA 2 cung co mat.
    boxes_with_region = [b for b in (spatial_boxes or []) if b.get("region") is not None]
    audio_mentions = audio_mentions or []
    needs_wide_pool = bool(boxes_with_region) or bool(audio_mentions)
    rank_top_k = top_k * 4 if needs_wide_pool else top_k

    if mode == "rrf":
        result = _rank_rrf(distilled, rank_top_k, candidates=candidates, log=log)
    else:
        result = _rank_single(distilled, mode, rank_top_k, candidates=candidates, log=log)

    if needs_wide_pool and not result.empty:
        result = result.copy()
        n_pool = len(result)

        if boxes_with_region:
            with (log.timed("Cộng điểm vị trí khung vẽ (soft boost)") if log else contextlib.nullcontext(lambda *_a, **_k: None)) as set_detail:
                pos_boost = _spatial_position_boost(result, spatial_boxes or [])
                result["position_boost"] = pos_boost
                result["score_before_position_boost"] = result["score"]
                result["score"] = result["score"] + SPATIAL_POSITION_BOOST_WEIGHT * pos_boost
                if log:
                    set_detail(f"{int((pos_boost > 0).sum())}/{n_pool} frame (trong pool mở rộng) "
                               f"có object/OCR khớp vị trí khung vẽ")

        if audio_mentions:
            with (log.timed("ASR audio_mentions (soft boost, lời nói)") if log else contextlib.nullcontext(lambda *_a, **_k: None)) as set_detail:
                audio_boost = _audio_mention_boost(result, audio_mentions)
                result["audio_boost"] = audio_boost
                result["score_before_audio_boost"] = result["score"]
                result["score"] = result["score"] + AUDIO_MENTION_BOOST_WEIGHT * audio_boost
                if log:
                    terms = [m.get("term") for m in audio_mentions]
                    set_detail(f"nhắc tới={terms}, {int((audio_boost > 0).sum())}/{n_pool} frame "
                               f"(trong pool mở rộng) có lời nói gần đó khớp")

        result = result.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)

    result["is_backfill"] = False  # cot luon ton tai, ke ca khi result rong (0 dong van co cot)

    # BACKFILL: chi khi CO loc cung (hard_filter_active) VA con thieu so voi top_k (ke ca 0) -
    # bu them ket qua CLIP thuan (khong qua loc) cho DU dung top_k, giong cach cac he thong
    # khac lam (luon tra ve dung so luong yeu cau).
    if hard_filter_active and len(result) < top_k:
        n_missing = top_k - len(result)
        if mode == "rrf":
            backfill_pool = _rank_rrf(distilled, top_k, candidates=None)
        else:
            backfill_pool = _rank_single(distilled, mode, top_k, candidates=None)
        if not result.empty:
            existing_keys = set(zip(result["video_id"], result["frame_id"].astype(int)))
            backfill_pool = backfill_pool[~backfill_pool.apply(
                lambda r: (r["video_id"], int(r["frame_id"])) in existing_keys, axis=1
            )]
        backfill_rows = backfill_pool.head(n_missing).copy()
        backfill_rows["is_backfill"] = True
        result = pd.concat([result, backfill_rows], ignore_index=True) if not result.empty else backfill_rows
        if log:
            log.add("Bộ lọc cứng", f"chỉ khớp {top_k - n_missing}/{top_k} kết quả — BỔ SUNG thêm "
                     f"{len(backfill_rows)} kết quả bằng CLIP thuần (BỎ QUA lọc cứng) cho đủ "
                     f"{top_k}, đánh dấu ở cột is_backfill", 0.0)

    return result
