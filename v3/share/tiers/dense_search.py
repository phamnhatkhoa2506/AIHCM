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

from app_flags import ModalTimeoutError, call_modal_with_timeout
from config import DENSE_DIR, DENSE_META_PATH, INDEX_DIR
from label_translate import resolve as resolve_label_vi
from local_text_encoders import ENCODERS
from tiers.tier1_filter import (
    DEFAULT_OCR_ALGORITHM,
    OCR_MATCH_ALGORITHMS,
    _flexible_word_coverage_score,
    _flexible_word_match,
    _ordered_words_match,
    _strip_accents,
    by_metadata,
    match_words,
)

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
    return pd.read_parquet(DENSE_META_PATH)


@lru_cache(maxsize=None)
def _load_dense_row_pos() -> dict[tuple, int]:
    meta = _load_dense_meta()
    return {(vid, int(fid)): i for i, (vid, fid) in enumerate(zip(meta["video_id"], meta["frame_idx"]))}


@lru_cache(maxsize=None)
def _shot_bounds_by_video() -> dict[str, list[tuple[int, int, int]]]:
    """video_id -> list[(shot_idx, frame_idx_min, frame_idx_max)] SAP XEP theo shot_idx -
    "frame_idx_min/max" la khoang cua CAC KEYFRAME DA TRICH trong shot do (khong phai frame
    dau/cuoi THAT cua shot, vi dense corpus chi trich mau thua, khong phai moi frame - xem
    get_shot_frame_range() duoi day, ham do MOI la noi "vao khe" 2 shot lien tiep de co ranh
    gioi LIEN TUC, khong co khoang trong giua 2 shot).

    2026-08-16 (theo yeu cau nguoi dung "index/keyframe_meta_all.jsonl chinh la gioi han shot
    theo keyframe de hien thi video, chu khong lay co dinh nhu dang lam"): dense_meta.parquet
    da co san shot_idx (parse THANG tu ten file shotNNNN_fFFFFFFFF.jpg, xem
    build_dense_embeddings.py::_parse_filename) - CUNG 1 quy uoc danh so shot voi
    keyframe_meta_all.jsonl (da doi chieu thu, khop 100% cho video test) nhung PHU 873/873
    video (jsonl chi co 574/873 - phu 1 phan, co le tu 1 lan chay thu truoc do) - dung
    dense_meta lam nguon CHINH, khong can doc them file jsonl rieng."""
    # 2026-08-20 (toi uu, theo yeu cau nguoi dung "xem trong dự án còn có chỗ nào đang bị lỗ
    # hổng tốc độ không" - do that qua audit chu dong: ham nay ton ~5s LAN DAU/phien, dung cho
    # nut "▶ Video" o KIS/Q&A): ban CU dung `.iterrows()` tren ket qua groupby - NOI TIENG cham
    # (tao 1 pandas Series moi MOI dong, ep kieu lai) du chi ~vai chuc nghin dong (so shot, KHONG
    # phai so frame). Fix: `.reset_index()` + zip() tren MANG numpy tho (nhu da lam cho OCR) -
    # do that: 5.05s -> 0.33s (~15 lan nhanh hon), KET QUA GIONG HET (da so sanh tung phan tu).
    meta = _load_dense_meta()
    out: dict[str, list[tuple[int, int, int]]] = {}
    grouped = meta.groupby(["video_id", "shot_idx"], sort=False)["frame_idx"].agg(["min", "max"]).reset_index()
    for vid, shot_idx, lo, hi in zip(
        grouped["video_id"].values, grouped["shot_idx"].values,
        grouped["min"].values, grouped["max"].values,
    ):
        out.setdefault(vid, []).append((int(shot_idx), int(lo), int(hi)))
    for vid in out:
        out[vid].sort(key=lambda t: t[0])
    return out


def get_shot_frame_range(video_id: str, frame_id: int) -> tuple[int, int] | None:
    """Tra (frame_idx_start, frame_idx_end) TREN TRUC VIDEO GOC cua SHOT chua `frame_id` -
    dung lam gioi han THAT cho video_clip.py (thay the fixed +-3s cu). "Vao khe" giua 2 shot
    lien tiep (start[i]..start[i+1]-1) de KHONG co khoang trong giua 2 shot du keyframe mau
    thua khong phu het frame that. Shot CUOI CUNG cua video: end = max frame_idx cua chinh no
    (khong co shot sau de vao khe). Tra None neu video khong co trong dense corpus."""
    shots = _shot_bounds_by_video().get(video_id)
    if not shots:
        return None
    for i, (_shot_idx, lo, hi) in enumerate(shots):
        next_lo = shots[i + 1][1] if i + 1 < len(shots) else None
        end = (next_lo - 1) if next_lo is not None and next_lo > lo else hi
        if lo <= frame_id <= max(end, hi):
            return lo, max(end, hi)
    # frame_id ngoai moi khoang da biet (vd truoc shot dau/sau shot cuoi do sai so nho) -
    # fallback ve shot GAN NHAT.
    closest = min(shots, key=lambda t: min(abs(frame_id - t[1]), abs(frame_id - t[2])))
    return closest[1], closest[2]


@lru_cache(maxsize=None)
def _load_dense_index(model: str):
    """CHI dung o CHE DO LOCAL (AIC_LOCAL_MODELS=1) - xem _rank_single duoi day. Mac dinh
    (khong set bien nay) KHONG goi ham nay nua, xep hang chay qua server Modal
    aic2026-dense-index thay vi nap matrix.npy + faiss.index (~7.4GB ca 3 model) vao RAM local -
    2026-08-16, theo yeu cau nguoi dung "may minh moi lan chay len no chiem gan het RAM"."""
    matrix = np.load(DENSE_DIR / f"{model}_matrix.npy")
    index = faiss.read_index(str(DENSE_DIR / f"{model}_faiss.index"))
    return matrix, index


DENSE_INDEX_APP_NAME = "aic2026-dense-index"
DENSE_INDEX_CLASS_NAME = "DenseIndexServer"


@lru_cache(maxsize=1)
def _dense_index_server():
    import modal

    Server = modal.Cls.from_name(DENSE_INDEX_APP_NAME, DENSE_INDEX_CLASS_NAME)
    return Server()


@lru_cache(maxsize=1)
def _load_objects_index() -> pd.DataFrame | None:
    if not OBJECTS_INDEX_PATH.exists():
        return None
    return pd.read_parquet(OBJECTS_INDEX_PATH)


# ============================================================ Region-CLIP rerank (2026-08-15)
# 2 CHE DO qua bien moi truong AIC_LOCAL_MODELS (giong local_text_encoders.py, mac dinh TAT):
#   - TAT (mac dinh): goi .remote() toi server Modal nhe luon giu am (aic2026-region-rerank,
#     xem offline/modal_infra/region_rerank_app.py) - may local CHI gui (video_id, frame_id) +
#     nhan + cau thuoc tinh (rat nhe), khong giu region_embeddings_siglip.npy (5.3GB) trong RAM.
#   - BAT (AIC_LOCAL_MODELS=1): nap 5.3GB embedding + scope_detections_cache.parquet TRUC TIEP
#     tren may (README.md "Chay model local") - cham/an RAM lan dau phien Streamlit, nhung
#     khong can Modal/mang cho moi query.
REGION_RERANK_APP_NAME = "aic2026-region-rerank"
REGION_RERANK_CLASS_NAME = "RegionRerankServer"
REGION_CLIP_WEIGHT = 0.5  # cung gia tri voi ban BTC cu (query_planner.py) - da tune truoc do

REGION_SCOPE_CACHE_PATH = DENSE_DIR / "shards" / "_scope_detections_cache.parquet"
REGION_EMB_PATH = DENSE_DIR / "region_embeddings_siglip.npy"
REGION_EMB_IDS_PATH = DENSE_DIR / "region_embeddings_siglip_detection_ids.npy"


def _local_mode() -> bool:
    """Bien moi truong AIC_LOCAL_MODELS (giong local_text_encoders.py) - dung CHUNG cho ca
    Region-CLIP rerank VA dense index (Tier 2 vector search, xem _load_dense_index/_rank_single
    duoi day) - 1 cong tac DUY NHAT bat/tat toan bo "chay local thay vi Modal"."""
    import os

    return os.environ.get("AIC_LOCAL_MODELS", "0").strip().lower() in ("1", "true", "yes")


@lru_cache(maxsize=1)
def _region_rerank_server():
    import modal

    Server = modal.Cls.from_name(REGION_RERANK_APP_NAME, REGION_RERANK_CLASS_NAME)
    return Server()


class _LocalRegionRerank:
    """Y HET region_rerank_app.py::RegionRerankServer.load()/rerank(), chi bo wrapper Modal -
    doc thang tu index/dense/ local thay vi Volume "aic2026-region-index" (xem README.md)."""

    def __init__(self) -> None:
        import os

        from transformers import AutoModel, AutoProcessor

        from config import _V3_ROOT

        os.environ.setdefault("HF_HOME", str(_V3_ROOT / ".cache" / "huggingface"))
        for p in (REGION_SCOPE_CACHE_PATH, REGION_EMB_PATH, REGION_EMB_IDS_PATH):
            if not p.exists():
                raise RuntimeError(
                    f"Thiếu {p} — chạy offline/build_dense_region_embeddings_shard.py trước "
                    f"(xem README.md mục Region-CLIP)."
                )
        from local_text_encoders import SIGLIP_MODEL_NAME  # cung ten model, tranh lech ban

        self._siglip_model = AutoModel.from_pretrained(SIGLIP_MODEL_NAME).eval()
        self._siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
        self._scope_df = pd.read_parquet(REGION_SCOPE_CACHE_PATH)
        self._vecs = np.load(REGION_EMB_PATH)
        ids = np.load(REGION_EMB_IDS_PATH)
        self._id_to_row = {int(did): i for i, did in enumerate(ids)}

    def _encode_text(self, text: str):
        import torch

        inputs = self._siglip_processor(text=[text], return_tensors="pt", padding="max_length", truncation=True)
        with torch.no_grad():
            feats = self._siglip_model.get_text_features(**inputs)
        if hasattr(feats, "pooler_output"):
            feats = feats.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].numpy()

    def rerank(self, frame_keys: list[list], entity_labels: list[str], attribute_text: str) -> list[float]:
        text_vec = self._encode_text(attribute_text)
        frame_key_set = {(vid, int(fid)) for vid, fid in frame_keys}
        mi = pd.MultiIndex.from_arrays([self._scope_df["video_id"], self._scope_df["frame_idx"]])
        scoped = self._scope_df[mi.isin(frame_key_set) & self._scope_df["label"].isin(entity_labels)]

        frame_best: dict[tuple[str, int], float] = {}
        for detection_id, video_id, frame_idx in zip(scoped["detection_id"], scoped["video_id"], scoped["frame_idx"]):
            row_pos = self._id_to_row.get(int(detection_id))
            if row_pos is None:
                continue
            sim = float(np.dot(self._vecs[row_pos], text_vec))
            key = (video_id, int(frame_idx))
            if sim > frame_best.get(key, -1.0):
                frame_best[key] = sim
        return [frame_best.get((vid, int(fid)), 0.0) for vid, fid in frame_keys]


@lru_cache(maxsize=1)
def _local_region_rerank() -> _LocalRegionRerank:
    return _LocalRegionRerank()


REGION_CLIP_MIN_DYNAMIC_RANGE = 0.10  # 2026-08-17 (theo yeu cau nguoi dung, sau khi phat hien
# qua case that "con chó mặc áo màu đỏ" - xem hoi thoai): SigLIP2 zero-shot doi khi KHONG PHAN
# BIET DUOC thuoc tinh tren 1 loai entity (vd "cho mac ao mau do" - khai niem "mac do" hoc chu
# yeu tu du lieu NGUOI mac quan ao, khong phai CHO/dong vat) - luc do region_score cua CAC ung
# vien DANG XEP CAO (top CLIP) deu na ná nhau (do that: 4 frame top-CLIP dao dong CHI 0.120-
# 0.146, chenh lech ~0.026, du frame ro rang KHONG mac gi do van diem GAN BANG frame o trong xe
# do choi mau do) - cong mot the LUC nay = xao lai thu tu CLIP goc (von da hop ly) dua tren
# NHIEU gan nhu ngau nhien, khong phai tin hieu that.
#
# 2026-08-17 (TUNE LAN 2, nguoi dung chon "Tăng ngưỡng lên ~0.10-0.12"): nguong 0.05 ban dau
# (tinh tren CA pool 36-48 frame co match) KHONG bat duoc case goc - do that: 0.095 tren toan
# pool (co the do 1-2 frame outlier o cuoi pool keo range len), TRONG KHI nhom THAT SU canh
# tranh top-12 cuoi cung (da xep cao boi CLIP) van dinh sat nhau ~0.026 nhu do ban dau. Nang len
# 0.10 la fix DON GIAN (khong doi cach tinh, chi doi nguong) theo dung lua chon nguoi dung -
# danh doi da biet: co the tat rerank oan voi query co tin hieu that nhung YEU (range 0.05-0.10)
# - chap nhan duoc, uu tien AN TOAN (khong xao ket qua theo nhieu) hon ep dung moi truong hop.
#
# Fix: neu (max-min) cua CAC DIEM >0 cua 1 thuoc tinh < nguong nay, coi nhu model KHONG DU KHA
# NANG phan biet cho thuoc tinh do - BO QUA (khong cong vao region_score), giu nguyen thu tu Tang
# 2 cho phan do, thay vi cu ap dung mu quang. AP DUNG RIENG TUNG THUOC TINH (khong phai ca cum) -
# 1 cau co the co thuoc tinh phan biet duoc VA thuoc tinh khong phan biet duoc cung luc.


def apply_region_clip_rerank(
    results: pd.DataFrame, attributes: list[dict], top_k: int, log=None
) -> pd.DataFrame:
    """Rerank theo thuoc tinh (vd "nguoi mac ao dai mau tim") - CHI dung SigLIP2 (theo yeu cau
    nguoi dung 2026-08-15: khong chay Region-CLIP cho ca 3 model, bat ke search mode dang dung
    la gi). Tinh toan THAT SU chay tren server Modal (aic2026-region-rerank) - ham nay chi goi
    .remote() va gop ket qua, khong tu tinh cosine/loc data nua.

    2026-08-17: MOI thuoc tinh duoc kiem tra dynamic range TRUOC khi dua vao trung binh - xem
    REGION_CLIP_MIN_DYNAMIC_RANGE."""
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
                if _local_mode():
                    scores = _local_region_rerank().rerank(frame_keys, entity_labels, attribute_text)
                else:
                    # 2026-08-20 (theo yeu cau nguoi dung, tiep tuc sau timeout NIM) - .remote()
                    # khong co timeout, co the treo VO HAN - ModalTimeoutError (con so voi
                    # RuntimeError) van duoc except Exception ben duoi bat, GIU NGUYEN hanh vi
                    # graceful-degrade cu (coi nhu khong khop, khong lam sap ca query).
                    scores = call_modal_with_timeout(
                        _region_rerank_server().rerank, frame_keys, entity_labels, attribute_text,
                        context="Region-CLIP rerank",
                    )
            except Exception as e:
                scores = [0.0] * len(results)
                if log:
                    set_detail(f"LỖI gọi server rerank: {type(e).__name__} {str(e)[:120]} — coi như không khớp")
                per_attr_scores.append(scores)
                continue

            # 2026-08-17 (sua sau khi TU test phat hien sai): CHI tinh dynamic range tren cac
            # diem >0 (frame THAT SU co object khop nhan, duoc SigLIP2 cham diem that) - frame
            # KHONG co object nao khop bi gan 0.0 "mac dinh" (khong phai SigLIP2 cham), tinh ca
            # nhung diem 0.0 nay vao se lam range trong RONG GIA TAO (0.0 vs ~0.15 nhin "phan
            # biet duoc" trong khi giua CAC FRAME THAT SU CO CHO, diem van dinh sat nhau ~0.02-
            # 0.03 - dung case that phat hien qua hoi thoai "con chó mặc áo màu đỏ": 4 frame
            # DEU co Dog van chi dao dong 0.120-0.146).
            nonzero_scores = [s for s in scores if s > 0]
            dyn_range = (max(nonzero_scores) - min(nonzero_scores)) if len(nonzero_scores) >= 2 else 0.0
            if dyn_range < REGION_CLIP_MIN_DYNAMIC_RANGE:
                # BUG THAT (2026-08-17, xem REGION_CLIP_MIN_DYNAMIC_RANGE) - SigLIP2 khong phan
                # biet duoc thuoc tinh nay tren cac ung vien -> BO QUA, KHONG dua vao trung binh
                # (tuong duong 0.0 het lam region_score = 0 cho MOI frame VOI thuoc tinh nay,
                # giu nguyen thu tu Tang 2 cho phan do).
                per_attr_scores.append([0.0] * len(results))
                if log:
                    set_detail(f"nhãn={entity_labels} — BỎ QUA rerank (dynamic range={dyn_range:.3f} "
                               f"< {REGION_CLIP_MIN_DYNAMIC_RANGE} — model không đủ phân biệt thuộc "
                               f"tính này, giữ nguyên thứ tự CLIP gốc thay vì xáo theo nhiễu)")
                continue

            per_attr_scores.append(scores)
            n_match = sum(1 for s in scores if s > 0)
            if log:
                set_detail(f"nhãn={entity_labels}, {n_match}/{len(results)} frame có object khớp "
                           f"(dynamic range={dyn_range:.3f})")

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


def _cluster_ocr_rows_by_column(rows: list[dict]) -> list[dict]:
    """Gom cac box OCR CUNG 1 frame thanh CUM theo GIAO VUNG-X (2 box cung "cot"/cung 1 bien
    vat ly neu khoang [xmin,xmax] cua chung GIAO NHAU) - roi tra ve LAI list rows theo dung
    THU TU DOC: trong 1 cum sap theo ymin (tren->duoi), cac cum sap theo xmin nho nhat (trai->
    phai).

    BUG THAT (2026-08-17, nguoi dung phat hien qua vi du "TTTM QUỐC THÁI" - 1 bien KHAC nam
    CANH bien "CỬA KHẨU LONG BÌNH" - hoi "TTTM" co ymin nam GIUA ymin cua "CUAKHAU" va
    "LONGBINH" (2 dong CUNG 1 bien, xep chong len nhau theo chieu doc) chi vi 2 bien nam
    CHENH LECH do cao 1 chut): sap THO chi theo (ymin,xmin) nhu ban dau se ZIGZAG qua lai
    GIUA 2 bien canh nhau, chen "TTTM QUOCTHAI" (bien phai) vao GIUA "CUAKHAU" va "LONGBINH"
    (bien trai) trong danh sach tu gop - lam hong diem "do gon" cua OCR match score (frame co
    bien dung lai bi tinh diem THAP hon frame khac vi bi bien ben canh "lam nhieu"). Fix:
    GOM CUM truoc theo giao vung-x (dac trung dang tin cay hon ymin don le de phan biet "cung 1
    bien" vs "2 bien canh nhau" - 2 dong CUNG 1 bien luon co x-range GAN GIONG NHAU du le nao,
    2 bien KHAC nhau hau nhu luon co x-range TACH BIET), roi moi noi thu tu doc GIUA cac cum."""
    n = len(rows)
    if n <= 1:
        return rows
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            ax0, ax1 = rows[i]["xmin"], rows[i]["xmax"]
            bx0, bx1 = rows[j]["xmin"], rows[j]["xmax"]
            if min(ax1, bx1) - max(ax0, bx0) > 0:  # giao vung-x (khong xet ymin o day)
                union(i, j)

    groups: dict[int, list[dict]] = {}
    for idx in range(n):
        groups.setdefault(find(idx), []).append(rows[idx])
    clusters = list(groups.values())
    for c in clusters:
        c.sort(key=lambda r: r["ymin"])
    clusters.sort(key=lambda c: min(r["xmin"] for r in c))
    return [r for c in clusters for r in c]


class _OCRRowsByFrame:
    """Thay the dict[(video_id,frame_idx)] -> list[dict] cu (2026-08-20, toi uu lan 2 - xem
    docstring _ocr_columnar() ben duoi cho ly do/so lieu do dac). CHI materialize list[dict]
    (dang cac ham khac trong file nay dang mong doi) KHI THAT SU truy cap (.get() 1 frame cu
    the, hoac duyet .items()) - du lieu goc la MANG numpy DA SAP XEP, khong giu san 1.33 TRIEU
    dict Python trong bo nho. API tuong thich nguoc du (chi co .get()/.items(), dung du cho
    3 noi goi trong file nay - khong phai dict that nen KHONG ho tro [] / len() / v.v...)."""
    __slots__ = ("_index", "_text", "_ymin", "_xmin", "_ymax", "_xmax")

    def __init__(self, index, text_arr, ymin_arr, xmin_arr, ymax_arr, xmax_arr):
        self._index = index
        self._text = text_arr
        self._ymin = ymin_arr
        self._xmin = xmin_arr
        self._ymax = ymax_arr
        self._xmax = xmax_arr

    def _rows(self, start: int, end: int) -> list[dict]:
        return [
            {"text_norm": self._text[i], "ymin": self._ymin[i], "xmin": self._xmin[i],
             "ymax": self._ymax[i], "xmax": self._xmax[i]}
            for i in range(start, end)
        ]

    def get(self, key, default=None):
        bounds = self._index.get(key)
        if bounds is None:
            return default
        return self._rows(*bounds)

    def items(self):
        for key, (start, end) in self._index.items():
            yield key, self._rows(start, end)


@lru_cache(maxsize=1)
def _ocr_columnar():
    """Doc ocr_text.parquet 1 LAN, tra ve (index, video_id_arr, frame_idx_arr, text_arr,
    words_arr, ymin_arr, xmin_arr, ymax_arr, xmax_arr) - MANG numpy DA SAP XEP theo
    (video_id, frame_idx, ymin, xmin) (sort_values 1 lan, vectorized/C-level trong pandas),
    `index`: dict (video_id,frame_idx) -> (start,end) SLICE vao cac mang tren (tim ranh gioi
    group BANG so sanh mang numpy - KHONG loop Python tren tung dong). `words_arr[i]` = KET QUA
    SAN cua `text_arr[i].split()` (xem 2026-08-20 lan 2 duoi day).

    2026-08-20 (toi uu LAN 1, theo yeu cau nguoi dung "thí nghiệm trước khi đưa vào code...
    thử để xem hiệu quả rồi ta gắn vào sau" - da THI NGHIEM RIENG truoc, KHONG dung code that
    cho toi khi xac nhan dung+nhanh): ban truoc (zip+defaultdict, van tao list[dict] CHO MOI
    trong 1.33 TRIEU dong khi build) chi nhanh hon ban goc ~31% (14.35s -> ~8.6-9.9s) - VAN con
    cham vi van phai tao ngan ay dict Python. Cach nay (khong tao dict NAO khi build INDEX, chi
    index+mang) - phan BUILD INDEX rieng chi ~0.55-0.82s. Da kiem tra dung TREN TOAN BO 343,015
    frame (so voi ban dict cu) - 0 frame lech.

    2026-08-20 (toi uu LAN 2, phat hien khi do lai FULL _ocr_frame_words(): build index nhanh
    (~0.8s) nhung buoc GOP TU theo frame ben duoi VAN goi `.split()` tung dong trong loop Python
    - rieng buoc nay ton ~11.6s, khien tong the CHUA nhanh nhieu nhu ky vong). Fix: tach `.split()`
    ra khoi vong lap, dung `pandas.Series.str.split()` (vectorized, mot lan cho CA 1.33 TRIEU
    dong) NGAY TAI DAY - do that: ~1.9-2.0s (so voi ~4.5s neu goi .split() tung dong trong list
    comprehension, ~2.2 lan nhanh hon) - _ocr_frame_words() ben duoi chi con GHEP (extend) cac
    list DA TACH SAN, khong con parse chuoi nua.

    Sap xep THEO CA (ymin,xmin) (khong chi video_id/frame_idx) ngay tu buoc nay - de
    _ocr_frame_words() ben duoi doc THANG tu mang theo dung THU TU vi tri (khong can sort()
    rieng nua, tranh regress hanh vi cu "GOP LAI theo thu tu ymin,xmin")."""
    if not OCR_TEXT_PATH.exists():
        empty = np.array([])
        return {}, empty, empty, empty, empty, empty, empty, empty, empty
    df = pd.read_parquet(OCR_TEXT_PATH)
    df_sorted = df.sort_values(["video_id", "frame_idx", "ymin", "xmin"], kind="mergesort").reset_index(drop=True)
    vid_arr = df_sorted["video_id"].values
    fidx_arr = df_sorted["frame_idx"].values.astype(np.int64)
    text_arr = df_sorted["text_norm"].values
    words_arr = df_sorted["text_norm"].str.split().values  # vectorized, xem docstring "LAN 2"
    ymin_arr = df_sorted["ymin"].values
    xmin_arr = df_sorted["xmin"].values
    ymax_arr = df_sorted["ymax"].values
    xmax_arr = df_sorted["xmax"].values

    # Ranh gioi group: vi tri MA key (video_id,frame_idx) DOI so voi dong truoc - vectorized
    # (so sanh mang numpy, C-level), KHONG loop Python tren tung dong trong 1.33 TRIEU dong.
    change = np.ones(len(df_sorted), dtype=bool)
    if len(df_sorted) > 0:
        change[1:] = (vid_arr[1:] != vid_arr[:-1]) | (fidx_arr[1:] != fidx_arr[:-1])
    boundaries = np.flatnonzero(change)
    boundaries = np.append(boundaries, len(df_sorted))
    starts, ends = boundaries[:-1], boundaries[1:]
    # vong lap con lai (neu co) chi chay tren SO FRAME (~343k), khong phai so DONG (~1.33M) -
    # it hon ~4 lan, va chi tao tuple/int (khong tao dict) nen re hon nhieu.
    keys = list(zip(vid_arr[starts].tolist(), fidx_arr[starts].tolist()))
    index = dict(zip(keys, zip(starts.tolist(), ends.tolist())))

    return index, vid_arr, fidx_arr, text_arr, words_arr, ymin_arr, xmin_arr, ymax_arr, xmax_arr


@lru_cache(maxsize=1)
def _ocr_frame_words() -> dict[tuple[str, int], list[str]]:
    """(video_id, frame_idx) -> DANH SACH TU cua TAT CA cac box OCR trong frame do, GOP LAI
    theo thu tu (ymin,xmin) THO (KHONG cluster theo bien vat ly - xem
    _ocr_frame_words_clustered() ben duoi cho ban CHINH XAC HON, chi tinh LAZY cho 1 frame khi
    can, vi cluster CA corpus (300k+ frame) qua cham - da do timeout that khi thu eager).

    Dung cho HARD FILTER (_ocr_candidates/_ocr_box_candidates) - KHONG can chinh xac tuyet doi
    ve thu tu giua 2 bien canh nhau, vi _flexible_word_match da KHOAN DUNG tu chen giua BAT
    KE co lien quan hay khong - sort tho van cho ket qua BOOLEAN dung, chi anh huong DIEM GON
    (compactness) thoi, ma diem do duoc tinh RIENG qua ham clustered ben duoi tren tap nho da
    loc, khong dung ham nay.

    BUG THAT (2026-08-17, nguoi dung phat hien qua vi du "CỬA KHẨU LONG BÌNH" bi PaddleOCR chia
    thanh 2 box RIENG BIET - "CUAKHAU" 1 box, "LONG BINH" 1 box khac): code CU khop TUNG DONG
    OCR DOC LAP (moi dong 1 string rieng), khong bao gio ghep 2 box lai - query "CỬA KHẨU LONG
    BÌNH" ra 0 ket qua DU frame co ca 2 box dung 100%, vi khong dong RIENG LE nao chua du ca
    cum. Fix: GOP het cac box CUNG 1 frame thanh 1 danh sach tu duy nhat truoc khi khop, dung
    _flexible_word_match() (xem tier1_filter.py) de vua khoan dung box khac chen giua, vua
    khoan dung ranh gioi tu dinh/lech NGAY TRONG 1 box.

    2026-08-20 (toi uu LAN 2): doc THANG tu _ocr_columnar() (da SAP XEP san theo ymin,xmin VA
    da tach .split() san qua words_arr) - KHONG con di qua _ocr_rows_by_frame() (se tao
    list[dict] cho MOI frame, dung "duong cham" cu), KHONG con can sorted() rieng (du lieu da
    dung thu tu tu buoc sort chung), va KHONG con goi .split() trong vong lap (da tach vectorized
    1 lan trong _ocr_columnar(), xem docstring "LAN 2" o do) - chi con GHEP (extend) cac list
    tu DA TACH SAN."""
    index, _vid, _fidx, _text_arr, words_arr, *_rest = _ocr_columnar()
    out: dict[tuple[str, int], list[str]] = {}
    for key, (start, end) in index.items():
        words: list[str] = []
        for w in words_arr[start:end]:
            words.extend(w)
        out[key] = words
    return out


@lru_cache(maxsize=1)
def _ocr_rows_by_frame() -> _OCRRowsByFrame:
    """(video_id, frame_idx) -> list[{"text_norm","ymin","xmin","ymax","xmax"}] (qua .get()/
    .items(), xem _OCRRowsByFrame) - du lieu THO de _ocr_frame_words_clustered() cluster LAZY
    tung frame 1 (KHONG groupby toan bo corpus - qua cham, xem docstring _ocr_frame_words).

    2026-08-20 (toi uu LAN 2, xem docstring _ocr_columnar() cho so lieu do dac day du): chi la
    1 wrapper NHE quanh _ocr_columnar() (da cache rieng) - KHONG tu doc parquet, KHONG tu build
    gi them o day."""
    index, _vid, _fidx, text_arr, _words_arr, ymin_arr, xmin_arr, ymax_arr, xmax_arr = _ocr_columnar()
    return _OCRRowsByFrame(index, text_arr, ymin_arr, xmin_arr, ymax_arr, xmax_arr)


def _ocr_frame_words_clustered(video_id: str, frame_idx: int) -> list[str]:
    """Ban CHINH XAC cua danh sach tu 1 frame - CLUSTER theo bien vat ly (xem
    _cluster_ocr_rows_by_column) truoc khi noi thu tu doc, tranh zigzag giua 2 bien canh nhau
    lam sai diem "do gon". CHI tinh cho 1 frame/lan (lazy) - dung trong
    apply_ocr_match_quality_boost() tren tap KET QUA DA LOC (nho, ~top_k*4 dong), khong bao
    gio goi cho ca corpus."""
    rows = _ocr_rows_by_frame().get((video_id, frame_idx))
    if not rows:
        return []
    ordered = _cluster_ocr_rows_by_column(rows)
    words: list[str] = []
    for r in ordered:
        words.extend(r["text_norm"].split())
    return words


@lru_cache(maxsize=1)
def _ocr_frame_glued() -> dict[tuple[str, int], str]:
    """(video_id, frame_idx) -> TAT CA tu cua frame do NOI LIEN (khong khoang trang) - dung lam
    LOC SO BO RE (substring check, C-level, cuc nhanh) TRUOC khi chay thuat toan khop day du
    (_flexible_word_match) - xem docstring _ocr_prefilter_candidates."""
    return {key: "".join(words) for key, words in _ocr_frame_words().items()}


def _ocr_prefilter_candidates(needle_words: list[str]) -> list[tuple[str, int]]:
    """BUG THAT (2026-08-17, nguoi dung phat hien qua step log: "Lọc thô" ton 65s du CHI CO
    7 ung vien!): _ocr_candidates/_ocr_box_candidates CU chay thuat toan khop DAY DU
    (_flexible_word_match, vong lap long nhau, uu tien do CHINH XAC hon toc do) tren CA
    343,015 frame - da do rieng: ~50s/khung OCR ngay ca khi cache da am, vi thuat toan nay
    khong re (dac biet voi frame co nhieu chu, vd frame chay chu tin tuc co toi 357 tu).

    Fix: loc so bo BANG 1 substring check DON GIAN (tu Python `in`, chay o C-level, nhanh hon
    thuat toan khop day du HANG TRAM LAN) truoc - CHI frame nao co it nhat 1 tu needle xuat
    hien (dang chuoi con, trong ban NOI LIEN cua frame) moi duoc dua vao thuat toan khop day
    du. AN TOAN (khong mat ket qua dung): vi cac tu OCR khac nhau chen GIUA cac tu needle
    (truong hop da biet, xem _flexible_word_match_score) chi chen o RANH GIOI TU, khong bao gio
    "cat ngang" giua cac ky tu CUA 1 tu needle - nen neu 1 frame THAT SU khop, moi tu needle
    (it nhat 1) VAN phai xuat hien nguyen ven duoi dang chuoi con trong ban noi lien cua frame
    do, cho du bi tu khac chen o 2 ben."""
    if not needle_words:
        return list(_ocr_frame_words().keys())
    probe = needle_words[0] if len(needle_words[0]) >= len(needle_words[-1]) else needle_words[-1]
    return [key for key, glued in _ocr_frame_glued().items() if probe in glued]


def _ocr_match_candidate_keys(needle_words: list[str], algorithm: str) -> set[tuple[str, int]]:
    """Danh sach (video_id, frame_idx) khop needle_words theo `algorithm` da chon (xem
    tier1_filter.OCR_MATCH_ALGORITHMS). CHI dung prefilter substring (_ocr_prefilter_candidates)
    khi algorithm="flexible" - day la thuat toan DUY NHAT khop ky tu CHINH XAC tung tu, nen 1
    tu needle chac chan con nguyen ven dang chuoi con neu that su khop (xem docstring
    _ocr_prefilter_candidates). Cac thuat toan MEM DEO hon (rapidfuzz: cho phep loi ky tu;
    alignment: cho phep sim < 1.0 giua 2 tu) co the khop dung ma KHONG co tu needle nao xuat
    hien y nguyen - dung prefilter voi chung se MAT ket qua dung (false negative), nen phai quet
    toan bo cac frame co OCR (cham hon flexible, nhung la danh doi ro rang khi nguoi dung CHU
    DONG chon thuat toan nay, khong phai mac dinh)."""
    frame_words = _ocr_frame_words()
    if algorithm == "flexible":
        candidate_keys = _ocr_prefilter_candidates(needle_words)
    else:
        candidate_keys = list(frame_words.keys())
    return {key for key in candidate_keys if match_words(algorithm, needle_words, frame_words[key])[0]}


def _ocr_candidates(ocr_text: str, algorithm: str = DEFAULT_OCR_ALGORITHM) -> set[tuple[str, int]] | None:
    """Tra ve set (video_id, frame_idx) co chu KHOP ocr_text (word-boundary, khong phan biet
    dau, GOP tat ca box CUNG 1 frame - xem _ocr_frame_words), scope o OCR_TEXT_PATH cua bo
    dense. None neu chua co du lieu OCR dense (chua chay xong build_dense_ocr_index.py).
    algorithm: key trong tier1_filter.OCR_MATCH_ALGORITHMS (xem _ocr_match_candidate_keys ve
    danh doi toc do giua cac thuat toan)."""
    if not OCR_TEXT_PATH.exists():
        return None
    needle_words = _strip_accents(ocr_text).split()
    if not needle_words:
        return set(_ocr_frame_words().keys())
    return _ocr_match_candidate_keys(needle_words, algorithm)


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


# ============================================================ ASR HARD FILTER (2026-08-20)
# 2026-08-20 (theo yeu cau nguoi dung: "tích hợp search theo ASR... đảm bảo những câu như...
# nặng đến 211kg... mà embedding model thường không thấy") - TRUOC DAY ASR CHI dung lam SOFT
# BOOST (_audio_mention_boost, +diem nho, KHONG loai frame nao - xem audio_mentions o tren) -
# khong du manh cho case nhu "211kg" (1 con so CU THE nam trong lop tin CLIP khong "thay" duoc,
# can LOC CUNG giong OCR chu khong chi cong diem nhe). Them 1 duong LOC CUNG THEO LOI NOI moi,
# SONG SONG voi OCR (asr_text param, xem search_dense/_combine_candidates duoi day) - nguoi
# dung tu dien (giong o Loc chu OCR), KHONG tu dong (LLM audio_mentions van la duong RIENG,
# soft, giu nguyen khong doi).
@lru_cache(maxsize=1)
def _frame_idx_by_video() -> dict[str, np.ndarray]:
    """video_id -> mang frame_idx (dense_meta) DA SAP XEP - dung de tim NHANH (bisect, O(log n))
    cac frame nam trong 1 khoang [frame_idx_start, frame_idx_end] cua 1 doan ASR khop, khong
    can loc lai dense_meta (369k dong) moi lan goi. Chi ~873 video (video-level), KHONG can
    toi uu vector hoa nhu OCR (1.33 TRIEU dong) - groupby+loop o day ĐÃ đủ nhanh (do that
    <0.5s), khong dang chi phi ky thuat tuong tu."""
    meta = _load_dense_meta()
    out: dict[str, np.ndarray] = {}
    for vid, grp in meta.groupby("video_id", sort=False)["frame_idx"]:
        out[vid] = np.sort(grp.values)
    return out


def _asr_candidates(asr_text: str) -> set[tuple[str, int]] | None:
    """Tra ve set (video_id, frame_idx) - CAC FRAME DENSE nam trong khoang thoi gian cua 1 doan
    ASR co text_norm KHOP asr_text (ordered-substring, giong _audio_mention_boost) - None neu
    asr_text rong (khong loc) hoac chua co du lieu ASR dense.

    Khac OCR (khop TUNG FRAME rieng le): 1 doan ASR trai dai NHIEU frame (frame_idx_start..end)
    - MOI frame dense nam trong khoang do deu la ung vien hop le (nguoi noi cau do trong luc
    canh nao dang chieu tren man hinh, khong the biet CHINH XAC frame nao - giu CA khoang, giong
    nguyen tac ASR_CONTEXT_WINDOW_SECONDS cua soft-boost)."""
    if not asr_text or not asr_text.strip():
        return None
    asr = _load_dense_asr()
    if asr is None or asr.empty:
        return set()

    # 2026-08-20 (toi uu, do that qua benchmark: ban dau dung .apply(lambda...) - quet 47,585
    # dong BANG PYTHON LOOP moi lan goi (KHONG cache o day, khac OCR) - ton ~2s/query, se cong
    # don vao MOI lan search dung ASR filter. Fix: `text_norm` da SAN accent-stripped+lowercase
    # tu build_dense_asr_index.py (kiem tra that: text_norm == strip_accents(text_norm)) - dung
    # THANG `.str.contains()` (vectorized, C-level trong pandas) thay vi .apply(lambda) - do
    # that: ~0.05s/query (~40 lan nhanh hon). KHONG can _strip_accents(t) rieng cho tung dong
    # nua vi da chuan hoa san.
    needle = _strip_accents(asr_text)
    hit = asr[asr["text_norm"].str.contains(needle, regex=False, na=False)]
    if hit.empty:
        return set()

    # BUG THAT (2026-08-20, nguoi dung phat hien qua case that: "sân bay quốc tế" that su co
    # trong L25_V042 - text_norm khop 100% - nhung loc ra KHONG co video nay): dense corpus lay
    # mau THUA (~0.55-2.65s/frame TRUNG BINH, nhung co video/doan cu the THUA HON NHIEU o cac
    # canh tinh/shot dai) - do that CHINH XAC case nay: 2 frame dense gan nhat cua L25_V042
    # cach nhau toi 1440 frame (~57.6s o 25fps), trong khi doan ASR khop chi dai ~10s (frame_idx
    # 32068-32312) - KHONG CO frame dense nao roi dung vao khoang do -> match THAT SU nhung tra
    # ve 0 frame, "bien mat" khoi ket qua ma khong co dau hieu gi. Fix: neu KHONG co frame nao
    # nam CHAT trong [start,end], fallback ve frame dense GAN NHAT (truoc hoac sau, tuy cai nao
    # gan hon) - giong nguyen tac get_shot_frame_range() da lam (video/doan khong co frame dung
    # khop -> lay gan nhat, con hon KHONG co gi) - dam bao 1 doan ASR khop CHU KHONG BAO GIO
    # "bien mat" hoan toan chi vi lay mau thua.
    frame_idx_by_video = _frame_idx_by_video()
    candidates: set[tuple[str, int]] = set()
    for row in hit.itertuples(index=False):
        frames = frame_idx_by_video.get(row.video_id)
        if frames is None or len(frames) == 0:
            continue
        lo = int(np.searchsorted(frames, row.frame_idx_start, side="left"))
        hi = int(np.searchsorted(frames, row.frame_idx_end, side="right"))
        if lo < hi:
            for f in frames[lo:hi]:
                candidates.add((row.video_id, int(f)))
            continue
        # khong co frame nao nam CHAT trong khoang - fallback ve frame GAN NHAT (truoc/sau).
        mid = (row.frame_idx_start + row.frame_idx_end) / 2
        cand_positions = [p for p in (lo - 1, lo) if 0 <= p < len(frames)]
        if not cand_positions:
            continue
        best_pos = min(cand_positions, key=lambda p: abs(frames[p] - mid))
        candidates.add((row.video_id, int(frames[best_pos])))
    return candidates


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


def _ocr_box_candidates(text: str, algorithm: str = DEFAULT_OCR_ALGORITHM) -> set[tuple[str, int]]:
    """1 khung loai OCR (chu, khong dau) -> set (video_id, frame_idx) co chu do O DAU DO trong
    frame (KHONG xet vi tri nua - xem docstring o tren, GOP box cung frame - xem
    _ocr_frame_words). Tra set RONG (khong phai None) vi khung nguoi dung ve la rang buoc THAT,
    khong duoc coi nhu "khong loc gi" khi thieu du lieu."""
    if not OCR_TEXT_PATH.exists():
        return set()
    if not text:
        return set(_ocr_frame_words().keys())
    needle_words = _strip_accents(text).split()
    return _ocr_match_candidate_keys(needle_words, algorithm)


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


def _ocr_box_position_score_for_frame(
    text: str,
    region: tuple[float, float, float, float],
    video_id: str,
    frame_id: int,
    algorithm: str = DEFAULT_OCR_ALGORITHM,
) -> float:
    """IoU CAO NHAT giua vung nguoi dung ve va cac dong OCR (rieng le, khong gop) khop `text`
    trong 1 frame CU THE - dung lam SOFT BOOST (xem _spatial_position_boost), KHONG loai frame
    nao ca.

    BUG THAT (2026-08-17, nguoi dung phat hien qua step log: "Cộng điểm vị trí khung vẽ" ton
    101.7s!): ham CU (_ocr_box_position_scores, da xoa) doc LAI TOAN BO 1.3 trieu dong
    ocr_text.parquet TU DIA + quet match tren TOAN BO CORPUS - MOI LAN GOI, khong cache gi ca -
    2 khung OCR trong 1 query = 2 lan doc+quet full corpus. Fix: dung LAI _ocr_rows_by_frame()
    da cache san (xay dung 1 LAN/process, xem docstring) + CHI tinh cho 1 frame CU THE (goi lazy
    tu vong lap ket qua trong _spatial_position_boost, giong het pattern
    _ocr_match_quality_boost/_coverage_x_compactness) - khong con quet toan corpus nua."""
    rows = _ocr_rows_by_frame().get((video_id, frame_id))
    if not rows:
        return 0.0
    best = 0.0
    for r in rows:
        if text and not _ordered_words_match(text, r["text_norm"], algorithm):
            continue
        box = np.array([[r["ymin"], r["xmin"], r["ymax"], r["xmax"]]], dtype=float)
        iou = float(_box_iou_matrix(box, np.array([region], dtype=float))[0, 0])
        if iou > best:
            best = iou
    return best


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


def _spatial_position_boost(
    results: pd.DataFrame, spatial_boxes: list[dict], ocr_algorithm: str = DEFAULT_OCR_ALGORITHM
) -> np.ndarray:
    """Diem CONG THEM theo do khop VI TRI (IoU) cho tung khung CO ve vi tri - khung nguoi dung
    ve LECH van duoc GIU frame (0 diem, khong bi loai), chi khung khop tot moi duoc uu tien
    hon - xem hoi thoai 2026-08-15 (nguoi dung: "họ không thể vẽ chính xác")."""
    boxes_with_region = [b for b in spatial_boxes if b.get("region") is not None]
    if not boxes_with_region or results.empty:
        return np.zeros(len(results))
    boost = np.zeros(len(results))
    for box in boxes_with_region:
        if box["type"] == "ocr":
            # LAZY, tinh RIENG cho tung dong ket qua (khong con quet toan corpus - xem
            # docstring _ocr_box_position_score_for_frame).
            match = np.array([
                _ocr_box_position_score_for_frame(
                    box.get("text", ""), box["region"], r.video_id, int(r.frame_id), ocr_algorithm
                )
                for r in results.itertuples(index=False)
            ])
        else:
            scores_map = _object_box_position_scores(box.get("labels") or [], box["region"])
            match = np.array([
                scores_map.get((r.video_id, int(r.frame_id)), 0.0) for r in results.itertuples(index=False)
            ])
        boost += match
    return boost / len(boxes_with_region)


OCR_MATCH_QUALITY_BOOST_WEIGHT = 0.15  # cung thang do voi SPATIAL_POSITION_BOOST_WEIGHT


def _coverage_x_compactness(
    needle_words: list[str], video_id: str, frame_id: int, algorithm: str = DEFAULT_OCR_ALGORITHM
) -> float:
    _matched, score = match_words(algorithm, needle_words, _ocr_frame_words_clustered(video_id, frame_id))
    return score


def _ocr_match_quality_boost(
    results: pd.DataFrame, spatial_boxes: list[dict], ocr_algorithm: str = DEFAULT_OCR_ALGORITHM
) -> np.ndarray:
    """Diem CONG THEM theo do khop OCR (2026-08-17, theo yeu cau nguoi dung "nên tinh chỉnh
    theo độ khớp OCR, phải có score") = coverage (bao nhieu % tu needle tim thay - xem
    OCR_MATCH_MIN_COVERAGE, hard-filter da CHO QUA tu >=75% chu KHONG con doi 100%) NHAN VOI
    compactness (cum tu khop co LIEN TIEP hay bi box khac chen giua khong) - frame khop DAY DU
    + GON nhat duoc uu tien nhat, frame chi vua du nguong 75% + rai rac duoc it diem hon (van
    qua hard-filter, chi xep sau). CHI tinh CHO cac box loai "ocr" co text - tinh LAZY tren
    tung dong ket qua (_ocr_frame_words_clustered, xem docstring) vi CHI can cho tap ket qua DA
    LOC (nho), khong phai ca corpus.

    SOFT BOOST thuan tuy (khong loai frame) - frame khong co OCR (vd box Object) hoac khong
    khop van giu nguyen 0 diem, khong bi phat."""
    ocr_boxes = [b for b in spatial_boxes if b.get("type") == "ocr" and (b.get("text") or "").strip()]
    if not ocr_boxes or results.empty:
        return np.zeros(len(results))
    boost = np.zeros(len(results))
    for box in ocr_boxes:
        needle_words = _strip_accents(box["text"]).split()
        scores = np.array([
            _coverage_x_compactness(needle_words, r.video_id, int(r.frame_id), ocr_algorithm)
            for r in results.itertuples(index=False)
        ])
        boost += scores
    return boost / len(ocr_boxes)


# ============================================================ Thuat toan tinh diem (2026-08-17)
# 2026-08-17 (theo yeu cau nguoi dung: "có thuật toán nào đáng tin cậy hơn cosine similarity
# không" -> "làm hướng 1 thử giúp mình, nhưng đừng xóa cosine, làm tương tự như thuật toán như
# OCR") - GIONG HET pattern OCR_MATCH_ALGORITHMS (tier1_filter.py): registry + dispatch, "cosine"
# la MAC DINH (KHONG doi hanh vi cu), "sigmoid" la LUA CHON THEM nguoi dung tu bat qua UI.
#
# Y TUONG: SigLIP2 (khac CLIP/PE-Core/BEiT-3 dung contrastive softmax loss) duoc TRAIN bang
# SIGMOID LOSS (xem paper "Sigmoid Loss for Language Image Pre-Training") - cong thuc that su
# toi uu hoa la logit = cosine_sim * exp(logit_scale) + logit_bias, ROI sigmoid(logit) moi la
# "xac suat khop" ma model THAT SU hoc, KHONG phai cosine tho. code hien tai (_rank_single_*)
# CHI dung cosine tho (dung cho FAISS/dot-product, DUNG cho xep hang NOI BO 1 model vi sigmoid
# la ham DON DIEU TANG - khong doi THU TU trong CUNG 1 model), nhung SAI ve THANG DO khi CONG
# THEM cac soft-boost (SPATIAL_POSITION_BOOST_WEIGHT/OCR_MATCH_QUALITY_BOOST_WEIGHT/
# AUDIO_MENTION_BOOST_WEIGHT deu la trong so CO DINH 0.08-0.15, gia dinh ngam "score" nam trong
# khoang [0,1] hop ly - cosine SigLIP2 thuc te thuong chi ~0.05-0.35, sigmoid dua ve dung [0,1]
# CO Y NGHIA XAC SUAT, lam trong so boost "cong bang" hon).
#
# logit_scale/logit_bias la 2 THAM SO DA HOC CUA MODEL (co dinh sau khi train xong, KHONG doi
# theo query) - lay 1 LAN qua `AutoModel.from_pretrained(SIGLIP_MODEL_NAME).logit_scale/
# .logit_bias` (xem local_text_encoders.py::SIGLIP_MODEL_NAME), hardcode o day de KHONG PHAI
# nap them 1 ban SigLIP2 nua chi de doc 2 so vo huong.
_SIGLIP_LOGIT_SCALE_EXP = 112.66889953613281  # = exp(model.logit_scale), google/siglip2-base-patch16-224
_SIGLIP_LOGIT_BIAS = -16.771724700927734  # = model.logit_bias, cung model

SCORE_ALGORITHMS: dict[str, str] = {
    # key -> ten hien thi UI. "cosine": diem tho tu FAISS/dot-product (dang dung tu truoc, KHONG
    # doi). "sigmoid": sigmoid(cosine*scale+bias) - CHI co calibration THAT cho SigLIP2 (model
    # duy nhat train bang sigmoid loss trong 3 model dense hien co); voi pe_core/beit3 (khong co
    # logit_scale/bias rieng, train bang contrastive loss khac han) thi KHONG co cong thuc
    # calibration tin cay - _apply_score_algorithm() GIU NGUYEN cosine cho 2 model do du chon
    # "sigmoid" (an toan hon bia 1 cong thuc khong co co so, disclose ro trong UI help text).
    "cosine": "Cosine (mặc định)",
    "sigmoid": "Sigmoid hiệu chỉnh (chỉ SigLIP2)",
}
DEFAULT_SCORE_ALGORITHM = "cosine"


def _apply_score_algorithm(scores: np.ndarray, model: str, algorithm: str) -> np.ndarray:
    """Ap dung SAU KHI da co cosine tho (KHONG thay doi THU TU xep hang trong CUNG 1 model vi
    sigmoid don dieu tang - chi doi THANG DO tuyet doi, anh huong luc CONG BOOST/RRF-tie-break/
    hien thi diem). RRF (_rank_rrf) khong bi anh huong ve KET QUA cuoi (dung RANK, khong dung
    gia tri score tho, de tinh RRF) nhung score cua TUNG model con trong per_model_row van nen
    nhat quan - ap dung O DAY (trong _rank_single, TRUOC khi vao RRF) la dung 1 CHO DUY NHAT."""
    if algorithm == "sigmoid" and model == "siglip":
        return 1.0 / (1.0 + np.exp(-(scores * _SIGLIP_LOGIT_SCALE_EXP + _SIGLIP_LOGIT_BIAS)))
    return scores


def _encode_query(query: str, model: str) -> np.ndarray:
    # SUA (2026-08-14, theo yeu cau nguoi dung): encode QUERY TEXT chay LOCAL (giong het
    # pattern tier2_vector.py::encode_query() dung cho CLIP hien tai) thay vi goi Modal
    # remote() moi lan - tranh phu thuoc Modal app con song/da deploy cho duong hoi/dap ONLINE
    # (chi anh CORPUS moi thuc su can Modal GPU, da lam xong o build_dense_embeddings.py).
    v = ENCODERS[model](query)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    return v


def _rank_single_local(
    qvec: np.ndarray, model: str, top_k: int, candidates: set[tuple[str, int]] | None
) -> pd.DataFrame:
    """CHE DO LOCAL (AIC_LOCAL_MODELS=1) - y HET logic cu, nap matrix.npy + faiss.index TRUC
    TIEP tren may (xem _load_dense_index)."""
    matrix, index = _load_dense_index(model)
    meta = _load_dense_meta()

    if candidates is None:
        scores, idx = index.search(qvec, top_k)
        out = meta.iloc[idx[0]].copy()
        out["score"] = scores[0]
        return out.rename(columns={"frame_idx": "frame_id"}).reset_index(drop=True)
    elif not candidates:
        return meta.iloc[0:0].copy().rename(columns={"frame_idx": "frame_id"}).assign(score=[])
    else:
        row_pos = _load_dense_row_pos()
        positions = np.array([row_pos[k] for k in candidates if k in row_pos], dtype=np.int64)
        if len(positions) == 0:
            return meta.iloc[0:0].copy().rename(columns={"frame_idx": "frame_id"}).assign(score=[])
        sub_matrix = matrix[positions]
        scores = sub_matrix @ qvec[0]
        order = np.argsort(-scores)[:top_k]
        out = meta.iloc[positions[order]].copy()
        out["score"] = scores[order]
        return out.rename(columns={"frame_idx": "frame_id"}).reset_index(drop=True)


_EMPTY_RANK_COLUMNS = ["video_id", "frame_id", "shot_idx", "path", "score"]


def _rank_single_remote(
    qvec: np.ndarray, model: str, top_k: int, candidates: set[tuple[str, int]] | None
) -> pd.DataFrame:
    """CHE DO MAC DINH (khong set AIC_LOCAL_MODELS) - goi server Modal aic2026-dense-index
    (xem offline/modal_infra/dense_index_app.py) thay vi nap ~7.4GB matrix+faiss vao RAM may
    local (2026-08-16, theo yeu cau nguoi dung "may minh moi lan chay len no chiem gan het
    RAM"). candidates=None -> server tim tren TOAN BO index qua FAISS; [] -> tra rong NGAY
    (khong goi mang, giu dung ngu nghia "khong loc gi" cu).

    2026-08-20 (theo yeu cau nguoi dung: "vẫn chạy bằng Modal... TRAKE mở rộng pool lên 20000
    chạy cực kỳ lâu, 160s" - da do that: pool_k=20000 remote ~10.9s/model, local ~0.66s -
    KHONG phai FAISS server-side cham (do lai xac nhan FAISS gan nhu KHONG doi theo top_k) MA
    la PAYLOAD TRUYEN QUA MANG cang lon top_k cang nang, dac biet cot "path" (chuoi duong dan
    tuyet doi dai) x hang chuc nghin dong): goi rank(..., light=True) - server CHI tra
    [video_id, frame_id, score] (khong shot_idx/path) - GHEP LAI 2 cot do TU dense_meta.parquet
    CUC BO (_load_dense_meta(), khong qua Modal - file nay LUON co san local du AIC_LOCAL_MODELS
    bat/tat, xem docstring _load_dense_meta) qua _load_dense_row_pos() (O(1) tra cuu, da cache).
    Giam dang ke kich thuoc phan hoi mang, khong doi KET QUA (video_id/frame_id/score giong
    het, chi shot_idx/path lay tu nguon LOCAL thay vi qua mang)."""
    if candidates is not None and not candidates:
        return pd.DataFrame(columns=_EMPTY_RANK_COLUMNS)
    candidate_keys = [[vid, int(fid)] for vid, fid in candidates] if candidates is not None else None
    try:
        # 2026-08-20 (theo yeu cau nguoi dung, tiep tuc sau timeout NIM: diem treo THAT SU la
        # .remote() khong co timeout) - dung call_modal_with_timeout thay vi .remote() truc tiep.
        rows = call_modal_with_timeout(
            _dense_index_server().rank, model, qvec[0].tolist(), top_k, candidate_keys,
            light=True, context=f"xếp hạng {model}",
        )
    except ModalTimeoutError:
        raise
    except Exception as e:
        raise RuntimeError(
            f"Không gọi được server Modal aic2026-dense-index ({type(e).__name__}: {e}) — "
            f"đã deploy chưa? (modal deploy offline/modal_infra/dense_index_app.py), hoặc "
            f"set AIC_LOCAL_MODELS=1 để chạy local (xem README.md)."
        ) from e
    if not rows:
        return pd.DataFrame(columns=_EMPTY_RANK_COLUMNS)

    # 2026-08-20 (toi uu tiep - do that: goi RPC light=True DA nhanh that (k=20000: 1.9s vs 5.3s
    # ban day du, ~2.8 lan), nhung buoc GHEP LAI shot_idx/path o duoi TRUOC DAY dung vong lap
    # Python + pandas `.iat[]` tung dong (cham, dung nguyen mau hinh da gap voi OCR/shot_bounds
    # truoc do) - AN HET phan loi ich vua co duoc. Fix: vector hoa hoan toan bang numpy fancy
    # indexing thay vi loop + .iat[]."""
    row_pos = _load_dense_row_pos()
    video_ids_raw = [r[0] for r in rows]
    frame_ids_raw = [int(r[1]) for r in rows]
    scores_raw = [float(r[2]) for r in rows]
    positions = np.array(
        [row_pos.get((vid, fid), -1) for vid, fid in zip(video_ids_raw, frame_ids_raw)], dtype=np.int64
    )
    valid = positions >= 0  # phong than: khong nen co -1 (server dung CHUNG dense_meta voi local)
    if not valid.all():
        positions = positions[valid]
        video_ids_raw = [v for v, ok in zip(video_ids_raw, valid) if ok]
        frame_ids_raw = [f for f, ok in zip(frame_ids_raw, valid) if ok]
        scores_raw = [s for s, ok in zip(scores_raw, valid) if ok]

    meta = _load_dense_meta()
    shot_idx_arr = meta["shot_idx"].values
    path_arr = meta["path"].values
    return pd.DataFrame({
        "video_id": video_ids_raw, "frame_id": frame_ids_raw,
        "shot_idx": shot_idx_arr[positions].astype(int), "path": path_arr[positions],
        "score": scores_raw,
    })


def _rank_single(
    query: str, model: str, top_k: int, candidates: set[tuple[str, int]] | None = None,
    score_algorithm: str = DEFAULT_SCORE_ALGORITHM, log=None,
) -> pd.DataFrame:
    """1 model - tra ve DataFrame [video_id, frame_id (=frame_idx), shot_idx, path, score].
    candidates=None -> tim tren TOAN BO index qua FAISS (giong tier2_vector.rank()). candidates
    cu the (vd tu OCR hard-filter) -> tinh cosine TRUC TIEP tren dung tap do, KHONG dung
    FAISS-pool-roi-loc (cung nguyen tac voi tier2_vector.rank()). log: StepLog tuy chon (xem
    steplog.py) - ghi rieng thoi gian encode query + xep hang cho MODEL nay (2026-08-15, theo
    yeu cau nguoi dung: log thoi gian encode cho ca 4 che do siglip/pe_core/beit3/rrf - rrf goi
    lai chinh ham nay 3 lan, xem _rank_rrf duoi, nen tu no da duoc log rieng tung model).

    2026-08-16: viec XEP HANG THAT (doc matrix/faiss) gio o 1 trong 2 noi qua bien moi truong
    AIC_LOCAL_MODELS (_local_mode()) - xem _rank_single_local/_rank_single_remote."""
    with (log.timed(f"Encode query + xếp hạng — {model}") if log else contextlib.nullcontext(lambda *_a, **_k: None)) as set_detail:
        qvec = _encode_query(query, model)
        if _local_mode():
            result = _rank_single_local(qvec, model, top_k, candidates)
        else:
            result = _rank_single_remote(qvec, model, top_k, candidates)
        if len(result) and score_algorithm != "cosine":
            result = result.copy()
            result["score"] = _apply_score_algorithm(result["score"].to_numpy(), model, score_algorithm)
        set_detail(f"{len(result)} kết quả (candidates="
                    f"{'toàn corpus' if candidates is None else len(candidates)})")
    return result


RRF_FUSION_MODELS = ("siglip", "pe_core")  # 2026-08-18 (theo yeu cau nguoi dung, sau nhieu lan
# test that: "SigLIP-2 khá trội hơn so với 2 mô hình kia... BEiT-3 thì mình có thể chắc chắn nó
# không đủ mạnh để so với 2 mô hình kia... nên bỏ BEiT-3 ra khỏi all luôn") - RRF (mode "All")
# GIO CHI fusion SigLIP2+PE-Core (2 model), KHONG con BEiT-3 nua - ly do co so: RRF cong theo
# RANK (dong thuan), 1 model YEU HON van co "phieu" ngang hang 2 model manh trong cong thuc, co
# the KEO XUONG ket qua dung cua model manh (da thao luan co che trong hoi thoai truoc). BEiT-3
# VAN GIU NGUYEN option RIENG trong UI (radio "Model xếp hạng") - CHI bo khoi fusion, khong xoa
# hoan toan kha nang dung rieng (nguoi dung yeu cau ro "vẫn giữ nguyên option riêng cho nó").

def _rank_rrf(
    query: str, top_k: int, candidates: set[tuple[str, int]] | None = None, pool_k: int = 200,
    score_algorithm: str = DEFAULT_SCORE_ALGORITHM, log=None,
) -> pd.DataFrame:
    """Fusion RRF: lay top pool_k tu MOI model rieng le (xem RRF_FUSION_MODELS), tinh RRF-score
    = sum(1/(RRF_K+rank_i)) qua cac model co xuat hien (khong xuat hien trong top pool_k cua 1
    model nao do coi nhu rank vo cung, dong gop 0 tu model do - KHONG loai anh, chi khong duoc
    cong tu nguon do).
    log: truyen xuong _rank_single() cho TUNG model - xem timing rieng cua tung model trong 1
    lan chay rrf.

    BUG THAT (2026-08-19, phat hien qua TRAKE 4-moc "cắt nấm/củ năng/đậu hủ/lên bếp" - dense_
    model="rrf" (mac dinh UI, nut "All") tra ve 0 KET QUA HOAN TOAN, du dense_temporal.py da mo
    rong pool len 20000/moc): pool_k=200 CO DINH, KHONG scale theo top_k yeu cau tu callers -
    fusion chi tu toi da ~2*200=400 ung vien DUY NHAT (2 model trong RRF_FUSION_MODELS) BAT KE
    top_k=1000 hay 20000. Callers (dense_temporal._run_anchor_pool voi coarse_k lon, search_dense
    backfill voi top_k lon) tuong minh xin duoc top_k ung vien nhung THAT SU chi nhan duoc <=400.
    Fix: pool_k KHONG DUOC nho hon top_k - can it nhat top_k ung vien/model moi co co hoi fuse ra
    du top_k ket qua cuoi cung.

    2026-08-20 (theo yeu cau nguoi dung: "chạy song song mô hình embedding... vẫn giữ thời gian
    chạy như chạy từng cái") - 2 model TRUOC DAY chay TUAN TU (for loop) - tong thoi gian ~=
    thoi_gian(siglip) + thoi_gian(pe_core). Chuyen sang goi CA 2 CUNG LUC bang ThreadPoolExecutor
    (KHONG phai multiprocessing - _rank_single() hoac cho MANG (Modal .remote(), I/O-bound, nha
    GIL khi cho) hoac tinh toan numpy/faiss/torch (deu nha GIL trong phan C/C++ nang), nen
    threading la du, khong can multiprocessing nang hon) - tong thoi gian ~= max(thoi_gian(sig
    lip), thoi_gian(pe_core)) thay vi cong don, GIU NGUYEN thoi gian TUNG MODEL rieng le (khong
    doi logic/tham so cua _rank_single, chi doi CACH GOI). log.timed() ben trong _rank_single()
    van AN TOAN khi goi tu nhieu thread (list.append thread-safe trong CPython/GIL) - thu tu cac
    dong log co the doi (model nao xong TRUOC len log TRUOC) nhung khong sai lech du lieu."""
    pool_k = max(pool_k, top_k)
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=len(RRF_FUSION_MODELS)) as executor:
        futures = {
            model: executor.submit(
                _rank_single, query, model, pool_k, candidates=candidates,
                score_algorithm=score_algorithm, log=log,
            )
            for model in RRF_FUSION_MODELS
        }
        ranked_by_model = {model: fut.result() for model, fut in futures.items()}

    rrf_scores: dict[tuple, float] = {}
    per_model_row: dict[tuple, pd.Series] = {}
    for model in RRF_FUSION_MODELS:
        ranked = ranked_by_model[model]
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


def _spatial_box_candidates(
    spatial_boxes: list[dict], op: str = "and", ocr_algorithm: str = DEFAULT_OCR_ALGORITHM
) -> set[tuple[str, int]]:
    """spatial_boxes: [{"type": "ocr"|"object", "text": str, "labels": list[str],
    "min_count": int, "region": (ymin,xmin,ymax,xmax)|None}, ...] tu UI (xem app.py).
    op="and": frame phai khop DU MOI khung (giao). op="or": frame khop BAT KY khung nao
    (hop) - theo yeu cau nguoi dung 2026-08-15 (giong toggle "Objects in AND/OR" o UI tham
    khao)."""
    per_box = [
        _ocr_box_candidates(box.get("text", ""), ocr_algorithm) if box["type"] == "ocr"
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


# ============================================================ Tach menh de (2026-08-18)
# 2026-08-18 (theo yeu cau nguoi dung, sau khi TU CHUNG MINH bang so lieu that): cau query GHEP
# NHIEU mo ta (vd "Đoạn video quay một khu chợ ngoài trời... Giữa khung hình một phụ nữ đội nón
# lá...") encode THANH 1 VECTOR DUY NHAT se bi 1 menh de LAN AT menh de kia - do that: frame
# GT khop menh de "chợ" o hang #0/toan corpus nhung menh de "phụ nữ" chi hang #7700 (nguoi phu
# nu khong noi bat trong canh cho toan canh THAT) - encode ca cau ghep keo GT xuong hang #159.
# Thu 3 cach gop diem tach rieng: TRUNG BINH CONG (hang #80), trong so 0.7/0.3 (hang #16), MAX
# (hang #1 - THANG CONG) - MAX dung nguyen tac DA CO SAN trong _apply_region_clip_rerank ("lay
# diem CAO NHAT trong cac box, khong can MOI box khop") - 1 frame khop THAT SU chi can KHOP MANH
# it nhat 1 khia canh mo ta, khong can DEU TAY ca 2, khac han AVG vo tinh thien vi frame "khop
# vua vua ca 2" hon la "khop han 1 cai".
import re as _re

_CLAUSE_SPLIT_RE = _re.compile(r"(?<=[.!?])\s+")


def split_query_clauses(query: str) -> list[str]:
    """Tach query theo dau cham cau ket thuc cau (./!/?) thanh tung MENH DE doc lap - moi
    menh de se duoc CHUNG CAT + encode RIENG (xem _rank_multi_clause), khong con gop chung
    thanh 1 cau roi encode 1 lan. Cau chi co 1 cau (khong co dau cham giua) -> tra ve nguyen
    ven trong list 1 phan tu (khong tach gi, hanh vi tuong duong duong don-vector cu)."""
    parts = [p.strip() for p in _CLAUSE_SPLIT_RE.split(query.strip()) if p.strip()]
    return parts if len(parts) > 1 else [query]


def _rank_multi_clause(
    query: str, mode: str, top_k: int, candidates: set[tuple[str, int]] | None = None,
    pool_k: int = 200, score_algorithm: str = DEFAULT_SCORE_ALGORITHM,
    distill_model: str | None = None, log=None,
) -> pd.DataFrame:
    """Xep hang theo tung MENH DE rieng (xem split_query_clauses), gop lai bang MAX diem/frame
    (khong phai encode ca cau thanh 1 vector) - xem docstring khoi tren cho ly do/so lieu that.
    Neu query chi co 1 menh de (khong tach duoc), goi thang _rank_rrf/_rank_single NHU CU - ham
    nay chi THEM hanh vi, khong doi ket qua cho cau 1-menh-de.

    distill_model: xem docstring search_dense - truyen thang xuong distill_query() cho MOI
    menh de (None -> dung DEFAULT_DISTILL_MODEL cua query_distill.py)."""
    from query_distill import DEFAULT_DISTILL_MODEL, distill_query  # cung 1 cho voi search_dense

    distill_model = distill_model or DEFAULT_DISTILL_MODEL
    clauses = split_query_clauses(query)
    if len(clauses) <= 1:
        distilled = distill_query(query, distill_model)
        if mode == "rrf":
            return _rank_rrf(distilled, top_k, candidates=candidates, score_algorithm=score_algorithm, log=log)
        return _rank_single(distilled, mode, top_k, candidates=candidates, score_algorithm=score_algorithm, log=log)

    best_score: dict[tuple, float] = {}
    best_row: dict[tuple, pd.Series] = {}
    for clause in clauses:
        step_name = f"Mệnh đề riêng ('{clause[:60]}{'...' if len(clause) > 60 else ''}')"
        with (log.timed(step_name) if log else contextlib.nullcontext(lambda *_a, **_k: None)) as set_detail:
            distilled = distill_query(clause, distill_model)
            if mode == "rrf":
                ranked = _rank_rrf(distilled, pool_k, candidates=candidates, score_algorithm=score_algorithm, log=log)
            else:
                ranked = _rank_single(distilled, mode, pool_k, candidates=candidates, score_algorithm=score_algorithm, log=log)
            if log:
                set_detail(f'Chưng cất: "{distilled}" — {len(ranked)} kết quả')
        for _, row in ranked.iterrows():
            key = (row["video_id"], int(row["frame_id"]))
            sc = float(row["score"])
            # MAX diem giua cac menh de (KHONG phai trung binh cong) - xem docstring khoi tren:
            # frame khop MANH 1 menh de la du, khong can khop DEU ca cac menh de.
            if key not in best_score or sc > best_score[key]:
                best_score[key] = sc
                best_row[key] = row

    order = sorted(best_score.items(), key=lambda kv: -kv[1])[:top_k]
    rows = []
    for key, score in order:
        row = best_row[key].copy()
        row["score"] = score
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame(columns=_EMPTY_RANK_COLUMNS)


def _combine_candidates(
    *,
    authors: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    keywords_any: list[str] | None = None,
    must_have_labels: list[str] | None = None,
    min_count: dict[str, int] | None = None,
    ocr_text: str | None = None,
    asr_text: str | None = None,
    spatial_boxes: list[dict] | None = None,
    spatial_op: str = "and",
    ocr_algorithm: str = DEFAULT_OCR_ALGORITHM,
) -> set[tuple[str, int]] | None:
    """Gop metadata (video-level) + object + OCR (frame-level) + ASR (frame-level, xem
    _asr_candidates) + khung vi tri (spatial_boxes) thanh 1 tap (video_id, frame_idx). Cac
    khung vi tri gop VOI NHAU theo spatial_op ("and"/"or"), ROI ket qua do lai AND voi metadata/
    object/OCR/ASR toan cuc (giong tier1_filter.apply() nhung key la frame_idx)."""
    video_allowed = by_metadata(authors, date_from, date_to, keywords_any)
    object_allowed = _object_candidates(must_have_labels, min_count)
    text_allowed = _ocr_candidates(ocr_text, ocr_algorithm) if ocr_text else None
    audio_allowed = _asr_candidates(asr_text) if asr_text else None
    spatial_allowed = _spatial_box_candidates(spatial_boxes, spatial_op, ocr_algorithm) if spatial_boxes else None

    frame_sets = [s for s in (object_allowed, text_allowed, audio_allowed, spatial_allowed) if s is not None]
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
    asr_text: str | None = None,
    spatial_boxes: list[dict] | None = None,
    spatial_op: str = "and",
    audio_mentions: list[dict] | None = None,
    ocr_algorithm: str = DEFAULT_OCR_ALGORITHM,
    score_algorithm: str = DEFAULT_SCORE_ALGORITHM,
    multi_clause: bool = False,
    distill_model: str | None = None,
    log=None,
) -> pd.DataFrame:
    """mode in DENSE_MODES ("siglip"/"pe_core"/"beit3"/"rrf"). ocr_text: hard-filter chu tren
    man hinh (giong tier1_filter.by_text) - None = khong loc, "" cung coi nhu None.
    authors/date_from/date_to/keywords_any: loc video (tier1_filter.by_metadata, dung chung
    voi bo BTC vi la du lieu video-level). must_have_labels/min_count: loc theo Object toan cuc
    (khong xet vi tri - OWLv2 closed-set, xem _object_candidates). spatial_boxes: khung OCR/
    Object CO vi tri tu canvas (xem _spatial_box_candidates + app.py). spatial_op: "and" (frame
    phai khop DU moi khung) hoac "or" (khop BAT KY khung nao). audio_mentions: [{"term": str}]
    tu LLM (query_planner.py) - soft-boost theo loi noi GAN frame (xem _audio_mention_boost) -
    KHONG loai frame nao, chi cong diem nhe.

    asr_text (2026-08-20, theo yeu cau nguoi dung: "tích hợp search theo ASR... đảm bảo những
    câu như... nặng đến 211kg... mà embedding model thường không thấy") - HARD FILTER theo loi
    noi (khac audio_mentions o tren - do la SOFT boost tu LLM tu dong, con day la nguoi dung TU
    DIEN, giong OCR): None/"" = khong loc. Frame hop le = nam trong khoang thoi gian cua 1 doan
    ASR co text_norm chua asr_text (ordered-substring, xem _asr_candidates).
    ocr_algorithm: key trong tier1_filter.OCR_MATCH_ALGORITHMS ("flexible" mac dinh - nguoi
    dung chon o UI, xem app.py) - dung CHUNG cho ca hard-filter OCR (ocr_text/spatial_boxes)
    LAN soft-boost do gon khop (_ocr_match_quality_boost/_spatial_position_boost).
    score_algorithm: key trong SCORE_ALGORITHMS ("cosine" mac dinh - nguoi dung chon o UI) -
    xem _apply_score_algorithm cho ly do/cong thuc "sigmoid" (chi calibration THAT cho SigLIP2).
    multi_clause: mac dinh False (giu nguyen hanh vi cu - encode CA CAU thanh 1 vector). Bat
    True -> tach cau theo dau cham cau (./!/?) thanh tung MENH DE, encode RIENG tung menh de,
    gop diem bang MAX (xem _rank_multi_clause) - theo yeu cau nguoi dung 2026-08-18, sau khi
    TU CHUNG MINH bang so lieu that: cau ghep "khu chợ...phụ nữ đội nón lá..." encode chung 1
    vector khien menh de "phụ nữ" LAN AT menh de "khu chợ" (GT rot tu hang #0 rieng menh de
    xuong hang #159 khi ghep chung) - MAX giu duoc hang #1 (thang cong so voi AVG #80, trong so
    0.7/0.3 #16 - da tu do trong hoi thoai).
    distill_model: key trong query_distill.DISTILL_MODELS (2026-08-18, theo yeu cau nguoi
    dung "mình muốn thử với nhiều LLM hơn... model từ bé đến lớn") - None -> dung
    DEFAULT_DISTILL_MODEL (model 8B cu, KHONG doi hanh vi truoc do).
    log: StepLog tuy chon (xem steplog.py) - ghi lai buoc chung cat query (query_distill.py) de debug."""
    if mode not in DENSE_MODES:
        raise ValueError(f"mode phai la 1 trong {DENSE_MODES}, nhan '{mode}'")
    if ocr_algorithm not in OCR_MATCH_ALGORITHMS:
        ocr_algorithm = DEFAULT_OCR_ALGORITHM
    if score_algorithm not in SCORE_ALGORITHMS:
        score_algorithm = DEFAULT_SCORE_ALGORITHM

    # 2026-08-16 (BUG THAT nguoi dung phat hien: "tong 30s nhung cong step log lai chi ~5s") -
    # buoc nay TRUOC DAY khong duoc log() bao boc GI CA - doc objects_index.parquet (134MB,
    # lru_cache CHI nhanh tu LAN GOI THU 2 tro di trong CUNG process) + OCR_TEXT_PATH (KHONG
    # cache, doc lai tu dia MOI LAN goi, xem _ocr_candidates/_ocr_box_candidates) co the ton
    # vai giay ma hoan toan "vo hinh" trong log - day la 1 trong 2 nguon gay lech "tong 30s vs
    # log cong lai chi 5s" (nguon con lai: xem backfill duoi day).
    with (log.timed("Lọc thô (metadata/object/OCR/ASR/khung vẽ)") if log else contextlib.nullcontext(lambda *_a, **_k: None)) as set_detail:
        candidates = _combine_candidates(
            authors=authors, date_from=date_from, date_to=date_to, keywords_any=keywords_any,
            must_have_labels=must_have_labels, min_count=min_count, ocr_text=ocr_text,
            asr_text=asr_text, spatial_boxes=spatial_boxes, spatial_op=spatial_op, ocr_algorithm=ocr_algorithm,
        )
        if log:
            set_detail("không lọc gì (candidates=None)" if candidates is None else f"{len(candidates)} frame ứng viên")

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
    # 2026-08-18 (BUG THAT nguoi dung phat hien: "UnboundLocalError: distilled") - `distilled`
    # van PHAI tinh du multi_clause=True hay khong, vi backfill (BO LOC CUNG - bu them CLIP
    # thuan, xem duoi) dung LAI bien nay VO DIEU KIEN, khong biet gi ve multi_clause. Multi_clause
    # CHI doi cho XEP HANG CHINH khong dung `distilled` nay nua (_rank_multi_clause tu chung cat
    # RIENG tung menh de) - `distilled` (ca cau) van tinh binh thuong de backfill dung duoc.
    from query_distill import DEFAULT_DISTILL_MODEL, DISTILL_MODELS, distill_query

    if distill_model not in DISTILL_MODELS:
        distill_model = DEFAULT_DISTILL_MODEL

    if log:
        with log.timed("Chưng cất query (dịch + rút gọn cho embedding)") as set_detail:
            distilled = distill_query(query, distill_model)
            set_detail(f'Gốc: "{query}"  \nChưng cất: "{distilled}"')
    else:
        distilled = distill_query(query, distill_model)

    # 2026-08-15 (theo yeu cau nguoi dung: khung ve tay khong the chinh xac) - khung CO vi tri
    # gio la SOFT BOOST (khong loai frame), can pool RONG HON top_k de con "cua" ma sap xep lai
    # sau khi cong diem, giong pattern Region-CLIP rerank. ASR audio_mentions cung la soft boost
    # - dung CHUNG 1 lan mo rong pool (khong mo rong 2 lan rieng) neu CA 2 cung co mat.
    boxes_with_region = [b for b in (spatial_boxes or []) if b.get("region") is not None]
    audio_mentions = audio_mentions or []
    needs_wide_pool = bool(boxes_with_region) or bool(audio_mentions)
    rank_top_k = top_k * 4 if needs_wide_pool else top_k

    if multi_clause:
        result = _rank_multi_clause(
            query, mode, rank_top_k, candidates=candidates, score_algorithm=score_algorithm,
            distill_model=distill_model, log=log,
        )
    elif mode == "rrf":
        result = _rank_rrf(distilled, rank_top_k, candidates=candidates, score_algorithm=score_algorithm, log=log)
    else:
        result = _rank_single(distilled, mode, rank_top_k, candidates=candidates, score_algorithm=score_algorithm, log=log)

    if needs_wide_pool and not result.empty:
        result = result.copy()
        n_pool = len(result)

        if boxes_with_region:
            with (log.timed("Cộng điểm vị trí khung vẽ (soft boost)") if log else contextlib.nullcontext(lambda *_a, **_k: None)) as set_detail:
                pos_boost = _spatial_position_boost(result, spatial_boxes or [], ocr_algorithm)
                result["position_boost"] = pos_boost
                result["score_before_position_boost"] = result["score"]
                result["score"] = result["score"] + SPATIAL_POSITION_BOOST_WEIGHT * pos_boost
                if log:
                    set_detail(f"{int((pos_boost > 0).sum())}/{n_pool} frame (trong pool mở rộng) "
                               f"có object/OCR khớp vị trí khung vẽ")

            with (log.timed("Cộng điểm độ gọn khớp OCR (soft boost)") if log else contextlib.nullcontext(lambda *_a, **_k: None)) as set_detail:
                ocr_quality_boost = _ocr_match_quality_boost(result, spatial_boxes or [], ocr_algorithm)
                result["ocr_match_quality"] = ocr_quality_boost
                result["score_before_ocr_quality_boost"] = result["score"]
                result["score"] = result["score"] + OCR_MATCH_QUALITY_BOOST_WEIGHT * ocr_quality_boost
                if log:
                    set_detail(f"{int((ocr_quality_boost > 0).sum())}/{n_pool} frame có cụm chữ "
                               f"khớp OCR - điểm gọn TB={ocr_quality_boost[ocr_quality_boost > 0].mean() if (ocr_quality_boost > 0).any() else 0:.2f}")

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
        # BUG THAT (2026-08-16, nguoi dung phat hien "tong 30s nhung log cong lai chi 5s") -
        # 2 dong goi _rank_rrf/_rank_single duoi day TRUOC DAY KHONG truyen log=log -> GOI LAI
        # encode+xep hang THAT SU (1 round-trip Modal nua, co the ca 3 model neu mode="rrf")
        # nhung HOAN TOAN VO HINH trong step log - day la nguon chinh gay lech "tong 30s vs
        # log cong lai chi 5s" (nguon con lai: xem _combine_candidates o dau ham). Truyen
        # log=log de buoc "Encode query + xếp hạng" ben trong tu ghi lai dung.
        with (log.timed("Bộ lọc cứng — bù thêm bằng CLIP thuần (backfill)") if log else contextlib.nullcontext(lambda *_a, **_k: None)) as set_detail:
            if mode == "rrf":
                backfill_pool = _rank_rrf(distilled, top_k, candidates=None, log=log)
            else:
                backfill_pool = _rank_single(distilled, mode, top_k, candidates=None, log=log)
            if not result.empty:
                existing_keys = set(zip(result["video_id"], result["frame_id"].astype(int)))
                backfill_pool = backfill_pool[~backfill_pool.apply(
                    lambda r: (r["video_id"], int(r["frame_id"])) in existing_keys, axis=1
                )]
            backfill_rows = backfill_pool.head(n_missing).copy()
            backfill_rows["is_backfill"] = True
            result = pd.concat([result, backfill_rows], ignore_index=True) if not result.empty else backfill_rows
            if log:
                set_detail(f"chỉ khớp {top_k - n_missing}/{top_k} kết quả — BỔ SUNG thêm "
                           f"{len(backfill_rows)} kết quả bằng CLIP thuần (BỎ QUA lọc cứng) cho đủ "
                           f"{top_k}, đánh dấu ở cột is_backfill")

    return result
