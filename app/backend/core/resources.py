"""Tài nguyên dùng chung cho cả pipeline search (tiers/) — tải 1 lần (lazy singleton),
mọi tầng đọc qua resources.get(), không tự load riêng.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from config import MODEL_CACHE_DIR

MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(MODEL_CACHE_DIR / "hub"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import json

import numpy as np
import pandas as pd

from config import (
    INDEX_DIR,
    INDEX_META_PATH,
    OBJECTS_INDEX_PATH,
    VIDEO_METADATA_PATH,
)

# 2026-08-20 (theo yeu cau nguoi dung: "dọn dẹp triệt để... không còn gọi CLIP") - TRUOC DAY
# Resources con giu model (SentenceTransformer CLIP_TEXT_MODEL_NAME)/index (faiss)/matrix
# (clip_matrix.npy) CHI de phuc vu tiers/tier2_vector.py (pipeline CLIP-32 cu, DA XOA) - nap
# CA 3 THU nay TON KHONG NHO (SentenceTransformer + faiss index + numpy matrix mac du KHONG AI
# DOC) MOI LAN app.py khoi dong (get() la lazy singleton, nhung app.py van goi qua filter
# cho hard-filter OCR/object nen van trigger load). Bo han 3 truong nay - Resources gio CHI con
# du lieu filter.py THAT SU dung (video_meta/objects_index/ocr_index/asr_index/row_pos).
# `meta` (INDEX_META_PATH) van doc de xay row_pos, nhung KHONG con expose ca DataFrame (khong ai
# can) - chi giu row_pos (dict) nhe hon nhieu.

OPEN_VOCAB_DETECTIONS_PATH = INDEX_DIR / "open_vocab_detections.parquet"
SUPPRESSED_AUDIT_PATH = INDEX_DIR / "closed_set_suppressed_by_open_vocab.parquet"
# OCR (2026-08-10): xem offline/build_ocr_index_demo.py (gia lap, cho test luong truoc) ->
# se thay bang offline/build_ocr_index.py (PaddleOCR that + gom cum) sau. Schema: video_id,
# local_idx_start/end (khoang frame hieu luc cua 1 "run" chu lien tuc), text_raw, text_norm
# (da bo dau san luc build - xem filter._strip_accents), ymin/xmin/ymax/xmax, score, source.
OCR_TEXT_PATH = INDEX_DIR / "ocr_text.parquet"
# ASR (2026-08-10): xem offline/build_asr_index.py (PhoWhisper qua Modal) - schema GIONG
# ocr_text.parquet (source="asr", ymin/xmin/ymax/xmax=NaN vi khong co vi tri khong gian, chi
# co local_idx_start/end tu map giay->frame). CHUA gop chung voi ocr_index - doc rieng, dung
# theo huong SOFT boost (query_planner._apply_asr_boost), khac han OCR la HARD filter (loi
# noi khong dong bo chat voi hinh, xem hoi thoai 2026-08-10).
ASR_TEXT_PATH = INDEX_DIR / "asr_text.parquet"

# BUG THAT phat hien 2026-08-07 (nguoi dung hoi): gop mu closed_set + open_vocab khong du -
# nhan closed_set SAI (vd "Dog" tren dau lan mua lan) van con nguyen, Tang 1 van khop khi query
# "con cho" du open_vocab da co "lion dance costume" dung o CUNG vi tri. Fix: box nao cua
# closed_set CHONG LAN cao (IoU) voi 1 box open_vocab TRONG CUNG FRAME -> coi nhu da bi "de"
# (open-vocab dang tin hon vi la ly do VLM/DINO duoc goi cho dung frame do) -> danh dau
# suppressed=True, KHONG xoa dong (van giu de audit/soat lai, giu vi tri detection_id nguyen
# ven cho region_embeddings.npy) - chi loai khoi ket qua khop nhan o Tang 1 (xem filter.py).
IOU_SUPPRESS_THRESHOLD = 0.5

# SUA LAN 2 (2026-08-07, van trong cung phien - test thu phat hien): chi IoU KHONG DU phan
# biet "1 vat the bi gan 2 nhan" (bug that: Dog/Boat/Vehicle tren costume/graphic) voi "2 vat
# the KHAC NHAU dung gan nhau tu nhien" (Person + Clothing/robe ho luon co IoU cao vi trang
# phuc phu gan het than nguoi - test that: IoU=0.73-0.82, cao hon ca truong hop Boat/Vehicle).
# Fix: loai han 2 nhom nhan RA KHOI dien duoc suppress - "person" (Person/Man/Woman/Boy/Girl...)
# va "clothing_accessory" (Clothing...) - day la 2 nhom hau nhu LUON dung dan du co chong lan
# vi tri voi bat ky gi khac, khac voi nhom "animal"/"vehicle"/"object" (Dog, Boat, Vehicle...)
# noi bug that su xay ra. Dung lai label_types.json da co san (build_label_types.py).
_PROTECTED_CATEGORIES = {"person", "clothing_accessory"}


def _load_protected_labels() -> set[str]:
    path = INDEX_DIR / "label_types.json"
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        label_types = json.load(f)
    return {lb for lb, cats in label_types.items() if set(cats) & _PROTECTED_CATEGORIES}


def _box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """boxes_*: (N,4) dang (ymin,xmin,ymax,xmax). Tra ve (len(a), len(b)) ma tran IoU."""
    ay0, ax0, ay1, ax1 = boxes_a[:, 0:1], boxes_a[:, 1:2], boxes_a[:, 2:3], boxes_a[:, 3:4]
    by0, bx0, by1, bx1 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]
    iy0, ix0 = np.maximum(ay0, by0), np.maximum(ax0, bx0)
    iy1, ix1 = np.minimum(ay1, by1), np.minimum(ax1, bx1)
    inter = np.clip(iy1 - iy0, 0, None) * np.clip(ix1 - ix0, 0, None)
    area_a = np.clip(ay1 - ay0, 0, None) * np.clip(ax1 - ax0, 0, None)
    area_b = np.clip(by1 - by0, 0, None) * np.clip(bx1 - bx0, 0, None)
    union = area_a + area_b - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _mark_suppressed_closed_rows(closed: pd.DataFrame, open_vocab: pd.DataFrame) -> pd.Series:
    """Voi tung frame CO ca 2 nguon: closed_set row nao IoU > nguong voi BAT KY open_vocab
    row nao -> suppressed=True. Chi duyet frame co open_vocab (nho hon nhieu so voi toan bo
    closed_set) de nhanh - dung groupby, khong loop tung cap tren toan corpus."""
    suppressed = pd.Series(False, index=closed.index)
    frames_with_open = set(zip(open_vocab.video_id, open_vocab.local_idx))
    if not frames_with_open:
        return suppressed

    protected_labels = _load_protected_labels()
    closed_scope = closed[
        closed.set_index(["video_id", "local_idx"]).index.isin(frames_with_open)
        & ~closed["label"].isin(protected_labels)
    ]
    closed_groups = {k: v for k, v in closed_scope.groupby(["video_id", "local_idx"])}
    open_groups = open_vocab.groupby(["video_id", "local_idx"])

    box_cols = ["ymin", "xmin", "ymax", "xmax"]
    for key, g_open in open_groups:
        g_closed = closed_groups.get(key)
        if g_closed is None or g_closed.empty:
            continue
        boxes_open = g_open[box_cols].to_numpy(dtype=float)
        boxes_closed = g_closed[box_cols].to_numpy(dtype=float)
        iou = _box_iou_matrix(boxes_closed, boxes_open)  # (n_closed, n_open)
        hit = (iou > IOU_SUPPRESS_THRESHOLD).any(axis=1)
        suppressed.loc[g_closed.index[hit]] = True
    return suppressed


@dataclass
class Resources:
    row_pos: dict[tuple[str, int], int]
    video_meta: pd.DataFrame
    objects_index: pd.DataFrame | None
    ocr_index: pd.DataFrame | None
    asr_index: pd.DataFrame | None


def _load_merged_objects_index() -> pd.DataFrame | None:
    """Gop objects_index.parquet (closed-vocab, Faster R-CNN, 514 nhan, TOAN BO corpus) voi
    open_vocab_detections.parquet (open-vocab, Grounding DINO, chi frame anchor VLM da gan co
    - xem vocab_discovery.py/run_grounding_dino.py). Cot "source" phan biet nguon goc (giu ca
    2, khong ghi de - dung tinh than gap-honesty, van soat lai duoc sau).

    QUAN TRONG: concat(ignore_index=True) giu NGUYEN vi tri (RangeIndex) cua objects_index goc
    o dau (0..N-1) - detection_id cac noi khac dang dung (vd region_embeddings.npy do
    build_region_embeddings.py tao, tra theo detection_id = vi tri dong goc) KHONG bi lech, vi
    dong open-vocab chi noi tiep vao SAU, khong chen giua."""
    if not OBJECTS_INDEX_PATH.exists():
        return None
    closed = pd.read_parquet(OBJECTS_INDEX_PATH)
    closed["source"] = "closed_set"

    closed["suppressed"] = False

    if not OPEN_VOCAB_DETECTIONS_PATH.exists():
        return closed

    open_vocab = pd.read_parquet(OPEN_VOCAB_DETECTIONS_PATH)
    open_vocab["suppressed"] = False

    suppressed_mask = _mark_suppressed_closed_rows(closed, open_vocab)
    closed.loc[suppressed_mask, "suppressed"] = True
    if suppressed_mask.any():
        closed.loc[suppressed_mask].to_parquet(SUPPRESSED_AUDIT_PATH, index=False)

    return pd.concat([closed, open_vocab], ignore_index=True)


_resources: Resources | None = None


def get() -> Resources:
    """Trả về Resources đã load (load lần đầu gọi, cache lại cho các lần sau)."""
    global _resources
    if _resources is not None:
        return _resources

    meta = pd.read_parquet(INDEX_META_PATH)
    row_pos = {(vid, li): i for i, (vid, li) in enumerate(zip(meta["video_id"], meta["local_idx"]))}
    video_meta = pd.read_parquet(VIDEO_METADATA_PATH)
    objects_index = _load_merged_objects_index()
    ocr_index = pd.read_parquet(OCR_TEXT_PATH) if OCR_TEXT_PATH.exists() else None
    asr_index = pd.read_parquet(ASR_TEXT_PATH) if ASR_TEXT_PATH.exists() else None

    _resources = Resources(
        row_pos=row_pos,
        video_meta=video_meta,
        objects_index=objects_index,
        ocr_index=ocr_index,
        asr_index=asr_index,
    )
    return _resources
