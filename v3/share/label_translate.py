"""Resolver: từ tiếng Việt (người dùng gõ) -> nhãn OpenImages V4 tiếng Anh (Objects dùng).

3 lớp, độ tin cậy giảm dần:
  1. Khớp CHÍNH XÁC với bản dịch (label_vi.json) — chắc chắn.
  2. Từ điển đồng nghĩa tay (SYNONYMS) — chắc chắn, nhưng phải thêm tay từng trường hợp.
  3. Substring 2 chiều — khá chắc (biến thể ngắn/dài của cùng cụm từ).

KHÔNG dùng CLIP text-text similarity làm auto-filter: đã kiểm chứng (2026-08-05) similarity
quá sát nhau giữa mọi cặp cụm từ ngắn (0.94-0.99), "cún con" xếp hạng gần nhất toàn nhãn sai
(Linh dương, Cú, Cây...) — silent-wrong-nhưng-trông-tự-tin, đúng loại lỗi cần tránh (gap
honesty). Nên suggest() dưới đây CHỈ để hiển thị gợi ý cho người dùng bấm chọn, KHÔNG được
tự động áp vào filter.
"""
from __future__ import annotations

import json
import os
import unicodedata

from config import CLIP_TEXT_MODEL_NAME, INDEX_DIR, MODEL_CACHE_DIR

LABEL_VI_PATH = INDEX_DIR / "label_vi.json"
LABEL_SYNONYMS_PATH = INDEX_DIR / "label_synonyms.json"
LABEL_EMB_PATH = INDEX_DIR / "label_vi_embeddings.npy"
# Ban dich cho nhan open-vocab (Grounding DINO, xem translate_open_vocab.py 2026-08-06) - top
# 300 nhan theo tan suat trong open_vocab_detections.parquet, dich qua NIM LLM.
OPEN_VOCAB_VI_PATH = INDEX_DIR / "open_vocab_vi.json"

# Từ điển đồng nghĩa tay — mở rộng dần khi phát hiện case mới, KHÔNG cần build lại gì.
SYNONYMS: dict[str, str] = {
    "thùng rác": "Waste container",
    "thùng chứa rác": "Waste container",
    "xe con": "Car",
    "ô tô": "Car",
    "xe ô tô": "Car",
    "cún": "Dog",
    "cún con": "Dog",
    "chó con": "Dog",
    "mèo con": "Cat",
    "miu": "Cat",
    "điện thoại": "Mobile phone",
    "smartphone": "Mobile phone",
    "laptop": "Laptop",
    "máy tính xách tay": "Laptop",
    "xe máy": "Motorcycle",
    "mô tô": "Motorcycle",
    "xe hơi": "Car",
    "xe tải": "Truck",
    # danh tu vai tro (chi nguoi noi chung) -> Person - phat hien thieu khi test query_planner
    "diễn giả": "Person",
    "phóng viên": "Person",
    "mc": "Person",
    "người dẫn chương trình": "Person",
    # vá tay nhãn open-vocab dịch sai (2026-08-07): LLM dịch "lion dance costume" ->
    # "Áo múa hổ" (tiger, sai) thay vì đúng "múa lân" - patch truc tiep, khong sua lai file
    # open_vocab_vi.json (giu nguyen de audit sau neu can).
    "múa lân": "lion dance costume",
    "lân": "lion dance costume",
    "con lân": "lion dance costume",
    "múa rồng": "dragon dance costume",
}

_label_vi: dict[str, str] | None = None
_vi_norm_to_label: dict[str, str] | None = None
_synonyms_norm: dict[str, str] | None = None


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s).strip().lower()
    return " ".join(s.split())


def _load() -> None:
    global _label_vi, _vi_norm_to_label, _synonyms_norm
    if _label_vi is not None:
        return
    with open(LABEL_VI_PATH, encoding="utf-8") as f:
        _label_vi = json.load(f)
    # bản dịch chính (label_vi.json) là NGUỒN GỐC, nạp trước — không bị synonym sinh tự động
    # ghi đè lên, chỉ điền thêm key còn trống.
    _vi_norm_to_label = {_normalize(vi): en for en, vi in _label_vi.items()}

    # nhãn open-vocab (Grounding DINO) - cùng lớp ưu tiên với label_vi.json (bản dịch chính),
    # chỉ điền thêm key chưa có (label_vi.json 514 nhãn gốc luôn ưu tiên nếu trùng tình cờ).
    if OPEN_VOCAB_VI_PATH.exists():
        with open(OPEN_VOCAB_VI_PATH, encoding="utf-8") as f:
            open_vocab_vi = json.load(f)
        for en, vi in open_vocab_vi.items():
            key = _normalize(vi)
            if key and key not in _vi_norm_to_label:
                _vi_norm_to_label[key] = en

    if LABEL_SYNONYMS_PATH.exists():
        with open(LABEL_SYNONYMS_PATH, encoding="utf-8") as f:
            label_synonyms = json.load(f)
        for en, syns in label_synonyms.items():
            for vi in syns:
                key = _normalize(vi)
                if key and key not in _vi_norm_to_label:
                    _vi_norm_to_label[key] = en

    # SYNONYMS tay đè lên cùng (ưu tiên cao nhất — patch tay cho case đã biết là sai/thiếu).
    for vi, en in SYNONYMS.items():
        _vi_norm_to_label[_normalize(vi)] = en

    _synonyms_norm = {_normalize(k): v for k, v in SYNONYMS.items()}


def resolve(term_vi: str) -> list[str]:
    """Khớp CHẮC CHẮN (exact/synonym/substring). Trả [] nếu không tìm được — không đoán."""
    _load()
    assert _vi_norm_to_label is not None and _synonyms_norm is not None
    q = _normalize(term_vi)
    if not q:
        return []

    if q in _vi_norm_to_label:
        return [_vi_norm_to_label[q]]
    if q in _synonyms_norm:
        return [_synonyms_norm[q]]

    # Substring 2 chiều theo TỪ (word-boundary), không phải ký tự thô — tránh case
    # "áo" (2 ký tự) khớp nhầm vào giữa từ "báo" trong "biển báo giao thông" (2026-08-06,
    # bug thật: query "áo xanh" -> resolve ra "Traffic sign"/"Sports uniform"). Đệm khoảng
    # trắng 2 đầu để containment check luôn rơi đúng ranh giới từ. Từ quá ngắn (<3 ký tự,
    # vd "áo", "xe") không đủ đặc trưng để substring match — bỏ qua, tránh false-positive.
    if len(q) < 3:
        return []
    qp = f" {q} "
    hits = []
    for vi_norm, en in _vi_norm_to_label.items():
        # SUA 2026-08-08 (bug that: "đá viên" -> "Football" vi nhan khac dich la "đá" - 2 ky
        # tu - lot TRON trong query dai): lan truoc chi chan query NGAN chui vao nhan dai
        # (len(q)<3 o tren), nhung CHUA chan chieu nguoc lai - nhan dich NGAN chui vao query
        # dai. Ap dung do dai toi thieu cho CA 2 phia, khong chi 1 phia.
        # SUA LAN 2 (2026-08-08): nguong <3 (chan 1-2 ky tu) van khong du - "cây" (3 ky tu,
        # nhan "Tree") lot vao "máy ép trái cây" (juicer) vi chi co 1 hit duy nhat nen khong
        # co gi de "da so phieu" so sanh. Nang nguong len <4 (chan luon tu 3 ky tu don, thuong
        # qua chung chung/de trung ngau nhien). Query truc tiep dung tu ngan (vd goi "cây" =
        # tree that) van hoat dong binh thuong qua lop KHOP CHINH XAC (dong tren, khong bi
        # anh huong boi nguong nay - nguong nay chi ap dung cho lop substring long leo).
        if len(vi_norm) < 4:
            continue
        vip = f" {vi_norm} "
        if qp in vip or vip in qp:
            hits.append((vi_norm, en))
    if not hits:
        return []
    # SUA 2026-08-08 (bug that: "bát" -> "Baseball bat" thay vi "Bowl"): truoc day chon theo
    # match DAI NHAT - "bát bóng chày" (13 ky tu, Baseball bat, phien am muon "bat" tieng Anh)
    # thang "bát ăn cơm" (10 ky tu, Bowl) chi vi dai hon, du Bowl co toi 5 vi_norm khop dung
    # nghia con Baseball bat chi co 2 (nham lan tu muon). Doi sang DA SO PHIEU: nhan EN nao co
    # NHIEU vi_norm khop nhat (trong so cac hit) thi thang - phan anh dung "nghia pho bien"
    # hon la "chuoi dai hon", it bi anh huong boi 1-2 tu muon/hiem gap.
    en_counts: dict[str, int] = {}
    for _, en in hits:
        en_counts[en] = en_counts.get(en, 0) + 1
    max_count = max(en_counts.values())
    return [en for en, c in en_counts.items() if c == max_count]


# ============================================================ Gợi ý mờ (KHÔNG auto-apply)
_model = None
_labels_ordered: list[str] | None = None
_label_embeddings = None


def _load_embeddings() -> None:
    global _model, _labels_ordered, _label_embeddings
    if _label_embeddings is not None:
        return
    import numpy as np

    _load()
    assert _label_vi is not None
    _labels_ordered = list(_label_vi.keys())

    if LABEL_EMB_PATH.exists():
        _label_embeddings = np.load(LABEL_EMB_PATH)
        return

    os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))
    from sentence_transformers import SentenceTransformer

    _model = SentenceTransformer(CLIP_TEXT_MODEL_NAME, cache_folder=str(MODEL_CACHE_DIR))
    vi_texts = [_label_vi[lb] for lb in _labels_ordered]
    _label_embeddings = _model.encode(vi_texts, normalize_embeddings=True, show_progress_bar=False)
    np.save(LABEL_EMB_PATH, _label_embeddings)


def suggest(term_vi: str, top_k: int = 5):
    """Gợi ý mờ theo similarity CLIP text (KHÔNG dùng để filter tự động — chỉ hiển thị cho
    người dùng chọn tay). Trả list[(label_en, label_vi, score)]."""
    global _model
    import numpy as np

    _load_embeddings()
    assert _labels_ordered is not None and _label_embeddings is not None and _label_vi is not None

    if _model is None:
        os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(CLIP_TEXT_MODEL_NAME, cache_folder=str(MODEL_CACHE_DIR))

    qe = _model.encode([term_vi], normalize_embeddings=True)[0]
    sims = _label_embeddings @ qe
    top_idx = np.argsort(-sims)[:top_k]
    return [(_labels_ordered[i], _label_vi[_labels_ordered[i]], float(sims[i])) for i in top_idx]


if __name__ == "__main__":
    import sys

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    for term in sys.argv[1:] or ["thùng rác", "cún con", "người", "không tồn tại xyz"]:
        r = resolve(term)
        print(f"resolve({term!r}) -> {r}")
        if not r:
            print(f"  suggest: {suggest(term, top_k=3)}")
