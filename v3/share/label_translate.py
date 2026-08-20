"""Resolver: từ tiếng Việt (người dùng gõ) -> nhãn OpenImages V4 tiếng Anh (Objects dùng).

3 lớp, độ tin cậy giảm dần:
  1. Khớp CHÍNH XÁC với bản dịch (label_vi.json) — chắc chắn.
  2. Từ điển đồng nghĩa tay (SYNONYMS) — chắc chắn, nhưng phải thêm tay từng trường hợp.
  3. Substring 2 chiều — khá chắc (biến thể ngắn/dài của cùng cụm từ).

KHÔNG dùng CLIP text-text similarity làm auto-filter: đã kiểm chứng (2026-08-05) similarity
quá sát nhau giữa mọi cặp cụm từ ngắn (0.94-0.99), "cún con" xếp hạng gần nhất toàn nhãn sai
(Linh dương, Cú, Cây...) — silent-wrong-nhưng-trông-tự-tin, đúng loại lỗi cần tránh (gap
honesty). Từng có suggest() dùng CLIP text-text similarity gợi ý khi resolve() không khớp -
ĐÃ XOÁ (2026-08-20, theo yêu cầu người dùng, xác nhận 0 caller thật) vì đúng lý do nói trên:
similarity quá sát nhau nên gợi ý không đáng tin, và không có UI nào thật sự hiển thị nó.
"""
from __future__ import annotations

import json
import unicodedata

from config import INDEX_DIR

LABEL_VI_PATH = INDEX_DIR / "label_vi.json"
LABEL_SYNONYMS_PATH = INDEX_DIR / "label_synonyms.json"
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


def list_labels_vi() -> list[str]:
    """Danh sach nhan tieng Viet CHINH THUC (514 nhan closed-set BTC, label_vi.json) - dung
    cho UI selectbox (2026-08-15, theo yeu cau nguoi dung: chan nguoi dung go nhan ngoai
    khong resolve duoc) thay vi text_input tu do. Sap xep theo bang chu cai tieng Viet
    (NFC-normalize) - KHONG bao gom open-vocab/synonym (chi 514 nhan chinh, chac chan
    resolve() tra dung 1 nhan)."""
    _load()
    assert _label_vi is not None
    return sorted(_label_vi.values(), key=lambda s: unicodedata.normalize("NFC", s.lower()))


def find_known_terms(text_vi: str) -> dict[str, str]:
    """Quet CA CAU (khong phai 1 tu/cum rieng le nhu resolve()) tim cac tu/cum tieng Viet co
    trong tu dien da biet (label_vi.json + SYNONYMS + label_synonyms.json) xuat hien lam
    SUBSTRING (word-boundary) - dung lam glossary hint cho LLM dich (2026-08-15, xem
    query_distill.py) de tranh LLM tu doan sai thuat ngu dac thu van hoa/domain (vd "lân"
    -> bi doan nham thanh "ox" + bia them canh, thay vi dung "lion dance costume" da co san
    trong SYNONYMS). Nguong do dai >=3 ky tu (long hon nguong 4 cua resolve() vi day chi la
    GOI Y cho LLM, khong phai auto-apply hard-filter, chap nhan rong hon mot chut).

    SUA BUG THAT (2026-08-15, phat hien qua test tong quat hoa): ban dau chi filter substring
    THO, KHONG loai overlap - "bàn tay" (hand) bi tach nham thanh "bàn" (Table) + "tay" (Human
    hand) 2 hit RIENG LE (dung LOP BUG COMPOUND NOUN da biet o query_planner.py SYSTEM_PROMPT -
    tu dau tien cua 1 cum ghep lai la 1 danh tu doc lap voi nghia KHAC). Fix: GREEDY LONGEST-
    MATCH-FIRST, khong cho 2 hit CHONG LAN vi tri ky tu trong cau goc - "bàn tay" (dai hon,
    khop truoc) se LOAI "bàn"/"tay" rieng le khoi ket qua."""
    _load()
    assert _vi_norm_to_label is not None
    q_norm = _normalize(text_vi)
    qp = f" {q_norm} "

    candidates = []  # (start, end, vi_norm, en) - vi tri TU CHINH trong qp (KHONG tinh 2 khoang
    # trang dem 2 dau - 2 tu lien nhau dung chung 1 ky tu khoang trang lam ranh gioi, neu tinh
    # ca khoang trang do vao span se bao overlap GIA giua 2 tu THAT SU khac nhau, vd "bàn" va
    # "tay" trong "bàn tay" dung chung 1 dau cach o giua - da phat hien qua test).
    for vi_norm, en in _vi_norm_to_label.items():
        if len(vi_norm) < 3:
            continue
        needle = f" {vi_norm} "
        start = qp.find(needle)
        while start != -1:
            candidates.append((start + 1, start + 1 + len(vi_norm), vi_norm, en))
            start = qp.find(needle, start + 1)

    # greedy: uu tien cum DAI HON truoc (nhieu ky tu hon = it mo ho hon), bo qua cum nao CHONG
    # LAN vi tri voi 1 cum da chon (vd "tay" trong "bàn tay" khong duoc chon rieng vi "bàn tay"
    # da chiem dung vi tri do).
    candidates.sort(key=lambda c: -(c[1] - c[0]))
    taken_spans: list[tuple[int, int]] = []
    hits: dict[str, str] = {}
    for start, end, vi_norm, en in candidates:
        if any(start < t_end and end > t_start for t_start, t_end in taken_spans):
            continue
        taken_spans.append((start, end))
        hits[vi_norm] = en
    return hits


# 2026-08-18 (theo yeu cau nguoi dung, phat hien qua case that "rổ xoài xanh" KHONG resolve
# duoc du tu chinh "xoài" thuc su co nhan "Mango" (label_vi.json dich la "quả xoài") - cum LLM
# trich co CA tien to phan loai ("rổ") LAN tinh tu bo sung ("xanh") bao quanh danh tu CHINH,
# khien containment 2 chieu (duoi) FAIL CA 2 PHIA (khac cac bug tung fix truoc do - vd "áo
# xanh" - chi lech 1 phia la du sua). DA TU KIEM CHUNG huong "tach tung tu roi thu resolve rieng
# le" la KHONG AN TOAN: "hành"->"Luggage and bags", "tím"/"xanh"->"Zucchini", "vàng"->"Lemon",
# "tây"->"Turkey" - tu MAU SAC/tinh tu TRUNG NGAU NHIEN voi nhan hoan toan khong lien quan,
# dung LOAI BUG "cây" da biet (xem comment trong resolve()). Fix AN TOAN HON: CHI boc 1 TIEN TO
# LOAI TU/phan loai đã biết (loại từ tiếng Việt - xem query_planner.py SYSTEM_PROMPT muc
# "Vietnamese classifier words") o DAU cum - KHONG dung cho tinh tu mau sac/kich thuoc (rui ro
# da chung minh o tren) - roi thu lai CHINH XAC logic containment, KHONG doan mo gi them.
_CLASSIFIER_PREFIXES = ("rổ ", "giỏ ", "quả ", "trái ", "chiếc ", "cái ", "con ", "tấm ", "túi ", "cây ", "củ ")


def _strip_classifier_prefix(s: str) -> str:
    for p in _CLASSIFIER_PREFIXES:
        if s.startswith(p):
            return s[len(p):]
    return s


def _containment_hits(q: str, strip_labels: bool = False) -> list[tuple[str, str]]:
    """Substring 2 chiều theo TỪ (word-boundary), không phải ký tự thô — tránh case
    "áo" (2 ký tự) khớp nhầm vào giữa từ "báo" trong "biển báo giao thông" (2026-08-06,
    bug thật: query "áo xanh" -> resolve ra "Traffic sign"/"Sports uniform"). Đệm khoảng
    trắng 2 đầu để containment check luôn rơi đúng ranh giới từ. Từ quá ngắn (<3 ký tự,
    vd "áo", "xe") không đủ đặc trưng để substring match — bỏ qua, tránh false-positive.
    Tách riêng khỏi resolve() (2026-08-18) để dùng LẠI cho cả bản gốc LẪN bản fallback đã bóc
    tiền tố phân loại (xem _strip_classifier_prefix/resolve).

    strip_labels: mặc định False (GIỮ NGUYÊN 100% hành vi gốc, dùng cho lượt khớp CHÍNH) — BUG
    THẬT tự phát hiện khi test (2026-08-18, lần 1): bật cờ này ở lượt chính làm "áo xanh"
    resolve NHẦM ra "Tree" (nhãn "cây xanh" bị bóc "cây " thành "xanh", rồi " xanh " lọt vào
    " áo xanh " — TÁI PHÁT đúng bug lịch sử "áo"/"báo" 2026-08-06). CHỈ bật True ở lượt FALLBACK
    (resolve() — sau khi lượt chính đã thất bại VÀ chính query cũng có tiền tố phân loại).

    Khi strip_labels=True: BẮT BUỘC phần cốt lõi (đã bóc tiền tố phân loại của CẢ query lẫn
    nhãn) phải khớp Ở ĐẦU CỤM (prefix), KHÔNG chấp nhận khớp ở BẤT KỲ đâu như lượt chính — BUG
    THẬT tự phát hiện khi test (2026-08-18, lần 2): dùng containment-ở-bất-kỳ-đâu vẫn làm
    "rổ xoài xanh" (bóc "rổ" -> "xoài xanh") khớp NHẦM THÊM "Tree" (nhãn "cây xanh" bóc "cây"
    -> "xanh", " xanh " nằm Ở CUỐI " xoài xanh ") — CÙNG LOẠI BUG, chỉ dịch từ lượt chính sang
    fallback. Neo prefix là AN TOÀN vì tiếng Việt: danh từ CHÍNH luôn đứng NGAY SAU từ phân loại
    (đầu cụm còn lại sau khi bóc), tính từ bổ nghĩa (màu/kích thước...) luôn đứng SAU CÙNG — nên
    chỉ cốt lõi Ở ĐẦU mới đáng tin, cốt lõi Ở CUỐI/GIỮA (như "xanh") gần như luôn là tính từ,
    không phải danh từ chính."""
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
        vi_core = _strip_classifier_prefix(vi_norm) if strip_labels else vi_norm
        if strip_labels and len(vi_core) < 4:
            continue  # sau khi boc co the ngan lai duoi nguong an toan - loai luon, khong xet
        vip = f" {vi_core} "
        if strip_labels:
            # prefix THEO TU (khong phai ky tu tho) - dung qp/vip da co san khoang trang dem 2
            # dau, giong nguyen tac word-boundary cua nhanh khong-strip ben duoi, chi doi
            # "chua o bat ky dau" thanh "BAT DAU bang" (xem docstring ve ly do neo prefix).
            if qp.startswith(vip) or vip.startswith(qp):
                hits.append((vi_core, en))
            continue
        if qp in vip or vip in qp:
            hits.append((vi_core, en))
    return hits


def _majority_label(hits: list[tuple[str, str]]) -> list[str]:
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

    hits = _containment_hits(q)
    if hits:
        return _majority_label(hits)

    # 2026-08-18 (xem _strip_classifier_prefix/_containment_hits docstring) - CHI thu lai neu
    # QUERY boc duoc gi do (khac voi q ban dau, tranh quet lai y het lan dau vo ich khi khong
    # co tien to nao khop) - dieu kien nay TU THU HEP be mat rui ro cua strip_labels=True (chi
    # kich hoat khi CHINH query cung bat dau bang 1 tien to phan loai, khong phai MOI query).
    q_stripped = _strip_classifier_prefix(q)
    if q_stripped != q:
        hits2 = _containment_hits(q_stripped, strip_labels=True)
        if hits2:
            return _majority_label(hits2)
    return []


# 2026-08-20 (theo yeu cau nguoi dung: "CLIP_TEXT_MODEL_NAME ... mình không còn dùng cái này
# nữa") - XOA suggest()/_load_embeddings() (goi y mo theo similarity CLIP text khi resolve()
# khong khop chinh xac) - da xac nhan 0 caller thuc su trong toan repo (chi __main__ demo o duoi
# tung goi, da sua). resolve() (ben tren) KHONG can - chi dung containment/synonym string match,
# khong dinh gi model nao.
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
