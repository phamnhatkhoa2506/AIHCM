"""TẦNG 1 — Lọc thô: định nghĩa tập ứng viên (video_id, local_idx) TRƯỚC khi vào Tầng 2.
Chỉ include/exclude, KHÔNG tính điểm — filter càng chặt thì Tầng 2 càng ít việc.

3 nguồn lọc, dùng riêng hoặc cùng lúc:
  - by_metadata(): video_metadata.parquet (kênh/ngày đăng/từ khoá) — lọc theo VIDEO.
  - by_objects():  objects_index.parquet (nhãn OpenImages V4, đã lọc score>=0.3) — lọc theo FRAME.
  - by_text():     ocr_text.parquet (chữ trên màn hình, OCR) — lọc theo FRAME.

apply() gộp cả 3 (AND) thành 1 kết quả duy nhất. Trả về:
  - None                         -> không lọc gì, Tầng 2 tìm trên toàn corpus.
  - set[(video_id, local_idx)]   -> đúng tập frame được phép (có thể rỗng nếu bộ lọc quá hẹp).
"""
from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd

import resources


def _strip_accents(s: str) -> str:
    """Bỏ dấu tiếng Việt + hạ chữ thường — dùng để so khớp 'mon ngon' với 'Món Ngon'.
    keywords_text đã có sẵn bản sao không dấu trộn lẫn (dữ liệu gốc từ YouTube tags), nhưng
    title thì KHÔNG — nếu chỉ .lower() thôi thì gõ không dấu sẽ không khớp được title."""
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d")  # "đ" không tách qua NFD như các ký tự có dấu phụ khác


_GLUE_MAX_SPAN = 4  # gop toi da 4 tu lien tiep thanh 1 "sieu tu" khi thu khop dinh - du cho
# cum dai nhat tung gap (vd "TTTM QUOC THAI" = 3 tu), khong qua rong de tranh khop nham.


def _flexible_word_match_score(needle_words: list[str], haystack_words: list[str]) -> tuple[bool, float]:
    """Thuat toan khop TU tong quat (2026-08-17) - hop nhat 2 lop fix truoc do (2026-08-15,
    2026-08-17) thanh 1 thuat toan DUY NHAT, ro rang hon: 2 con tro quet dong thoi qua
    needle_words/haystack_words (GIU THU TU, khong phai bag-of-words - tranh khop nham cau dao
    nghia), o MOI vi tri thu 3 kieu khop, uu tien tu tren xuong:
      1. Khop TUYET DOI 1-1 (needle_words[i] == haystack_words[j]).
      2. Khop GOP-NEEDLE: N tu LIEN TIEP cua needle (N=2..4) noi lien (bo khoang trang) BANG
         DUNG 1 tu haystack - xu ly ca OCR DINH nhieu tu needle thanh 1 token (vd needle "an
         hoa tu" khop haystack "anhoatu" 1 token duy nhat).
      3. Khop GOP-HAYSTACK: 1 tu needle BANG DUNG N tu LIEN TIEP cua haystack noi lien (N=2..4)
         - xu ly chieu nguoc lai (it gap hon, van de phong khi OCR tach 1 tu needle ra nhieu
         manh).
    Khong khop o vi tri nao ca -> BO QUA 1 tu haystack (tolerate tu la/chen giua khong lien
    quan, giu nguyen tinh chat "khong bat buoc lien ke" da co tu ban 2026-08-15).

    Vi sao AN TOAN, khong tai phat bug "áo" lot vao "báo" (2026-08-06) hay bug "cua khau" lot
    nham (gia thuyet): khop gop CHI xay ra giua CAC TU DA TACH SAN (list, khong phai ky tu tho
    trong 1 chuoi) - "áo" (needle 1 tu) khong bao gio duoc thu ghep VOI CHINH NO (can >=2 tu de
    kich hoat lop 2/3), va "bao" (1 tu haystack) khong the "gop" thanh "áo" duoc vi gop la NOI
    THEM tu ke ben, khong phai cat GIUA 1 tu co san - loai bo hoan toan kha nang substring gia
    tao truoc day.

    2026-08-17 (theo yeu cau nguoi dung: "nên tinh chỉnh theo độ khớp OCR, phải có score") -
    tra THEM diem "do GON" cua cum khop = so tu haystack THAT SU tieu thu / do dai doan
    haystack chua chung (span) - vd khop "cuakhau"+"longbinh"+"4.8km" LIEN TIEP nhau (span=3,
    tieu thu 3) -> gon = 1.0; neu co chu KHONG LIEN QUAN chen GIUA cac tu da khop (vd 1 bien
    hieu khac o gan do bi OCR gop vao giua) -> span > tieu thu -> gon < 1.0. Dung lam SOFT
    BOOST xep hang GIUA cac frame DA QUA hard-filter (KHONG dung de loai - van chi 1 nguong
    duy nhat la match=True/False, xem _ordered_words_match/_ocr_box_candidates)."""
    i, j = 0, 0
    n, m = len(needle_words), len(haystack_words)
    first_j: int | None = None
    last_j: int | None = None
    consumed = 0
    while i < n and j < m:
        step_span = 1
        step_matched = False
        if haystack_words[j] == needle_words[i]:
            i += 1
            step_matched = True
        else:
            for k in range(2, _GLUE_MAX_SPAN + 1):
                if i + k <= n and haystack_words[j] == "".join(needle_words[i:i + k]):
                    i += k
                    step_matched = True
                    break
            if not step_matched:
                for k in range(2, _GLUE_MAX_SPAN + 1):
                    if j + k <= m and needle_words[i] == "".join(haystack_words[j:j + k]):
                        step_span = k
                        i += 1
                        step_matched = True
                        break
        if step_matched:
            if first_j is None:
                first_j = j
            j += step_span
            last_j = j - 1
            consumed += step_span
            continue
        j += 1  # tu haystack nay khong lien quan - bo qua, tiep tuc tim tu needle[i]
    if i != n:
        return False, 0.0
    span = (last_j - first_j + 1) if first_j is not None else 0
    compactness = (consumed / span) if span > 0 else 1.0
    return True, compactness


OCR_MATCH_MIN_COVERAGE = 0.75  # 2026-08-17 (theo yeu cau nguoi dung "có thuật toán nào mềm
# dẻo hơn không") - nguong PHU TOI THIEU de VAN COI LA KHOP (hard-filter), THAY vi doi DU
# 100% so tu nhu truoc (_flexible_word_match_score). Vd needle 4 tu duoc phep THIEU 1 tu (vd
# "TTTM QUỐC THÁI 4km" thieu dung "4km" do PaddleOCR bo sot - bug that phat hien 2026-08-17) ma
# VAN qua duoc hard-filter. Needle NGAN (1-2 tu) van GAN NHU doi du 100% MOT CACH TU NHIEN (lam
# tron: 1 tu can >=0.75 -> phai co ca 1/1; 2 tu can >=1.5 -> lam tron len 2/2) - khong can xu ly
# rieng, tranh khop qua long cho cau ngan (van giu nguyen tac "cang ngan cang can chac chan"
# da co xuyen suot du an).


def _flexible_word_coverage_score(needle_words: list[str], haystack_words: list[str]) -> tuple[float, float]:
    """Ban "MEM DEO" cua _flexible_word_match_score() - KHONG doi hoi khop DU 100% so tu, tu
    needle nao KHONG tim thay o BAT KY dau trong PHAN HAYSTACK CON LAI thi bi COI LA MISS va
    BO QUA (khong chan cac tu needle SAU no bi anh huong day, khac han han che cua ban goc -
    ban goc "ket" cung tai vi tri dau tien khong tim duoc, khong bao gio thu tiep tu ke tiep).
    Van GIU DUNG THU TU cho cac tu THAT SU khop duoc (khong phai bag-of-words).

    Tra (coverage, compactness):
      - coverage = so TU NEEDLE khop duoc / tong so tu needle (0.0 neu khong tu nao khop).
      - compactness = so TU HAYSTACK thuc su tieu thu / do dai doan haystack chua chung (giong
        _flexible_word_match_score, nhung tinh RIENG tren cac tu DA khop duoc, bo qua tu MISS)."""
    n, m = len(needle_words), len(haystack_words)
    if n == 0:
        return 1.0, 1.0
    i, j = 0, 0
    matched_needle_words = 0
    consumed_haystack_tokens = 0
    first_j: int | None = None
    last_j: int | None = None
    while i < n:
        j_start = j
        scan_j = j
        step_span_needle = 1
        step_span_haystack = 1
        step_matched = False
        while scan_j < m:
            if haystack_words[scan_j] == needle_words[i]:
                step_matched = True
                break
            for k in range(2, _GLUE_MAX_SPAN + 1):
                if i + k <= n and haystack_words[scan_j] == "".join(needle_words[i:i + k]):
                    step_span_needle = k
                    step_matched = True
                    break
                if scan_j + k <= m and needle_words[i] == "".join(haystack_words[scan_j:scan_j + k]):
                    step_span_haystack = k
                    step_matched = True
                    break
            if step_matched:
                break
            scan_j += 1
        if step_matched:
            if first_j is None:
                first_j = scan_j
            j = scan_j + step_span_haystack
            last_j = j - 1
            matched_needle_words += step_span_needle
            consumed_haystack_tokens += step_span_haystack
            i += step_span_needle
        else:
            i += 1  # tu needle nay KHONG TON TAI o dau trong phan haystack con lai - bo qua,
            j = j_start  # khong tieu thu gi ca, thu tiep tu needle KE TIEP tu DUNG vi tri nay

    span = (last_j - first_j + 1) if first_j is not None else 0
    compactness = (consumed_haystack_tokens / span) if span > 0 else 0.0
    coverage = matched_needle_words / n
    return coverage, compactness


def _flexible_word_match(needle_words: list[str], haystack_words: list[str]) -> bool:
    """True neu coverage >= OCR_MATCH_MIN_COVERAGE - xem _flexible_word_coverage_score() cho
    diem chi tiet (dung khi can ca coverage LAN compactness, vd lam soft-boost xep hang)."""
    coverage, _compactness = _flexible_word_coverage_score(needle_words, haystack_words)
    return coverage >= OCR_MATCH_MIN_COVERAGE


def _ordered_words_match(needle: str, haystack_norm: str, algorithm: str = "flexible") -> bool:
    """Wrapper 1-chuoi-vs-1-chuoi cho thuat toan khop TU (mac dinh "flexible") - dung khi
    haystack la 1 dong text_norm DUY NHAT (vd ASR/by_text() ban BTC cu). Voi OCR bo dense
    (nhieu box/frame co the chua 1 cum chu bi CHIA thanh nhieu dong rieng), dung THANG thuat
    toan tren danh sach tu da GOP TU NHIEU BOX cua CUNG 1 frame - xem
    dense_search.py::_ocr_frame_words. Xem OCR_MATCH_ALGORITHMS cho cac lua chon khac."""
    needle_words = _strip_accents(needle).split()
    if not needle_words:
        return True
    matched, _score = match_words(algorithm, needle_words, haystack_norm.split())
    return matched


# ---------------------------------------------------------------------------
# 2026-08-17 (theo yeu cau nguoi dung: "implement các thuật toán này rồi trên giao diện sẽ để
# các nút để người dùng chọn thuật toán tự chọn") - 3 THUAT TOAN khop tu CO THE CHON o UI,
# dung CHUNG 1 interface (needle_words, haystack_words) -> (matched: bool, score: float) de
# dense_search.py/app.py dispatch qua OCR_MATCH_ALGORITHMS ma khong can biet chi tiet ben trong.
# score dung lam soft-boost xep hang (cang cao cang "gon"/giong), matched dung cho hard-filter.
# ---------------------------------------------------------------------------

RAPIDFUZZ_MIN_SCORE = 0.72  # nguong tren token_sort_ratio (0..1) - hieu chinh boi vi
# token_sort_ratio khong doi hoi dung THU TU/vi tri nhu 2 thuat toan con lai, nen can nguong
# cao hon 1 chut de tranh khop qua long (vd hoan vi tu vo nghia van duoc diem cao).

ALIGNMENT_MIN_SCORE = 0.6  # nguong tren diem alignment chuan hoa (0..1, xem match_alignment).
_ALIGN_GAP_HAYSTACK = 0.3  # phat khi BO QUA 1 tu haystack (chen tu la khong lien quan)
_ALIGN_GAP_NEEDLE = 1.0  # phat khi BO QUA 1 tu needle (needle THAT SU thieu trong haystack)

_MIN_FUZZY_WORD_LEN = 3  # BUG THAT phat hien khi TU KIEM TRA truoc khi giao nguoi dung (2026-
# 08-17): ca rapidfuzz LAN alignment (diem tuong dong KY TU, khong phai khop tuyet doi) tai
# phat DUNG bug lich su "áo" lot vao "báo" (2026-08-06, xem _flexible_word_match_score docstring)
# - vd sim("ao","bao")=0.8 (difflib/rapidfuzz DEU cho diem nay) GAN NHU BANG sim("an hoa tu" vs
# "anhoatu")=0.875 (truong hop dung, can khoan dung) - KHONG THE tach 2 truong hop bang 1 NGUONG
# DUY NHAT tren toan cum. Fix: tu needle NGAN (<3 ky tu, de nham nhat) BAT BUOC xuat hien
# NGUYEN VEN (1-1, khong qua fuzzy) trong haystack_words - xem _short_words_verbatim_ok(). Danh
# doi: 2 thuat toan nay vi vay KHONG khoan dung viec OCR dinh/tach RIENG cac tu ngan (vd needle
# "an hoa tu" bi dinh thanh "anhoatu" se KHONG qua duoc guard nay o rapidfuzz/alignment nua) -
# chap nhan duoc vi do la the manh cua thuat toan MAC DINH (flexible, co lop gop tu rieng), 2
# thuat toan nay chi bo sung KHA NANG CHIU LOI KY TU (khong phai gluing) cho tu DU DAI.


def _short_words_verbatim_ok(needle_words: list[str], haystack_words: list[str]) -> bool:
    """True neu MOI tu needle NGAN (<_MIN_FUZZY_WORD_LEN ky tu) xuat hien NGUYEN VEN (dung 1
    tu, khong fuzzy) trong haystack_words - xem _MIN_FUZZY_WORD_LEN. Dung lam GUARD BO SUNG cho
    match_rapidfuzz/match_alignment (khong ap dung cho match_flexible - thuat toan do da AN
    TOAN san, xem _flexible_word_match_score docstring)."""
    haystack_set = set(haystack_words)
    return all(w in haystack_set for w in needle_words if len(w) < _MIN_FUZZY_WORD_LEN)


def match_flexible(needle_words: list[str], haystack_words: list[str]) -> tuple[bool, float]:
    """Thuat toan MAC DINH (don gian nhat, dang dung tu 2026-08-17) - xem
    _flexible_word_coverage_score()/OCR_MATCH_MIN_COVERAGE. score tra ve = coverage*compactness
    de vua phan anh do day du vua phan anh do gon khi so sanh giua nhieu ket qua."""
    coverage, compactness = _flexible_word_coverage_score(needle_words, haystack_words)
    return coverage >= OCR_MATCH_MIN_COVERAGE, coverage * compactness


def match_rapidfuzz(needle_words: list[str], haystack_words: list[str]) -> tuple[bool, float]:
    """RapidFuzz (thu vien C++ toi uu, da co san trong requirements) - dung
    fuzz.token_sort_ratio TREN CA CUM (khong phai tung tu rieng) de dung nguong Levenshtein-
    based, chiu duoc LOI KY TU (OCR nham chu, vd "hoa" thanh "hoq") ma thuat toan flexible
    (khop tuyet doi tung tu) khong xu ly duoc - danh doi: khong giu chat thu tu/vi tri nhu
    flexible/alignment, nen chi dung khi nguoi dung chu dong chon (khong phai mac dinh)."""
    from rapidfuzz import fuzz

    needle = " ".join(needle_words)
    haystack = " ".join(haystack_words)
    if not needle:
        return True, 1.0
    score = fuzz.token_sort_ratio(needle, haystack) / 100.0
    matched = score >= RAPIDFUZZ_MIN_SCORE and _short_words_verbatim_ok(needle_words, haystack_words)
    return matched, score


RAPIDFUZZ_TOKEN_MIN_SCORE = 0.75  # nguong tren fuzz.ratio() TUNG TOKEN rieng le (0..1) - xem
# match_rapidfuzz_token. Cao hon RAPIDFUZZ_MIN_SCORE (0.72) mot chut vi day la so 1-1 voi 1
# token (chinh xac hon, it nhieu tu khac chen vao), co the doi hoi khat khe hon 1 chut.

_MIN_FUZZY_GLUED_LEN = 6  # BUG THAT tu phat hien khi test (2026-08-18): _short_words_verbatim_
# ok (dung cho match_rapidfuzz) KHONG hop voi match_rapidfuzz_token - guard do doi tu needle
# NGAN phai dung 1 MINH lam 1 token rieng trong haystack, nhung dung y CHINH cua thuat toan
# nay la needle bi OCR GOM CA CUM thanh 1 TOKEN DUY NHAT (vd "xu" trong "CHÙA XỨ..." khong bao
# gio dung rieng, luon dinh lien "chuo-xu.thonhmieu") - ap dung guard cu se LOAI OAN chinh case
# thuat toan nay sinh ra de xu ly. Fix: doi guard sang do dai CHUOI NEEDLE DA GLUE (khong phai
# tung tu rieng) - chuoi glued qua NGAN (vd "ao" 2 ky tu) khong du dac trung de tin tuong diem
# fuzzy (da chung minh "ao" vs "bao" van duoc 80% - qua giong nguong that), bat buoc KHOP CHINH
# XAC; chuoi glued DU DAI (vd "chuaxuthanhmieu" 15 ky tu) moi duoc tin tuong qua nguong fuzzy.


def match_rapidfuzz_token(needle_words: list[str], haystack_words: list[str]) -> tuple[bool, float]:
    """RapidFuzz PER-TOKEN (2026-08-18, theo yeu cau nguoi dung sau khi phat hien case that:
    OCR doc "CHÙA XỨ THÁNH MIẾU" thanh 1 TOKEN DUY NHAT dinh loi ky tu "chuo-xu.thonhmieu" NAM
    GIUA cac token khac khong lien quan ("loivao","hty") tren cung 1 frame - xem hoi thoai) -
    KHAC match_rapidfuzz o cho SO SANH TUNG TOKEN HAYSTACK RIENG LE (fuzz.ratio, glue needle
    thanh 1 chuoi) thay vi gop CA CHUOI haystack roi tinh token_sort_ratio 1 lan.

    BUG THAT match_rapidfuzz mac phai (da do): so ca chuoi gop khien 1 token khop THAT (81%
    neu so RIENG voi "chuaxuthanhmieu") bi PHA LOANG boi cac token thua KHONG LIEN QUAN trong
    cung frame, tut xuong con ~42% - duoi nguong, mat tin hieu that. Fix: KHONG gop haystack,
    lay diem CAO NHAT trong so sanh TUNG token rieng - giu duoc tin hieu 81% du frame co bao
    nhieu token khac chen vao. Danh doi: CHI phu hop khi needle bi OCR GOM THANH 1 TOKEN DUY
    NHAT (giong "CỬA KHẨU"/"LỐI VÀO" cac case truoc) - neu needle THAT SU tach thanh NHIEU
    token rieng biet trong haystack (khong glue), thuat toan nay KHONG bat duoc (dung
    match_flexible/match_rapidfuzz cho truong hop do).

    Dung LAI guard _short_words_verbatim_ok giong het match_rapidfuzz (tranh tai phat bug
    "áo" lot vao "báo" cho tu ngan)."""
    from rapidfuzz import fuzz

    needle_glued = "".join(needle_words)
    if not needle_glued:
        return True, 1.0
    best = 0.0
    for tok in haystack_words:
        r = fuzz.ratio(needle_glued, tok) / 100.0
        if r > best:
            best = r
    # xem _MIN_FUZZY_GLUED_LEN - guard theo DO DAI CHUOI DA GLUE (khong phai tung tu nhu
    # match_rapidfuzz), phu hop dung y thuat toan nay (needle bi OCR gom thanh 1 token).
    if len(needle_glued) < _MIN_FUZZY_GLUED_LEN:
        matched = needle_glued in haystack_words
    else:
        matched = best >= RAPIDFUZZ_TOKEN_MIN_SCORE
    return matched, best


def match_alignment(needle_words: list[str], haystack_words: list[str]) -> tuple[bool, float]:
    """Sequence Alignment kieu Needleman-Wunsch (global alignment, quy hoach dong O(n*m)) - moi
    tu needle duoc CO GANG gan (align) voi 1 tu haystack theo DUNG THU TU, cho phep BO QUA
    (gap) o CA 2 phia voi chi phi khac nhau:
      - bo qua 1 tu HAYSTACK (tu la chen giua, khong lien quan) - phat NHE (_ALIGN_GAP_HAYSTACK).
      - bo qua 1 tu NEEDLE (that su thieu trong haystack) - phat NANG (_ALIGN_GAP_NEEDLE).
    Diem khop 2 tu dung difflib.SequenceMatcher.ratio() (0..1, muc do giong ky tu) THAY VI chi
    ==, nen chiu duoc OCR dinh/tach tu nhe (vd "cuakhau" vs "cua" van co diem > 0) - nguyen tac
    hon 2 thuat toan kia (co co so ly thuyet ro rang thay vi heuristic thu cong), nhung O(n*m)
    nen CHAM hon tren haystack dai (news-ticker OCR co the toi 357 tu/frame).
    Tra (matched, score) voi score = dp[n][m] / n (chuan hoa theo do dai needle)."""
    import difflib

    n, m = len(needle_words), len(haystack_words)
    if n == 0:
        return True, 1.0
    if m == 0:
        return False, 0.0

    dp = np.zeros((n + 1, m + 1), dtype=float)
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] - _ALIGN_GAP_NEEDLE
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] - _ALIGN_GAP_HAYSTACK

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sim = difflib.SequenceMatcher(None, needle_words[i - 1], haystack_words[j - 1]).ratio()
            align_score = dp[i - 1][j - 1] + sim
            skip_haystack = dp[i][j - 1] - _ALIGN_GAP_HAYSTACK
            skip_needle = dp[i - 1][j] - _ALIGN_GAP_NEEDLE
            dp[i][j] = max(align_score, skip_haystack, skip_needle)

    score = max(dp[n][m] / n, 0.0)
    matched = score >= ALIGNMENT_MIN_SCORE and _short_words_verbatim_ok(needle_words, haystack_words)
    return matched, score


OCR_MATCH_ALGORITHMS: dict[str, tuple[str, object]] = {
    # key -> (ten hien thi tren UI, ham khop) - "flexible" PHAI la mac dinh (don gian nhat,
    # theo dung yeu cau nguoi dung "mặc định là cái dễ nhất").
    "flexible": ("Đơn giản (mặc định)", match_flexible),
    "rapidfuzz": ("RapidFuzz (chịu lỗi ký tự)", match_rapidfuzz),
    "rapidfuzz_token": ("RapidFuzz theo từng token (chịu lỗi ký tự, không bị pha loãng)", match_rapidfuzz_token),
    "alignment": ("Sequence Alignment (Needleman-Wunsch)", match_alignment),
}
DEFAULT_OCR_ALGORITHM = "flexible"


def match_words(
    algorithm: str, needle_words: list[str], haystack_words: list[str]
) -> tuple[bool, float]:
    """Dispatch chung - dense_search.py/app.py chi can truyen key (vd tu st.radio) ma khong
    can import/goi truc tiep tung ham thuat toan."""
    _label, fn = OCR_MATCH_ALGORITHMS.get(algorithm, OCR_MATCH_ALGORITHMS[DEFAULT_OCR_ALGORITHM])
    return fn(needle_words, haystack_words)


def by_metadata(
    authors: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    keywords_any: list[str] | None = None,
) -> set[str] | None:
    """Lọc theo video_metadata (kênh/ngày/từ khoá/tên chương trình) -> tập video_id được phép.

    keywords_any so khớp CẢ title lẫn keywords_text (substring, không dấu) — gộp làm 1 ô
    "từ khoá" duy nhất ở UI thay vì tách riêng tìm-theo-tên-chương-trình, vì ở quy mô 873
    video thì không cần bộ máy full-text search riêng (BM25/inverted index) — substring quét
    toàn bộ đã đủ nhanh (<10ms) và đây là lọc CỨNG (có/không), không phải xếp hạng liên quan."""
    if not authors and not date_from and not date_to and not keywords_any:
        return None

    df = resources.get().video_meta
    if authors:
        df = df[df["author"].isin(authors)]
    if date_from:
        df = df[df["publish_date"] >= pd.to_datetime(date_from)]
    if date_to:
        df = df[df["publish_date"] <= pd.to_datetime(date_to)]
    if keywords_any:
        # dem khoang trang 2 dau ca needle va haystack de containment check roi dung ranh gioi
        # TU, khong phai ky tu tho - bug that phat hien 2026-08-09: "tin" (tin tuc) lot vao
        # giua "tinh" trong "tinh nguyen" (tinh nguyen = volunteer, khong lien quan tin tuc)
        # sau khi bo dau, giong het lop bug da vay o label_translate.py ("ao" lot vao "bao").
        needles = [f" {_strip_accents(k)} " for k in keywords_any]
        haystack = " " + df["title"].apply(_strip_accents) + " " + df["keywords_text"].apply(_strip_accents) + " "
        df = df[haystack.apply(lambda h: any(n in h for n in needles))]
    return set(df["video_id"])


def by_objects(
    must_have_labels: list[str] | None = None,
    min_count: dict[str, int] | None = None,
    use_suppression: bool = True,
    include_open_vocab: bool = True,
) -> set[tuple[str, int]] | None:
    """Lọc theo Objects (nhãn OpenImages V4 tiếng Anh) -> tập (video_id, local_idx) được phép.

    Đây là bộ lọc CỨNG, đúng nguyên tắc: filter định nghĩa tập ứng viên TRƯỚC, Tầng 2 (vector)
    chỉ xếp hạng TRONG tập đó — không phải vector trước rồi giao (mất recall với query có
    ràng buộc chặt, vd "2 người" mà CLIP không xếp hạng cao).
    """
    if not must_have_labels and not min_count:
        return None

    objects_index = resources.get().objects_index
    if objects_index is None:
        raise RuntimeError("objects_index.parquet chưa build — chạy build_objects_index.py trước")

    need: dict[str, int] = {}
    for lb in must_have_labels or []:
        need[lb] = max(need.get(lb, 1), 1)
    for lb, c in (min_count or {}).items():
        need[lb] = max(need.get(lb, 1), c)

    # include_open_vocab=False -> CHI khop tren closed_set (514 nhan goc BTC), bo qua han
    # nhan open_vocab_dino - dung de so sanh/fallback ve dung hanh vi truoc khi co Grounding
    # DINO. Khac voi use_suppression: cai do chi tat co che LOAI closed_set sai, van cho phep
    # open_vocab tham gia khop; cai nay tat HAN open_vocab, khong lien quan suppress.
    scope = objects_index if include_open_vocab else objects_index[objects_index["source"] == "closed_set"]

    # cot "suppressed" (2026-08-07): nhan closed_set nao bi open-vocab (Grounding DINO) de len
    # (IoU cao, cung vi tri) coi nhu SAI, khong duoc dung de khop nua - xem
    # resources._mark_suppressed_closed_rows(). Dong van con nguyen (audit), chi loai khoi day.
    # use_suppression=False -> tat han co che nay (fallback ve hanh vi cu, dung khi muon so
    # sanh/nghi ngo suppress sai) - KHONG can build lai gi, chi doi 1 cot boolean luc filter.
    not_suppressed = ~scope["suppressed"] if use_suppression else True

    allowed: set[tuple[str, int]] | None = None
    for lb, min_c in need.items():
        sub = scope[(scope["label"] == lb) & not_suppressed]
        cnt = sub.groupby(["video_id", "local_idx"]).size()
        keys = set(cnt[cnt >= min_c].index)
        allowed = keys if allowed is None else (allowed & keys)
        if not allowed:
            return set()
    return allowed


def by_text(
    ocr_text: str | None = None,
    ocr_region: tuple[float, float, float, float] | None = None,
    ocr_region_iou_threshold: float = 0.05,
) -> set[tuple[str, int]] | None:
    """Lọc theo chữ trên màn hình (OCR) -> tập (video_id, local_idx) được phép.

    Filter CỨNG (AND với object/metadata) — khác secondary_entities (soft-boost) vì khớp
    chuỗi ký tự chính xác gần như không có false-positive, đáng tin hơn hẳn object detection
    (score mờ) hay CLIP (score clustering đã đo). Nếu user chắc chắn có chữ "X", ứng viên
    KHÔNG có chữ đó bị loại thẳng, không chỉ giảm điểm.

    ocr_region: (ymin,xmin,ymax,xmax) chuẩn hoá [0,1] — vùng user khoanh tay trên canvas
    (xem app.py, streamlit-drawable-canvas). ocr_region_iou_threshold THẤP (0.05, không phải
    0.5 như suppression) vì đây là ước lượng thô "khoảng đó" của tay người, không phải 2 bbox
    object thật chồng khít — ngưỡng cao sẽ loại oan kết quả đúng chỉ vì tay vẽ hơi lệch."""
    if not ocr_text and ocr_region is None:
        return None

    ocr_index = resources.get().ocr_index
    if ocr_index is None:
        raise RuntimeError("ocr_text.parquet chưa build — chạy build_ocr_index.py trước")

    df = ocr_index
    if ocr_text:
        df = df[df["text_norm"].apply(lambda t: _ordered_words_match(ocr_text, t))]
    if ocr_region is not None and len(df):
        boxes = df[["ymin", "xmin", "ymax", "xmax"]].to_numpy(dtype=float)
        iou = resources._box_iou_matrix(boxes, np.array([ocr_region], dtype=float))
        df = df[iou.flatten() > ocr_region_iou_threshold]

    allowed: set[tuple[str, int]] = set()
    for row in df.itertuples(index=False):
        for li in range(row.local_idx_start, row.local_idx_end + 1):
            allowed.add((row.video_id, li))
    return allowed


def apply(
    authors: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    keywords_any: list[str] | None = None,
    must_have_labels: list[str] | None = None,
    min_count: dict[str, int] | None = None,
    use_suppression: bool = True,
    include_open_vocab: bool = True,
    ocr_text: str | None = None,
    ocr_region: tuple[float, float, float, float] | None = None,
) -> set[tuple[str, int]] | None:
    """Gộp by_metadata() + by_objects() + by_text() thành 1 tập (video_id, local_idx) duy
    nhất — Tầng 2 chỉ cần đọc kết quả này, không cần biết filter tới từ nguồn nào."""
    video_allowed = by_metadata(authors, date_from, date_to, keywords_any)
    frame_allowed = by_objects(must_have_labels, min_count, use_suppression, include_open_vocab)
    text_allowed = by_text(ocr_text, ocr_region)

    frame_sets = [s for s in (frame_allowed, text_allowed) if s is not None]
    if video_allowed is None and not frame_sets:
        return None

    if frame_sets:
        combined = frame_sets[0]
        for s in frame_sets[1:]:
            combined = combined & s
        if video_allowed is None:
            return combined
        return {(v, l) for v, l in combined if v in video_allowed}

    row_pos = resources.get().row_pos
    assert video_allowed is not None
    return {(vid, li) for vid, li in row_pos if vid in video_allowed}
