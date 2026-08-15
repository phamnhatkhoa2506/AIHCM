"""Phân rã truy vấn phức hợp (nhiều ràng buộc: entity/số lượng/mô tả) thành:
  - entity + min_count -> giao TẦNG 1 (hard filter, đã có: must_have_labels/min_count)
  - phần mô tả còn lại (màu sắc, hành động, bối cảnh...) -> giao TẦNG 2 (CLIP xếp hạng)

Lý do cần: CLIP xử lý tốt câu mô tả 1 khía cạnh ("con chó", "cây nấm"), nhưng câu ghép nhiều
ràng buộc như ví dụ BTC thật —
  "Tìm video về một diễn giả mặc áo đỏ phát biểu tại một cuộc họp báo ngoài trời,
   phía sau có nhiều cây xanh."
— CLIP không đảm bảo ép được TỪNG ràng buộc cùng lúc, đặc biệt ràng buộc SỐ LƯỢNG ("nhiều cây
xanh") mà similarity toàn câu dễ bỏ qua. Tầng 1 (đếm object thật) ép được cái CLIP không ép được.

Dùng LLM TEXT-ONLY (không cần ảnh) — nhẹ, nhanh, dùng lại model đã dành riêng cho việc phân
rã/agentic từ trước (xem memory v1: agent_llm = meta/llama-3.1-8b-instruct, chọn từ bench
latency nhưng CHƯA từng có code nào gọi tới — đây là lần đầu dùng đúng mục đích đã định).
"""


from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "share"))  # module dung chung o goc v3/

import contextlib
import json
import os

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

import numpy as np

import resources
from label_translate import resolve as resolve_label_vi
from region_clip import encode_regions_batch
from search import search
from steplog import StepLog
from text_signals import ocr_asr_agreement
from tiers.tier1_filter import _strip_accents as _strip_accents_local
from tiers.tier2_vector import encode_query as encode_clip_text

load_dotenv()

PLANNER_MODEL = "meta/llama-3.1-8b-instruct"

_nim_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.environ["NVIDIA_NIM_API_KEY"])

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                    "min_count": {"type": "integer"},
                },
                "required": ["term", "min_count"],
            },
        },
        "secondary_entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                },
                "required": ["term"],
            },
        },
        "attributes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity_term": {"type": "string"},
                    "attribute_text": {"type": "string"},
                },
                "required": ["entity_term", "attribute_text"],
            },
        },
        "audio_mentions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"term": {"type": "string"}},
                "required": ["term"],
            },
        },
        "clip_text": {"type": "string"},
    },
    "required": ["entities", "secondary_entities", "attributes", "audio_mentions", "clip_text"],
}

SYSTEM_PROMPT = """You decompose a Vietnamese video-search query into structured constraints.

Extract "entities": concrete, countable OBJECTS/PEOPLE/ANIMALS ONLY when the query has an
EXPLICIT quantity signal for that noun — a number ("hai", "3", "hai mươi"...) or a word meaning
"several" ("nhiều", "vài", "một vài" -> use 2). A PLAIN mention with no quantity word (even with
the indefinite article "một" = "a/an", which is NOT a quantity emphasis) is NOT an entity — it
belongs only in "clip_text" (soft CLIP matching), never as a hard filter. This matters a lot:
a query naturally mentioning 3-4 objects with no counts (e.g. "một bát salad dưa leo trộn rau")
must NOT turn into 4 simultaneous hard-AND-filtered entities — real object detectors rarely
co-detect that many labels in one frame even when a human clearly sees them all, so this would
wrongly return zero/near-zero candidates. When genuinely unsure whether a quantity word is
present, leave the noun OUT of entities (soft CLIP handles it fine either way). Use the
Vietnamese noun itself as "term" (singular form, e.g. "cây", "người", "xe hơi") — do NOT
translate to English. Do NOT extract colors, actions, emotions, or scene descriptions as
entities — only physical countable objects. IMPORTANT: clothing/accessory nouns (áo, quần, mũ,
kính, giày, túi...) that merely describe what a person is WEARING are NOT separate entities —
they belong ONLY in "attributes" below, tied to the person. Never output an entity whose term is
just a color+noun phrase (e.g. "áo xanh") — that whole phrase is an attribute_text, never an
entity term.

IMPORTANT — Vietnamese classifier words (loại từ): words like "cái", "con", "chiếc", "quả",
"tấm", "cây" (when used as a classifier, not meaning "tree"), "củ" are GRAMMATICAL PARTICLES
placed before a noun (like "a/an/the" in English) — they are NEVER an entity by themselves.
"cái thìa" means "the spoon" — the entity term is "thìa" (or "cái thìa" together is fine too),
but NEVER extract bare "cái" alone as if it were an object. Similarly, NEVER extract a bare
VERB (cầm, giữ, cầm nắm, đứng, ngồi, nhìn...) as an entity or secondary_entity — verbs describe
actions, not objects, even when they appear right next to a noun (e.g. in "cầm thìa" the entity
is "thìa", not "cầm").

CRITICAL — Vietnamese compound nouns: Vietnamese writes each SYLLABLE separated by a space, so
a single word can look like multiple words. Many common body-part/object compounds have a FIRST
syllable that is ALSO a common standalone noun with a DIFFERENT meaning — you must read the
FULL compound, never just the first syllable. Examples: "bàn tay" (hand) is NOT "bàn" (table);
"đầu gối" (knee) is NOT "đầu" (head); "cổ tay" (wrist) is NOT "cổ" (neck); "mắt cá" (ankle) is
NOT "mắt" (eye) or "cá" (fish). Before extracting any 1-syllable entity, check the ORIGINAL
query text: if that syllable is immediately followed by another syllable forming a known
compound noun, extract (or ignore, per the rules above) the WHOLE compound — never just the
leading syllable.

Extract "secondary_entities": concrete objects mentioned in the query that are NOT the main
subject and are typically SMALL, easily occluded, or hard to detect reliably (hands, fingers,
ears, individual body parts, small held items, jewelry, etc.) — these matter for the query but
should NOT be a strict hard requirement, only a soft ranking boost when actually detected. Use
this category instead of "entities" for such objects — do NOT put them in both. Only "term" is
needed (no min_count — presence alone is what gets boosted, not exact count).

CRITICAL — do NOT hallucinate implied objects: only extract a secondary_entity if its noun is
LITERALLY WRITTEN in the query text. Do NOT infer unstated body parts from a verb — e.g. "cầm"
(hold) does NOT imply you should add "bàn tay" (hand) unless the query text actually contains
the word "tay"/"bàn tay". If in doubt whether a noun is actually present in the text, leave it
out entirely rather than guess.

Extract "attributes": visual attributes (color, clothing, pattern, accessory, material, pose)
tied to ONE specific concrete subject in the query — e.g. "áo đỏ" describing "diễn giả".
"entity_term" is the Vietnamese noun for that subject — it does NOT need to also appear in
"entities" above (a subject can have an attribute even when it has no quantity signal and so is
absent from "entities" — attribute matching resolves its own label independently, it is not a
hard filter either). "attribute_text" is a short Vietnamese phrase combining the entity + the
attribute (e.g. "người mặc áo đỏ") — this will be matched against a CROPPED region containing
ONLY that entity (nothing outside its bounding box is visible to this match), so it MUST be
self-contained and describe something that is physically part of / touching the entity itself.

CRITICAL — do NOT put SCENE/LOCATION/SURROUNDING context into attribute_text, only what is ON
or PART OF the entity: "ở sân bay" (at the airport), "trong công viên" (in the park), "phía sau
có cây xanh" (with trees behind), "trên đường phố đông người" (on a crowded street) describe the
BACKGROUND, which falls OUTSIDE a cropped box of just the entity — a Region-CLIP match against
"ô tô ở sân bay" run on a car crop can never see the airport, so it is comparing an image with
nothing airport-like against text describing an airport - meaningless/misleading score. Location
context belongs ONLY in "clip_text" (whole-frame match already covers it), never in
attribute_text. If a subject has no attribute confined to itself (only surrounding context),
omit it from "attributes" entirely rather than smuggling the location in.

Extract "audio_mentions": short topic phrases the query says are SPOKEN/SAID/HEARD in the
video — built around verbs like "nói"/"nhắc tới"/"đề cập"/"kể về"/"nói rằng" (say/mention/talk
about/tell about/say that) whose object is a TOPIC of speech, not a visible thing — e.g. "MC
nhắc tới Pháp" (the host mentions France) -> {"term": "Pháp"}; "người nói về giá xăng" (someone
talks about gas prices) -> {"term": "giá xăng"}. "term" is the SHORT Vietnamese phrase that was
said (a word/name/short phrase, NOT the whole clause) — this is matched against a speech-to-text
transcript, not the image, so it must be the literal topic word(s), not a paraphrase. Only
extract if the query explicitly frames it as spoken content (a verb of saying/mentioning) — do
NOT extract on-screen text/captions/signs here (that is a different, separate mechanism), and do
NOT invent a mention that is not stated. Leave empty if the query has no spoken-content clause.

"clip_text": the ORIGINAL query, unchanged — always keep it in full (colors, actions, scene,
location all matter for whole-frame semantic ranking too, even for entities/attributes
already extracted separately).

Reply with ONLY a JSON object, no explanation, no markdown fences.

Example query: "Tìm video về một diễn giả mặc áo đỏ phát biểu tại một cuộc họp báo ngoài trời, phía sau có nhiều cây xanh."
Example output: {"entities": [{"term": "cây", "min_count": 2}], "attributes": [{"entity_term": "diễn giả", "attribute_text": "người mặc áo đỏ"}], "clip_text": "Tìm video về một diễn giả mặc áo đỏ phát biểu tại một cuộc họp báo ngoài trời, phía sau có nhiều cây xanh."}
(Note: "diễn giả" is NOT in entities — "một diễn giả" has no quantity emphasis ("một" here is just
"a/an"), only "nhiều cây xanh" has a real quantity signal. "diễn giả" still gets its attribute via
attributes[].entity_term, resolved independently — it does not need to be in entities for that.)

Example query: "người đàn ông mặc áo xanh"
Example output: {"entities": [], "attributes": [{"entity_term": "người đàn ông", "attribute_text": "người mặc áo xanh"}], "clip_text": "người đàn ông mặc áo xanh"}
(No quantity word anywhere -> entities is empty; CLIP handles "người đàn ông" via clip_text, and
the color attribute still applies via Region-CLIP on "người đàn ông" independently.)

Example query: "một bát salad dưa leo trộn rau"
Example output: {"entities": [], "attributes": [], "clip_text": "một bát salad dưa leo trộn rau"}
(4 distinct nouns mentioned — bát/salad/dưa leo/rau — but ZERO quantity words, so NONE become
hard-filter entities. Turning all 4 into simultaneous hard-AND-filters would very likely return
zero candidates, since real detectors rarely co-detect that many labels in exactly one frame.
Let CLIP handle the whole sentence via clip_text instead — much safer than an empty result.)

Example query: "hai người mặc áo trắng và xám đang nhìn vào bàn tay ở giữa"
Example output: {"entities": [{"term": "người", "min_count": 2}], "secondary_entities": [{"term": "bàn tay"}], "attributes": [{"entity_term": "người", "attribute_text": "người mặc áo trắng và xám"}], "clip_text": "hai người mặc áo trắng và xám đang nhìn vào bàn tay ở giữa"}
(Note: "bàn tay" here is ONE compound noun meaning "hand" — do NOT extract "bàn" (table) as a
separate entity. It IS mentioned and matters (people are looking at it), but hands are small/
easily occluded so it goes in "secondary_entities" — a soft boost, not a strict hard filter.)

Example query: "Sau cảnh ba ô tô ở sân bay, có chữ NANTES và MC nhắc tới Pháp"
Example output: {"entities": [{"term": "ô tô", "min_count": 3}], "attributes": [], "audio_mentions": [{"term": "Pháp"}], "clip_text": "Sau cảnh ba ô tô ở sân bay, có chữ NANTES và MC nhắc tới Pháp"}
(attributes is EMPTY — "ở sân bay" is the SCENE surrounding the cars, not something on/part of a
car itself, so it must NOT become {"entity_term": "ô tô", "attribute_text": "ô tô ở sân bay"}
(that would compare a CROPPED car image, which never shows the airport, against airport text —
meaningless). The location still reaches the ranker via clip_text (whole-frame match) — it just
does not belong in attributes. "MC nhắc tới Pháp" IS a spoken-content clause ("nhắc tới" = mention)
-> audio_mentions=[{"term": "Pháp"}], matched against the ASR transcript, not the image. "NANTES"
is on-screen TEXT, not speech — it stays OUT of audio_mentions (handled separately by OCR).)"""


def plan_query(query: str) -> dict:
    resp = _nim_client.chat.completions.create(
        model=PLANNER_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        max_tokens=300,
        temperature=0.0,
    )
    content = resp.choices[0].message.content.strip()
    content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # phong than: khong phan ra duoc -> chi dung CLIP nhu cu
        return {"entities": [], "secondary_entities": [], "attributes": [], "audio_mentions": [], "clip_text": query}


# Nhan bo phan co the (OpenImages) - KHONG BAO GIO duoc lam hard-filter entity, du LLM co
# xep vao entities hay khong (xem chan trong planned_search ben duoi). Object nho, hay bi che
# khuat, detector kem tin cay (vd "Human hand" diem TB 0.51, chi ~8.8% frame - da do 2026-08-07).
_BODY_PART_LABELS = {
    "Human arm", "Human beard", "Human ear", "Human eye", "Human face", "Human foot",
    "Human hair", "Human hand", "Human head", "Human leg", "Human mouth", "Human nose",
}


def extract_entities(query: str, log: StepLog | None = None) -> dict:
    """LLM phan ra plan_query() -> resolve entities sang nhan OpenImages (voi chan cung bo
    phan co the - xem _BODY_PART_LABELS). Tra ve plan (dict) da them cac key
    resolved_must_have_labels/resolved_min_count/unresolved. TACH RIENG (2026-08-15, theo yeu
    cau nguoi dung) tu planned_search() de dung CHUNG duoc cho ca duong BTC cu (planned_search
    duoi day) LAN duong dense mac dinh (app.py goi thang ham nay, khong qua planned_search/
    search() BTC nua - xem tiers/dense_search.py)."""
    if log:
        with log.timed("LLM phân rã câu (NIM, meta/llama-3.1-8b-instruct)") as set_detail:
            plan = plan_query(query)
            set_detail(f"{len(plan.get('entities', []))} entity trích được")
    else:
        plan = plan_query(query)

    must_have_labels: list[str] = []
    min_count: dict[str, int] = {}
    unresolved: list[str] = []
    demoted_to_secondary: list[str] = []  # xem _BODY_PART_LABELS ben duoi

    _noop = lambda *_a, **_k: None  # noqa: E731
    with (log.timed("Resolve entity sang nhãn OpenImages") if log else contextlib.nullcontext(_noop)) as set_detail:
        for ent in plan.get("entities", []):
            term, cnt = ent["term"], ent.get("min_count", 1)
            hits = resolve_label_vi(term)
            if not hits:
                unresolved.append(term)
                continue
            # CHAN CUNG BANG CODE (2026-08-11, bug that phat hien qua benchmark 100 mau): LLM
            # van thinh thoang xep bo phan co the (tay/chan/mat...) vao entities (hard filter)
            # du SYSTEM_PROMPT da co quy tac cam - vd "tay" -> "Human hand" lam must_have_labels,
            # trong khi Human hand chi detect dung ~8.8% frame, score TB 0.51 (da do truoc do) ->
            # hard-filter mot nhan khong dang tin nay xoa sach ket qua oan. KHONG the tin
            # instruction-following cua LLM 100% (da xac nhan "whack-a-mole" nhieu lan) -> chan
            # o code, khong chi o prompt: nhan nao trong _BODY_PART_LABELS bi HA XUONG
            # secondary_entities (soft-boost) thay vi must_have_labels (hard filter).
            if any(label in _BODY_PART_LABELS for label in hits):
                demoted_to_secondary.append(term)
                continue
            for label in hits:
                must_have_labels.append(label)
                min_count[label] = max(min_count.get(label, 1), cnt)
        if log:
            set_detail(f"resolved={must_have_labels}, unresolved={unresolved}, "
                       f"ha xuong secondary (bo phan co the)={demoted_to_secondary}")

    plan["unresolved"] = unresolved
    plan["resolved_must_have_labels"] = must_have_labels
    plan["resolved_min_count"] = min_count
    if demoted_to_secondary:
        existing_secondary_terms = {s["term"] for s in plan.get("secondary_entities", [])}
        for term in demoted_to_secondary:
            if term not in existing_secondary_terms:
                plan.setdefault("secondary_entities", []).append({"term": term})
    return plan


def planned_search(
    query: str, top_k: int = 12, log: StepLog | None = None, use_region_clip: bool = True, **filters
) -> tuple[pd.DataFrame, dict]:
    """Phân rã query -> resolve entity sang nhãn tiếng Anh -> gọi search() với must_have_labels/
    min_count (Tầng 1) + clip_text (Tầng 2). Trả (kết quả, plan) — plan để hiển thị debug/soát.
    CHẠY TRÊN BỘ BTC CŨ (search() -> tier1_filter/tier2_vector) - xem extract_entities() ở trên
    để dùng LLM phân rã trên bộ dense (app.py gọi thẳng, không qua hàm này)."""
    plan = extract_entities(query, log)
    must_have_labels = plan["resolved_must_have_labels"]
    min_count = plan["resolved_min_count"]

    # gop voi filter nguoi dung tu truyen vao (vd tu app.py) - filter tu plan uu tien hon
    user_must_have = filters.pop("must_have_labels", None) or []
    merged_must_have = list({*user_must_have, *must_have_labels}) or None
    merged_min_count = {**(filters.pop("min_count", None) or {}), **min_count} or None

    attributes = plan.get("attributes", []) if use_region_clip else []
    secondary_entities = plan.get("secondary_entities", [])
    needs_wide_pool = bool(attributes) or bool(secondary_entities)
    rerank_pool = top_k * 4 if needs_wide_pool else top_k  # lay du rong hon de re-rank co gi ma chon

    results = search(
        plan.get("clip_text", query),
        top_k=rerank_pool,
        must_have_labels=merged_must_have,
        min_count=merged_min_count,
        log=log,
        **filters,
    )

    if attributes and not results.empty:
        # neu con secondary_entities can xu ly sau, GIU NGUYEN pool rong (khong cat top_k
        # ngay o day) - cat 1 LAN duy nhat o cuoi ham, sau khi ca 2 buoc boost da chay xong.
        keep_k = rerank_pool if secondary_entities else top_k
        results = _apply_region_clip_rerank(results, attributes, keep_k, log)
    elif log and plan.get("attributes") and not use_region_clip:
        log.add("Region-CLIP", "TẮT (toggle người dùng) — bỏ qua bước rerank theo thuộc tính, chỉ dùng Tầng 1+2", 0.0)

    if secondary_entities and not results.empty:
        results = _apply_secondary_entity_boost(results, secondary_entities, log)
        results = _apply_asr_boost(results, secondary_entities, log)

    if not results.empty:
        results = _apply_ocr_asr_agreement_boost(results, log)

    results = results.head(top_k).reset_index(drop=True)
    return results, plan


# Trong so cong diem cho secondary_entities (2026-08-07) - NHE HON REGION_CLIP_WEIGHT vi day
# chi la tin hieu PHU (object nho/de bi che khuat, detector khong tin cay - vd "Human hand"
# diem trung binh chi 0.51 tren corpus, xem hoi thoai). KHONG loai frame nao ca (khac han
# entities/must_have_labels o Tang 1) - chi cong diem neu THAY, khong tru diem neu KHONG thay
# (detector co the bo sot object nho du no thuc su co trong khung hinh).
SECONDARY_ENTITY_BOOST_WEIGHT = 0.15


def _apply_secondary_entity_boost(
    results: pd.DataFrame, secondary_entities: list[dict], log: StepLog | None
) -> pd.DataFrame:
    """Voi moi secondary entity (vd "bàn tay"): resolve sang nhan, tim diem detection CAO NHAT
    cua nhan do trong TUNG frame ung vien (0.0 neu khong co detection nao). CONFIDENCE-WEIGHTED
    (2026-08-07, sua sau khi ban ve nhieu) - KHONG dung co/khong nhi phan nua, vi detection sat
    nguong loc (~0.3, nhieu kha nang la nhieu/nham) truoc day duoc cong diem NGANG BANG detection
    tin cay cao (~0.9) - gio detection mo nhat chi duoc cong rat it, giam anh huong cua nhieu ma
    khong can loc bo cung nhac. Van KHONG loai frame nao ca (khac han entities o Tang 1)."""
    step_name = f"Secondary entities (soft boost, confidence-weighted) — {[e['term'] for e in secondary_entities]}"
    with (log.timed(step_name) if log else contextlib.nullcontext(lambda *_a, **_k: None)) as set_detail:
        objects_index = resources.get().objects_index
        if objects_index is None or results.empty:
            if log:
                set_detail("BỎ QUA — objects_index chưa có hoặc không còn kết quả")
            return results

        results = results.copy()
        boost = np.zeros(len(results))
        n_resolved = 0
        resolved_info = []
        for sec in secondary_entities:
            term = sec["term"]
            labels = resolve_label_vi(term)
            if not labels:
                resolved_info.append(f"{term}->KHÔNG resolve được")
                continue
            n_resolved += 1
            sub = objects_index[objects_index["label"].isin(labels)]
            # diem CAO NHAT theo (video_id, local_idx) - nhieu detection cung nhan trong 1
            # frame thi lay cai tin cay nhat, khong cong don.
            best_score = sub.groupby(["video_id", "local_idx"])["score"].max()
            match = np.array([
                best_score.get((r.video_id, int(r.local_idx)), 0.0) for r in results.itertuples(index=False)
            ], dtype=float)
            boost += match
            n_frames_hit = int((match > 0).sum())
            resolved_info.append(f"{term}->{labels}, {n_frames_hit}/{len(results)} frame có (điểm TB={match[match>0].mean():.2f})" if n_frames_hit else f"{term}->{labels}, 0/{len(results)} frame có")

        if n_resolved > 0:
            boost /= n_resolved
            results["secondary_boost"] = boost
            results["score_before_secondary_boost"] = results["score"]
            results["score"] = results["score"] + SECONDARY_ENTITY_BOOST_WEIGHT * boost
            results = results.sort_values("score", ascending=False).reset_index(drop=True)

        if log:
            set_detail("; ".join(resolved_info) if resolved_info else "không có entity phụ nào resolve được")
    return results


# Trong so cong diem cho ASR (2026-08-10) - NHE HON ca SECONDARY_ENTITY_BOOST_WEIGHT (0.15) vi
# loi noi KHONG dong bo chat voi hinh (phong vien co the noi ve X trong khi hinh dang chieu
# canh khac - xem hoi thoai) - la tin hieu YEU HON detection object trong khung hinh, khong
# nen dung hard-filter (khac OCR, chu tren man hinh THAT SU nam dung frame do).
ASR_BOOST_WEIGHT = 0.08


def _apply_asr_boost(
    results: pd.DataFrame, secondary_entities: list[dict], log: StepLog | None
) -> pd.DataFrame:
    """Dung LAI dung secondary_entities da trich (khong can UI nhap rieng cho ASR) - kiem tra
    xem tu do co duoc NOI RA gan frame ung vien khong (asr_index, xem build_asr_index.py).
    Nhi phan (co/khong, khac secondary_entity confidence-weighted) vi ASR khong tra confidence
    tin cay tin cay per-doan (xem asr_app.py). KHONG loai frame nao ca."""
    step_name = f"ASR (soft boost, lời nói) — {[e['term'] for e in secondary_entities]}"
    with (log.timed(step_name) if log else contextlib.nullcontext(lambda *_a, **_k: None)) as set_detail:
        asr_index = resources.get().asr_index
        if asr_index is None or results.empty:
            if log:
                set_detail("BỎ QUA — asr_text.parquet chưa có hoặc không còn kết quả")
            return results

        results = results.copy()
        boost = np.zeros(len(results))
        n_hit_total = 0
        for sec in secondary_entities:
            needle = f" {_strip_accents_local(sec['term'])} "
            hit = asr_index[asr_index["text_norm"].apply(lambda t: needle in f" {t} ")]
            if hit.empty:
                continue
            # danh dau (video_id, local_idx) nam TRONG khoang local_idx_start..end cua tung
            # doan transcript khop - dung set thay vi quet tung dong (it doan/video, khong can toi)
            hit_frames: set[tuple[str, int]] = set()
            for row in hit.itertuples(index=False):
                for li in range(row.local_idx_start, row.local_idx_end + 1):
                    hit_frames.add((row.video_id, li))
            match = np.array([
                1.0 if (r.video_id, int(r.local_idx)) in hit_frames else 0.0
                for r in results.itertuples(index=False)
            ])
            boost += match
            n_hit_total += int(match.sum())

        if n_hit_total > 0:
            boost = np.clip(boost, 0, 1)  # nhieu entity cung noi trung 1 frame khong duoc cong don qua 1.0
            results["asr_boost"] = boost
            results["score_before_asr_boost"] = results["score"]
            results["score"] = results["score"] + ASR_BOOST_WEIGHT * boost
            results = results.sort_values("score", ascending=False).reset_index(drop=True)

        if log:
            set_detail(f"{n_hit_total} frame có nhắc tới trong lời nói" if n_hit_total
                       else "không frame nào được nhắc tới trong lời nói")
    return results


# Trong so cong diem tin hieu cheo OCR-ASR (2026-08-11, thu nghiem theo yeu cau nguoi dung) -
# NHE HON ca ASR_BOOST_WEIGHT (0.08) vi day chi la tin hieu CUNG CO GIAN TIEP (chu tren man
# hinh + loi noi TRUNG token nhau quanh cung frame, KHONG lien quan truc tiep den tu khoa
# truy van) - dung de tang tin cay cho frame co CA 2 nguon deu "on ao" cung luc (thuong la tin
# tuc quan trong, so lieu, ten rieng duoc nhac + hien chu), khong phai tin hieu khop noi dung
# truy van. KHONG loai frame nao ca (giong het pattern secondary/asr boost).
OCR_ASR_AGREEMENT_WEIGHT = 0.05


def _apply_ocr_asr_agreement_boost(results: pd.DataFrame, log: StepLog | None) -> pd.DataFrame:
    """Cong diem THEM cho frame co ca OCR va ASR cung nhac toi noi dung TRUNG NHAU (xem
    text_signals.ocr_asr_agreement) - KHONG phu thuoc tu khoa truy van, chi la tin hieu do
    "do on ao" doc lap cua 2 nguon xung quanh frame co trung khop hay khong."""
    step_name = "OCR-ASR agreement (soft boost, tín hiệu chéo)"
    with (log.timed(step_name) if log else contextlib.nullcontext(lambda *_a, **_k: None)) as set_detail:
        res = resources.get()
        if res.ocr_index is None or res.asr_index is None:
            if log:
                set_detail("BỎ QUA — chưa có đủ ocr_index/asr_index")
            return results

        results = results.copy()
        agreement = np.array([
            ocr_asr_agreement(r.video_id, int(r.local_idx)) for r in results.itertuples(index=False)
        ])
        n_hit = int((agreement > 0).sum())
        if n_hit > 0:
            results["ocr_asr_agreement"] = agreement
            results["score_before_ocr_asr_boost"] = results["score"]
            results["score"] = results["score"] + OCR_ASR_AGREEMENT_WEIGHT * agreement
            results = results.sort_values("score", ascending=False).reset_index(drop=True)

        if log:
            set_detail(f"{n_hit}/{len(results)} frame có chữ+lời nói trùng nội dung" if n_hit
                       else "không frame nào có chữ+lời nói trùng nội dung")
    return results


# Trọng số cộng điểm region-CLIP vào score gốc khi re-rank — CHỈNH Ở ĐÂY khi tune (0 = tắt
# hẳn ảnh hưởng của region-CLIP, giữ nguyên thứ tự Tầng 2; càng cao càng ưu tiên khớp thuộc
# tính theo vùng hơn là độ khớp CLIP toàn khung hình).
REGION_CLIP_WEIGHT = 0.5

# Embedding region-CLIP đã precompute offline qua Modal GPU cho object Person/Animal
# (build_region_embeddings.py) — detection_id ở đây CHÍNH LÀ vị trí dòng gốc (RangeIndex)
# trong objects_index.parquet, do build_region_embeddings.py tạo bằng
# reset_index(drop=False) trên đúng file này, KHÔNG filter/sắp xếp lại trước đó. resources.py
# đọc objects_index bằng pd.read_parquet thô, không reset index -> objects_index.index (row
# label) trùng đúng detection_id, tra cứu trực tiếp không cần join thêm cột.
# Index (video_id, local_idx) -> list[(detection_id, label, ymin, xmin, ymax, xmax)] cua
# objects_index — build 1 LAN duy nhat (quet full 1.1 trieu dong 1 lan), tai su dung cho
# MOI query/attribute sau do. Truoc day moi (frame, attribute) filter boolean mask tren toan
# bo DataFrame (~O(1.1M) moi lan) - voi 48 frame x N attribute la 48*N lan quet toan bo, day
# la chi phi con lai chinh sau khi da wire embedding precomputed (xem hoi thoai truoc).
_OBJ_FRAME_INDEX_CACHE: dict[tuple[str, int], list[tuple]] | None = None


def _get_objects_by_frame_index(objects_index: pd.DataFrame) -> dict[tuple[str, int], list[tuple]]:
    global _OBJ_FRAME_INDEX_CACHE
    if _OBJ_FRAME_INDEX_CACHE is not None:
        return _OBJ_FRAME_INDEX_CACHE
    idx: dict[tuple[str, int], list[tuple]] = {}
    for detection_id, video_id, local_idx, label, ymin, xmin, ymax, xmax in zip(
        objects_index.index,
        objects_index.video_id,
        objects_index.local_idx,
        objects_index.label,
        objects_index.ymin,
        objects_index.xmin,
        objects_index.ymax,
        objects_index.xmax,
    ):
        idx.setdefault((video_id, int(local_idx)), []).append(
            (int(detection_id), label, ymin, xmin, ymax, xmax)
        )
    _OBJ_FRAME_INDEX_CACHE = idx
    return idx


_REGION_EMB_CACHE: tuple[dict[int, int], np.ndarray] | None = None


def _load_region_embeddings() -> tuple[dict[int, int], np.ndarray]:
    global _REGION_EMB_CACHE
    if _REGION_EMB_CACHE is not None:
        return _REGION_EMB_CACHE
    from config import INDEX_DIR

    emb_path = INDEX_DIR / "region_embeddings.npy"
    id_path = INDEX_DIR / "region_embeddings_detection_ids.npy"
    if emb_path.exists() and id_path.exists():
        vecs = np.load(emb_path)
        ids = np.load(id_path)
        id_to_row = {int(did): i for i, did in enumerate(ids)}
    else:
        vecs = np.zeros((0, 512), dtype=np.float32)
        id_to_row = {}
    _REGION_EMB_CACHE = (id_to_row, vecs)
    return _REGION_EMB_CACHE


def _apply_region_clip_rerank(
    results: pd.DataFrame, attributes: list[dict], top_k: int, log: StepLog | None
) -> pd.DataFrame:
    """Với mỗi thuộc tính (vd "người mặc áo đỏ" gắn với entity "diễn giả"): với mỗi frame ứng
    viên, tìm object khớp nhãn entity trong objects_index. Object nào ĐÃ có embedding precompute
    (Person/Animal, xem build_region_embeddings.py) thì tra thẳng theo detection_id — không
    crop+encode lại. Object thiếu (ngoài scope precompute) mới crop+encode on-the-fly như cũ,
    fallback. Lấy điểm CAO NHẤT trong các box của frame đó (frame có >=1 object khớp thuộc
    tính là đủ, không cần MỌI box khớp). Nhiều thuộc tính -> lấy TRUNG BÌNH các điểm max (muốn
    ép NGHIÊM cả 2 thuộc tính cùng lúc thì đổi sang min() — xem comment ngay dưới combine)."""
    objects_index = resources.get().objects_index
    per_attr_scores: list[list[float]] = []

    for attr in attributes:
        entity_term, attribute_text = attr["entity_term"], attr["attribute_text"]
        step_name = f"Region-CLIP — '{attribute_text}' (entity: {entity_term})"
        with (log.timed(step_name) if log else contextlib.nullcontext(lambda *_a, **_k: None)) as set_detail:
            entity_labels = resolve_label_vi(entity_term)
            if not entity_labels or objects_index is None:
                scores = [0.0] * len(results)
                if log:
                    set_detail(f"BỎ QUA — entity '{entity_term}' không resolve được nhãn nào")
                per_attr_scores.append(scores)
                continue

            text_vec = encode_clip_text(attribute_text)
            id_to_row, precomputed_vecs = _load_region_embeddings()
            frame_index = _get_objects_by_frame_index(objects_index)
            entity_labels_set = set(entity_labels)

            # buoc 1: voi moi frame, tach object khop nhan thanh 2 nhom - da co embedding san
            # (tra thang) vs chua co (gom vao boxes_flat de crop+encode fallback 1 lan batch).
            # Tra cuu qua dict (O(1) theo key (video_id, local_idx)) thay vi filter boolean
            # mask tren toan bo objects_index moi frame.
            per_frame_precomputed: list[list[np.ndarray]] = []
            per_frame_missing_idx: list[list[int]] = []
            boxes_flat: list[tuple[str, int, tuple]] = []
            for row in results.itertuples(index=False):
                objs = frame_index.get((row.video_id, int(row.local_idx)), [])
                frame_precomputed = []
                frame_missing = []
                for detection_id, label, ymin, xmin, ymax, xmax in objs:
                    if label not in entity_labels_set:
                        continue
                    if detection_id in id_to_row:
                        frame_precomputed.append(precomputed_vecs[id_to_row[detection_id]])
                    else:
                        box = (ymin, xmin, ymax, xmax)
                        frame_missing.append(len(boxes_flat))
                        boxes_flat.append((row.video_id, int(row.local_idx), box))
                per_frame_precomputed.append(frame_precomputed)
                per_frame_missing_idx.append(frame_missing)

            n_precomputed = sum(len(v) for v in per_frame_precomputed)

            # buoc 2: 1 lan goi batch CHI cho box con thieu (fallback, thuong rat it/khong co
            # neu entity la Person/Animal - da precompute het 449,356 object roi).
            fresh_vecs = encode_regions_batch(boxes_flat) if boxes_flat else np.zeros((0, 512))

            # buoc 3: gop lai theo frame (precomputed + fresh) - max trong cac box cua frame do
            scores = []
            n_frames_with_match = 0
            for frame_precomputed, missing_idx in zip(per_frame_precomputed, per_frame_missing_idx):
                all_vecs = list(frame_precomputed) + [fresh_vecs[i] for i in missing_idx]
                if not all_vecs:
                    scores.append(0.0)
                    continue
                n_frames_with_match += 1
                sims = np.array(all_vecs) @ text_vec.reshape(-1)
                scores.append(float(sims.max()))

            per_attr_scores.append(scores)
            if log:
                set_detail(
                    f"nhãn={entity_labels}, {n_frames_with_match}/{len(results)} frame có object khớp, "
                    f"{n_precomputed} box dùng embedding precomputed, {len(boxes_flat)} box phải crop+encode mới (fallback)"
                )

    # combine nhieu thuoc tinh: trung binh cac max-score (doi thanh np.min(..., axis=0) neu
    # muon ep NGHIEM ca N thuoc tinh cung luc thay vi "cang nhieu cai dung cang tot")
    region_score = np.mean(per_attr_scores, axis=0) if per_attr_scores else np.zeros(len(results))
    results = results.copy()
    results["region_score"] = region_score
    results["score_before_rerank"] = results["score"]
    results["score"] = results["score"] + REGION_CLIP_WEIGHT * results["region_score"]
    results = results.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
    return results


if __name__ == "__main__":
    import sys

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    q = "Tìm video về một diễn giả mặc áo đỏ phát biểu tại một cuộc họp báo ngoài trời, phía sau có nhiều cây xanh."
    plan = plan_query(q)
    print("Plan:", json.dumps(plan, ensure_ascii=False, indent=2))

    results, full_plan = planned_search(q, top_k=8)
    print("\nResolved:", full_plan["resolved_must_have_labels"], full_plan["resolved_min_count"])
    if full_plan["unresolved"]:
        print("Khong resolve duoc:", full_plan["unresolved"])
    print(results[["video_id", "frame_idx", "score"]].to_string(index=False))
