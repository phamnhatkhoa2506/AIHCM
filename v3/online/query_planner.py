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

import openai
from dotenv import load_dotenv
from openai import OpenAI

from app_flags import NIM_TIMEOUT_SECONDS, NIMTimeoutError
from label_translate import resolve as resolve_label_vi
from steplog import StepLog

load_dotenv()

PLANNER_MODEL = "meta/llama-3.1-8b-instruct"

# timeout=NIM_TIMEOUT_SECONDS (2026-08-20, theo yeu cau nguoi dung - xem app_flags.py): tranh
# treo VO HAN khi NIM khong phan hoi (khong dat truoc day, xem BUG THAT trong query_distill.py).
_nim_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_NIM_API_KEY"],
    timeout=NIM_TIMEOUT_SECONDS,
)

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
    try:
        resp = _nim_client.chat.completions.create(
            model=PLANNER_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            max_tokens=300,
            temperature=0.0,
        )
    except openai.APITimeoutError as e:
        # 2026-08-20 (theo yeu cau nguoi dung) - RAISE ro rang thay vi de treo vo han/loi tho -
        # xem docstring NIMTimeoutError (app_flags.py) va query_distill.py cho boi canh day du.
        raise NIMTimeoutError("phân rã câu (plan_query/extract_entities)") from e
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

    full_plan = extract_entities(q)
    print("\nResolved:", full_plan["resolved_must_have_labels"], full_plan["resolved_min_count"])
    if full_plan["unresolved"]:
        print("Khong resolve duoc:", full_plan["unresolved"])
