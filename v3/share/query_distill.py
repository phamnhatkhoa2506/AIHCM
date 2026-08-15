"""Chung cat 1 cau query tieng Viet (co the dai/van ve, nhieu tu ngu mo ta cam xuc/boi canh
rom ra) thanh 1 caption tieng Anh NGAN, thuan mo ta THI GIAC - dung lam input cho 3 model
embedding CLIP-family cua bo dense (SigLIP2/PE-Core/BEiT-3, xem tiers/dense_search.py).

Ly do can buoc nay (2026-08-15, theo yeu cau nguoi dung):
  - PE-Core/BEiT-3: BAT BUOC tieng Anh (tokenizer/finetune checkpoint chi tieng Anh, dua
    tieng Viet vao la OOV, embedding vo nghia).
  - SigLIP2: ve ly thuyet nhan da ngon ngu (WebLI), nhung tieng Anh van chiem da so trong tap
    train - CHUA co benchmark rieng de khang dinh tieng Viet thang, nen DUNG CHUNG 1 duong
    tieng Anh cho ca 3 model thay vi tach rieng (don gian hoa, de maintain/so sanh).
  - Cau goc (kieu de thi AIC, van hoc/mo ta dai dong) thuong nhieu tu "van ve" (cam xuc, boi
    canh) LAM LOANG tin hieu thi giac cot loi - CLIP-family embedding hoat dong tot nhat voi
    cau NGAN, cu the, thuan mo ta canh/vat/hanh dong (giong prompt engineering CLIP thong
    thuong), khong phai van xuoi dai.

Goi 1 LAN duy nhat/query (KHONG goi rieng cho tung model) - ket qua dung CHUNG cho ca
siglip/pe_core/beit3/rrf (xem dense_search.py::_encode_query goi ham nay truoc khi ma hoa).
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

from label_translate import find_known_terms

load_dotenv()

DISTILL_MODEL = "meta/llama-3.1-8b-instruct"  # dung lai dung model da chon cho query_planner.py

_nim_client: OpenAI | None = None


def _client() -> OpenAI:
    global _nim_client
    if _nim_client is None:
        _nim_client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=os.environ["NVIDIA_NIM_API_KEY"])
    return _nim_client


SYSTEM_PROMPT = """You translate a Vietnamese video-search query into a SHORT English caption
for a CLIP-family image-text embedding model (SigLIP2/PE-Core/BEiT-3).

Rules:
- Keep every CONCRETE visual element: people, objects, colors, clothing, actions, setting,
  spatial relations, counts. Do NOT drop any object/attribute that is actually visible in a
  frame - only drop what is NOT visual.
- STRIP flowery/literary/narrative filler that does not describe what the frame LOOKS like:
  emotional commentary, rhetorical framing, meta-phrases like "video about", "a scene showing",
  "hãy tìm cho tôi", "đoạn video nói về", excessive subordinate clauses restating the same idea.
- Output ONE short declarative English sentence (roughly 6-20 words), present tense, like a
  photo caption - not a paragraph, not a question, no quotes, no explanation.
- If the query already IS short and visual, translate it directly with minimal change.

CRITICAL - NEVER HALLUCINATE: do NOT invent any object, setting, background, action, or
attribute that is not stated (or directly implied) in the original Vietnamese text. A short or
ambiguous query must stay short/ambiguous in English - do NOT pad it with an invented scene to
make it "sound like a real photo caption". E.g. "con lân" (3 words) must translate to a short
phrase about the actual subject, NOT expand into a fabricated scene with a setting that was
never mentioned.

If a "Known glossary" list is provided below the query, it gives VERIFIED Vietnamese->English
mappings for SINGLE words/short phrases (from this project's own label dictionary) - if a query
word/phrase appears in the glossary, prefer that exact English term instead of guessing a
literal/generic translation (these terms are often ambiguous or culturally specific and a
generic translation is usually WRONG). BUT: the glossary is looked up per-word and does NOT
know Vietnamese compound nouns - if a glossary word is actually the FIRST half of a longer
compound noun in the original text (e.g. "bàn" alone means "table", but inside "bàn tay" it
means "hand" - similarly "đầu"/"đầu gối"=knee not head, "cổ"/"cổ tay"=wrist not neck,
"mắt"/"mắt cá"=ankle not eye), read the ORIGINAL Vietnamese text yourself and use the compound's
real meaning, ignoring the glossary's single-word entry for that case.

Do NOT invent a subject, gender, pose, or action for a BARE noun/short phrase that names only
an object with no stated subject - e.g. "áo" (a shirt/clothing item, no wearer mentioned) must
stay about the clothing item itself, NOT become "a woman wearing a dress" (that invents a
person, a gender, and a specific garment that were never stated).

DROP anything that describes ONLY WHAT IS SPOKEN/SAID/heard in the video, not what the frame
LOOKS LIKE - e.g. "MC nhắc tới Pháp" (the host mentions France), "người nói về giá cả" (someone
talks about prices), "lời bài hát nói rằng..." (the lyrics say...), any clause built around
verbs like "nói"/"nhắc tới"/"đề cập"/"kể về" (say/mention/talk about/tell about) whose object is
a TOPIC of speech, not a visible thing. A CLIP-family image embedding model cannot see or hear
audio content - keeping such a clause in the caption only adds noise a vision model cannot use,
sometimes actively misleading it (e.g. "mentioning France" has no visual correlate at all -
France is not a color/object/scene). Keep only the genuinely visual parts of the same sentence.

DROP any mention of SPECIFIC on-screen TEXT/writing/captions/signs/license plates/subtitles the
query says appear in the frame (e.g. "có chữ NANTES", "biển số 79H-6072", "dòng chữ ghi rằng...")
- do NOT translate the literal text/word/number into the caption at all, not even paraphrased.
Reason: CLIP-family embedding models only have weak, unreliable text recognition - they cannot
distinguish one specific word/string from another, so including it just biases matching toward
"any frame with visible text/signage" in general, not the exact string - this is FALSE PRECISION
(looks like an exact filter but is not one). Exact on-screen text is handled by a SEPARATE,
reliable OCR exact-string hard-filter elsewhere in the system - it does not need to appear in
this caption at all. If the sentence has other visual content besides the text mention, keep
that; drop only the text/writing part.

Reply with ONLY the English caption, nothing else - no markdown, no quotes, no prefix.

Examples:
- query "con lân" with glossary {"con lân": "lion dance costume"} -> "A lion dance costume."
  (NOT "A white ox standing in a green field." - fabricates a species and a scene never mentioned.)
- query "áo" (no glossary hit) -> "A shirt." (NOT "A woman wearing a dress." - invents a person/
  gender/garment that were never stated.)
- query "ba người đang đứng gần bàn tay" with glossary {"tay": "Human hand", "bàn": "Table"} ->
  "Three people are standing near a hand." (the original text has the COMPOUND "bàn tay" = hand,
  not the standalone word "bàn" = table - the glossary's separate "bàn" entry does NOT apply here.)
- query "Sau cảnh ba ô tô ở sân bay, có chữ NANTES và MC nhắc tới Pháp" ->
  "Three cars are parked at an airport." (DROP BOTH "có chữ NANTES" - the literal word must not
  appear, CLIP cannot distinguish it from any other on-screen text, exact OCR handles this
  separately - AND "MC nhắc tới Pháp" - spoken audio content, not visible. Only "three cars at
  an airport" is genuinely visual and specific enough to keep.)"""


def distill_query(query_vi: str) -> str:
    """Tra ve 1 caption tieng Anh ngan. Phong than (2026-08-15, giong pattern plan_query() o
    query_planner.py): loi API/parse -> tra nguyen query GOC (khong lam sap he thong, chi mat
    di loi ich chung cat/dich - encode van chay duoc, ket qua co the kem hon).

    Ghep them "Known glossary" (2026-08-15, sua bug hallucination that: "con lân" -> LLM tu
    bia "A white ox is standing in a green field." thay vi dich dung nghia van hoa) - tu
    label_translate.find_known_terms(), CHINH la tu dien 514 nhan + SYNONYMS da co san (dung
    chung voi Object filter), ep LLM dung dung thuat ngu da xac minh thay vi doan."""
    query_vi = query_vi.strip()
    if not query_vi:
        return query_vi

    user_content = query_vi
    glossary = find_known_terms(query_vi)
    if glossary:
        glossary_str = ", ".join(f'"{vi}" -> "{en}"' for vi, en in glossary.items())
        user_content += f"\n\nKnown glossary: {glossary_str}"

    try:
        resp = _client().chat.completions.create(
            model=DISTILL_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=80,
            temperature=0.0,
        )
        content = resp.choices[0].message.content.strip()
        content = content.removeprefix("```").removesuffix("```").strip().strip('"')
        return content or query_vi
    except Exception:
        return query_vi


if __name__ == "__main__":
    import sys

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    q = (sys.argv[1] if len(sys.argv) > 1 else
         "Hãy tìm giúp tôi đoạn video ghi lại khoảnh khắc xúc động khi một diễn giả, "
         "trong bộ áo dài đỏ rực rỡ, đang say sưa phát biểu trước đám đông tại một cuộc "
         "họp báo ngoài trời, phía sau là hàng cây xanh mát rượi.")
    print(f"Goc: {q}")
    print(f"Chung cat: {distill_query(q)}")
