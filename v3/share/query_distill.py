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
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

from label_translate import find_known_terms

load_dotenv()

DISTILL_MODEL = "meta/llama-3.1-8b-instruct"  # dung lai dung model da chon cho query_planner.py

# 2026-08-18 (theo yeu cau nguoi dung: "mình muốn thử với nhiều LLM hơn... model từ bé đến
# lớn") - registry model CO THE CHON cho buoc chung cat, GIONG HET pattern VLM_OCR_MODELS
# (submission_pipeline.py)/OCR_MATCH_ALGORITHMS (tier1_filter.py). Test truc tiep tren case
# that (cau "nhiều người ngồi trên xe máy..." bi LLM 8B XOA MAT "nhiều người", chi con "xe máy
# đậu" - vi pham ro rang rule "Keep every CONCRETE visual element: people..." cua chinh
# SYSTEM_PROMPT) qua CAC MODEL NIM co san (client.models.list(), loc theo tu khoa llama/qwen/
# mistral/gemma/nemotron...):
#   - meta/llama-3.1-8b-instruct (MAC DINH, khong doi hanh vi cu): NHO NHAT, XOA MAT "nhiều
#     người" tren case that.
#   - meta/llama-3.1-70b-instruct: GIU DUNG "many people sit on motorcycles" tren case that.
#   - meta/llama-3.3-70b-instruct: CUNG giu dung, ban moi hon 70B.
# Model KHONG dung duoc qua endpoint completions don gian nay (loi AttributeError vi tra ve
# content=None - co the do dinh dang "reasoning" rieng, chua ho tro): nvidia/llama-3.3-
# nemotron-super-49b-v1.5, openai/gpt-oss-120b - DA LOAI khoi danh sach.
DISTILL_MODELS: dict[str, str] = {
    "meta/llama-3.1-8b-instruct": "LLaMA 3.1 8B (mặc định — nhỏ nhất, nhanh nhất)",
    "meta/llama-3.1-70b-instruct": "LLaMA 3.1 70B (giữ đúng chi tiết hơn khi test)",
    "meta/llama-3.3-70b-instruct": "LLaMA 3.3 70B (bản mới nhất, giữ đúng chi tiết hơn)",
}
DEFAULT_DISTILL_MODEL = DISTILL_MODEL

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
- Output declarative English sentence(s), present tense, like a photo caption - not a question,
  no quotes, no explanation. Length is NOT fixed at ~20 words - it should match how much the
  ORIGINAL query actually describes: a short/simple query stays short (roughly 6-20 words); a
  query that genuinely describes MULTIPLE distinct visual facts (e.g. explicitly separates
  "tiền cảnh" (foreground) vs "nền"/"phía sau" (background), or lists several independent
  objects/regions joined by "và") should keep ALL of them, even if that runs to ~40-50 words
  across one or two sentences. The real model limit is ~64 tokens (roughly 45-55 English
  words) - do NOT self-impose a stricter cap than that.
- If the query already IS short and visual, translate it directly with minimal change.

CRITICAL - DO NOT DROP A WHOLE CLAUSE JUST TO STAY SHORT: when a query names 2+ genuinely
different visual elements/regions of the same frame (a foreground subject AND a separately
described background/sky/composition detail, or several distinct objects), compressing means
tightening the WORDING of each clause (cut redundant adjectives, merge repeated ideas) - it does
NOT mean deleting an entire clause because the sentence is getting long. A frame with 2 distinct
described regions needs BOTH regions represented in the caption, or the embedding will only ever
match half the scene. Only drop a clause if it is pure narrative/emotional filler (see STRIP
rule above) or falls under one of the DROP rules below (spoken-audio-only, on-screen text) - a
clause that describes something physically visible in the frame is NEVER filler, no matter how
long the sentence already is.

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

KEEP any mention of SPECIFIC on-screen TEXT/writing/captions/signs/license plates/subtitles the
query says appear in the frame (e.g. "có chữ NANTES", "biển số 79H-6072", "dòng chữ ghi rằng...")
- include the literal text/word/number in the caption (e.g. "a sign reads NANTES", "a license
plate showing 79H-6072"), do not paraphrase it away.

SỬA 2026-08-17 (theo phát hiện thật của người dùng, ngược lại quy tắc DROP cũ): trước đây rule
này XOÁ chữ khỏi caption với lý do "CLIP-family embedding models đọc chữ yếu, không đáng tin -
giữ lại chỉ tạo FALSE PRECISION". Nhưng trên thực tế hệ thống HIỆN TẠI KHÔNG có cơ chế nào khác
biến chữ được nhắc trong 1 câu query TỰ DO (không vẽ khung canvas) thành hard-filter OCR thật -
nghĩa là XOÁ chữ khỏi caption đồng nghĩa với MẤT TRẮNG hoàn toàn tín hiệu đó, không có gì thay
thế. Người dùng đã kiểm chứng qua so sánh thật: GIỮ chữ lại trong caption (dù CLIP chỉ đọc được
1 phần, không chính xác tuyệt đối) vẫn cho kết quả TỐT HƠN rõ rệt so với xoá hẳn - tín hiệu mờ
còn hơn không có gì. Nếu sau này hệ thống có thêm cơ chế tự trích OCR từ câu tự do thành
hard-filter thật, có thể cân nhắc lại rule này.

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
  "Three cars are parked at an airport, a sign reads NANTES." (KEEP "có chữ NANTES" - include
  the literal word, it is a genuine (if weak) visual signal with no other hard-filter available
  for free-text queries - DROP ONLY "MC nhắc tới Pháp" - spoken audio content, not visible, has
  no visual correlate at all.)
- query "Một chiếc xuồng gỗ có mui che màu trắng neo sát bờ một cánh đồng ngập nước phủ đầy lục
  bình. Tiền cảnh là tấm bê tông vuông nhô ra mặt nước, nền trời nhiều mây trắng và hàng cây
  xanh trải dài phía xa." -> "A wooden boat with a white canopy is moored by the edge of a
  flooded field covered with water hyacinth. In the foreground, a square concrete slab juts
  out over the water; the sky is full of white clouds and a row of green trees stretches into
  the distance." (COMPOUND SCENE - the original explicitly separates a foreground boat/field
  from a SECOND foreground element (concrete slab) plus background sky/trees - keep BOTH
  sentences, do NOT drop the second one just to shorten. Note "lục bình" = water hyacinth, a
  floating aquatic weed - NOT "lotus", a different plant entirely; do not swap them.)"""


@lru_cache(maxsize=2048)
def distill_query(query_vi: str, model: str = DEFAULT_DISTILL_MODEL) -> str:
    """Tra ve 1 caption tieng Anh ngan. Phong than (2026-08-15, giong pattern plan_query() o
    query_planner.py): loi API/parse -> tra nguyen query GOC (khong lam sap he thong, chi mat
    di loi ich chung cat/dich - encode van chay duoc, ket qua co the kem hon).

    Ghep them "Known glossary" (2026-08-15, sua bug hallucination that: "con lân" -> LLM tu
    bia "A white ox is standing in a green field." thay vi dich dung nghia van hoa) - tu
    label_translate.find_known_terms(), CHINH la tu dien 514 nhan + SYNONYMS da co san (dung
    chung voi Object filter), ep LLM dung dung thuat ngu da xac minh thay vi doan.

    model: key trong DISTILL_MODELS (2026-08-18, theo yeu cau nguoi dung "mình muốn thử với
    nhiều LLM hơn... model từ bé đến lớn") - mac dinh DEFAULT_DISTILL_MODEL (model 8B cu, KHONG
    doi hanh vi truoc do). Cache theo CA (query_vi, model) - doi model se tinh lai (dung, vi ket
    qua co the khac that theo model, xem DISTILL_MODELS docstring).

    @lru_cache (2026-08-17, theo phat hien that cua nguoi dung: "temperature=0 nhung cau
    distill van doi lien tuc") - day la han che DA BIET cua hau het API LLM hosted (bao gom
    NIM): temperature=0 chi dam bao "luon chon token xac suat cao nhat", nhung logits van co
    the lech NHE giua cac lan goi do batching/kernel KHONG hoan toan tat dinh phia server -
    1 token lech som la ca cau phan ky han. Khong sua duoc tu phia client (khong phai bug o
    code nay). Cache theo CHINH XAC query_vi (khong chuan hoa/lowercase - 2 cau khac nhau ve
    hinh thuc van nen duoc chung cat rieng) dam bao CUNG 1 cau hoi trong SUOT PHIEN chay luon
    ra CUNG 1 ket qua - loai bo hoan toan nhieu do bat dinh LLM khi test lai/so sanh model,
    dong thoi tiet kiem 1 lan goi API neu nguoi dung go lai dung cau cu."""
    query_vi = query_vi.strip()
    if not query_vi:
        return query_vi
    if model not in DISTILL_MODELS:
        model = DEFAULT_DISTILL_MODEL

    user_content = query_vi
    glossary = find_known_terms(query_vi)
    if glossary:
        glossary_str = ", ".join(f'"{vi}" -> "{en}"' for vi, en in glossary.items())
        user_content += f"\n\nKnown glossary: {glossary_str}"

    try:
        resp = _client().chat.completions.create(
            model=model,
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
