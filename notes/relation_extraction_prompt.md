# Prompt trích xuất relation cho 1 keyframe (Scene Graph — semantic edge)

> Mục đích: sinh **cạnh ngữ nghĩa** (holding / riding / wearing...) cho scene graph phía dữ liệu.
> Đây là khâu YẾU NHẤT của hướng graph (SGG khó) → prompt phải ưu tiên PRECISION hơn recall:
> thà thiếu cạnh còn hơn cạnh sai. Cạnh không gian (left_of...) đã precompute bằng hình học box
> lúc offline — prompt này TẬP TRUNG vào interaction relation mà hình học không suy ra được.
>
> I/O tiếng Anh (đã kiểm chứng với NVIDIA NIM: ổn định hơn tiếng Việt); map lại sang tiếng Việt ở
> tầng query. Object + box được CẤP SẴN (từ Objects BTC), VLM KHÔNG tự detect lại → chống bịa object.

---

## SYSTEM PROMPT

```
You are a visual relation extractor for a video-retrieval knowledge graph.
You are given ONE image and a fixed list of already-detected objects (with IDs
and bounding boxes). Your ONLY job: report interaction/containment relations
that are DIRECTLY VISIBLE between objects in that list.

Never invent objects. Never guess. When unsure, omit the pair.
Precision matters far more than completeness: a missing relation is fine, a
wrong relation is a failure.
```

## USER PROMPT (template — điền `{{...}}` khi chạy)

```
DETECTED OBJECTS (use ONLY these IDs; do not add new objects):
{{object_list}}
# ví dụ:
#   o1: person       box=[x,y,w,h]  conf=0.88
#   o2: person       box=[x,y,w,h]  conf=0.83
#   o3: motorcycle   box=[x,y,w,h]  conf=0.71
#   o4: bag          box=[x,y,w,h]  conf=0.55

ALLOWED RELATIONS (choose ONLY from this list; otherwise omit the pair):
  interaction: holding, riding, wearing, carrying, sitting_on, standing_on,
               pushing, pulling, touching, looking_at, talking_to, feeding
  containment: inside, on_top_of, holding_up, part_of

RULES
1. Report a relation ONLY if clearly visible in the image. Unsure -> omit.
2. Use exact object IDs from the list. Relations are directional:
   subj -> rel -> obj  (e.g. person holding bag, NOT bag holding person).
3. For each relation give:
     conf     = 0.0-1.0  (your visual certainty)
     evidence = 3-6 words describing what you SEE that proves it
4. Do NOT output spatial relations (left_of/above/...) — those are computed
   separately from boxes.
5. If NO clear interaction exists, return {"relations": []}.
6. Output ONLY valid JSON. No prose, no markdown fences.

OUTPUT FORMAT (exactly this shape):
{"relations":[
  {"subj":"o1","rel":"riding","obj":"o3","conf":0.90,"evidence":"person seated astride motorcycle"},
  {"subj":"o2","rel":"carrying","obj":"o4","conf":0.70,"evidence":"bag hanging on shoulder"}
]}
```

---

## Các trục để tinh chỉnh (điều chỉnh rồi đo)

| Trục | Lựa chọn | Đánh đổi |
|---|---|---|
| **Tập relation** | rộng (nhiều động từ) ↔ hẹp (~10 phổ biến) | rộng = phủ nhiều nhưng dễ bịa; hẹp = chắc nhưng bỏ sót |
| **Ngưỡng conf giữ cạnh** | 0.5 / 0.6 / 0.7 | cao = graph sạch, ít cạnh; thấp = nhiều cạnh, lẫn rác |
| **Có gửi kèm box toạ độ không** | có ↔ chỉ tên object | có box giúp phân biệt object trùng loại (2 person), nhưng token nhiều hơn |
| **evidence bắt buộc hay optional** | bắt buộc | bắt buộc = grounding tốt hơn, nhưng tốn token/latency |
| **Số cặp tối đa/frame** | giới hạn (vd top-8) ↔ không | giới hạn = rẻ, tránh liệt kê tràn lan cặp vô nghĩa |

## Cơ chế tự-kiểm (gần miễn phí) — cross-check với spatial precomputed

Vì cạnh không gian đã tính sẵn bằng hình học box lúc offline, dùng nó bắt VLM "nói dối":

- VLM chỉ trả **interaction** (rule #4 đã cấm spatial) — nhưng nhiều interaction hàm ý vị trí:
  `riding` → subj phải chồng/trên box obj; `carrying` → 2 box gần/chồng nhau.
- Nếu VLM nói `o1 riding o3` mà box o1 và o3 **không chồng lấn chút nào** → cạnh đó gần như chắc sai
  → hạ conf hoặc loại. Đây là confidence-gate hình học, không tốn thêm 1 lần gọi model nào.

## Ví dụ few-shot (tuỳ chọn thêm vào để ổn định output)

```
# INPUT objects: o1 person, o2 dog, o3 leash
# GOOD output:
{"relations":[{"subj":"o1","rel":"holding","obj":"o3","conf":0.85,"evidence":"hand grips leash"},
              {"subj":"o3","rel":"part_of","obj":"o2","conf":0.6,"evidence":"leash attached to dog collar"}]}
# BAD output (những lỗi cần tránh):
#   - {"subj":"o1","rel":"walking","obj":"o2"}   ← "walking" KHÔNG trong tập cho phép
#   - {"subj":"o4",...}                          ← o4 không có trong danh sách object
#   - {"subj":"o2","rel":"holding","obj":"o3"}   ← sai hướng (chó không cầm dây)
```

---

## Ghi chú trạng thái (2026-07-23)

- Đây là bản để **tinh chỉnh trước**, chưa gọi model thật (chưa có keyframe domain AIC 2026;
  máy hiện không có GPU/ffmpeg sẵn — xem lịch sử thảo luận).
- Khi test: dùng đúng vision model đang cấu hình trong `v1` (`SETTINGS.llm.model_name`,
  hiện `meta/llama-3.2-11b-vision-instruct` trên NVIDIA NIM) — KHÔNG dùng `agent_llm` (text-only).
- Đo bằng: precision cạnh (bao nhiêu cạnh model trả về là ĐÚNG khi người nhìn ảnh kiểm), không chỉ
  recall — vì semantic edge ưu tiên precision.
