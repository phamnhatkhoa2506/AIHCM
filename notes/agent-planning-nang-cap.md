# Nâng cấp tầng Planning của Agent — từ "tra bảng rule" lên "lập kế hoạch có kiểm chứng"

> Nâng cấp `kien-truc-2-tang-agent-va-ui.md` (bản cũ: agent tra bảng rule chọn 1 Action kế tiếp).
> Tích hợp bài học từ SnapMind (LLM Planner + registry + autonomy levels), VIREO (precompute + preview),
> Fusionista 2.0 (interactive confirmation rerank), Exquisitor (rank aggregation).
> Ngày: 2026-07-09.

---

## 0. Vấn đề của thiết kế cũ

Bản cũ: `next_action(belief_state, last_event) -> Action` — tra bảng rule cứng, trả về **đúng 1 bước** kế tiếp.

Ba điểm yếu:
1. **Không có tầm nhìn xa (myopic).** Chỉ tối ưu bước kế tiếp, không biết "3 bước nữa mình sẽ ở đâu". Với KIS
   dưới đồng hồ, một chuỗi 5 bước tốt thắng 5 bước tham lam cục bộ.
2. **Không tận dụng được tri thức về công cụ.** Agent không "biết" rằng OCR mạnh cho query có chữ, region
   search mạnh cho object nhỏ, temporal search cần cho query có thứ tự — nó chỉ khớp `if/else`.
3. **Không giải thích được, không sửa được.** Người dùng không thấy "vì sao agent làm vậy", nên không thể can
   thiệp đúng chỗ — vi phạm chính nguyên tắc "kiểm soát được" đã đặt ra.

---

## 1. Kiến trúc Planning mới — 4 tầng

```
                    ┌──────────────────────────────────────────┐
   Query + Belief   │  1. QUERY ANALYZER                        │
   ───────────────► │     phân rã query -> "hồ sơ truy vấn"     │
                    │     (đặc tính nào? cần công cụ nào?)      │
                    └────────────────┬─────────────────────────┘
                                     ▼
                    ┌──────────────────────────────────────────┐
                    │  2. PLANNER (LLM, registry-bounded)       │
                    │     sinh 2-3 PLAN ứng viên (chuỗi bước)   │
                    │     mỗi plan: [Step, Step, ...] + lý do   │
                    └────────────────┬─────────────────────────┘
                                     ▼
                    ┌──────────────────────────────────────────┐
                    │  3. CRITIC (kiểm chứng TRƯỚC khi chạy)    │
                    │     ước lượng chi phí/thời gian mỗi plan  │
                    │     preview số kết quả (kiểu VIREO)       │
                    │     loại plan bất khả thi                 │
                    └────────────────┬─────────────────────────┘
                                     ▼
                    ┌──────────────────────────────────────────┐
                    │  4. EXECUTOR + MONITOR                    │
                    │     chạy từng Step, fusion tích luỹ       │
                    │     theo dõi tiến triển -> replan nếu kẹt │
                    └──────────────────────────────────────────┘
```

Điểm khác biệt cốt lõi so với SnapMind: SnapMind có Planner + Executor, nhưng **không có tầng Critic**
(kiểm chứng plan *trước khi* chạy) và **không có Monitor** (phát hiện kẹt để replan). Hai tầng này là nơi
mình có thể đi xa hơn.

---

## 2. Tầng 1 — Query Analyzer: "hồ sơ truy vấn" (Query Profile)

Trước khi lập kế hoạch, phải **hiểu query thuộc loại gì**. Đây là thứ bảng rule cũ làm rất thô.

```jsonc
{
  "raw": "tìm cảnh tôi đánh rơi chìa khóa có móc gấu bông hồng khi đi qua quầy hoa quả",
  "profile": {
    "has_small_object": true,        // "móc khóa" -> gợi ý region search / object detector
    "has_attribute_binding": true,   // "gấu bông HỒNG" -> attribute gắn object -> region-CLIP
    "has_temporal_order": false,     // không có "trước/sau"
    "has_action": true,              // "đánh rơi" -> hành động ngắn, keyframe có thể miss
    "has_text_in_scene": false,      // không nhắc chữ -> OCR ít giá trị
    "has_speech_cue": false,         // -> ASR ít giá trị
    "has_time_cue": false,           // không có mốc thời gian -> không hard-filter được
    "has_scene_context": true,       // "quầy hoa quả" -> scene-level CLIP mạnh
    "specificity": "high",           // mô tả chi tiết -> KIS, không phải AVS
    "ambiguity_score": 0.2           // thấp -> ít cần hỏi làm rõ
  },
  "predicates": [ ... ]              // schema Predicate cũ, giữ nguyên
}
```

**Vì sao quan trọng:** hồ sơ này là **đầu vào để Planner chọn công cụ**. Không có nó, Planner phải đoán mù.
Đây chính là thứ biến agent từ "tra if/else" thành "hiểu bài toán trước khi giải".

---

## 3. Tầng 2 — Planner: sinh nhiều plan ứng viên

### 3.1. Tool Registry (đóng, có mô tả năng lực)

Khác bản cũ (chỉ liệt kê Action UI), registry giờ mô tả **năng lực & chi phí** từng tool để Planner chọn đúng:

```jsonc
[
  { "tool": "clip_search",      "cost": "low",    "strength": "scene-level gist, query dài mô tả bối cảnh",
    "weakness": "object nhỏ, binding, đếm, phủ định" },
  { "tool": "ocr_search",       "cost": "low",    "strength": "chữ trong khung hình (banner, biển hiệu)",
    "weakness": "vô dụng nếu cảnh không có chữ", "gated_by": "profile.has_text_in_scene" },
  { "tool": "asr_search",       "cost": "low",    "strength": "lời thoại", "gated_by": "profile.has_speech_cue" },
  { "tool": "object_filter",    "cost": "low",    "strength": "có/không có object, đếm số lượng",
    "weakness": "chỉ 600 category Open Images" },
  { "tool": "region_search",    "cost": "medium", "strength": "object nhỏ + binding thuộc tính (PraK V4)",
    "gated_by": "profile.has_small_object OR profile.has_attribute_binding" },
  { "tool": "temporal_search",  "cost": "medium", "strength": "chuỗi 2+ sự kiện có thứ tự",
    "gated_by": "profile.has_temporal_order" },
  { "tool": "hyde_image",       "cost": "high",   "strength": "khái niệm khó diễn đạt bằng text (né modality gap)",
    "weakness": "hồi quy về nguyên mẫu -> rủi ro cho KIS" },
  { "tool": "vlm_verify",       "cost": "high",   "strength": "verify quan hệ/hành động phức tạp trên shortlist nhỏ" },
  { "tool": "ask_user",         "cost": "turn",   "strength": "bơm bit thật khi belief mơ hồ",
    "gated_by": "profile.ambiguity_score > 0.5" }
]
```

> **`gated_by` là mấu chốt**: tool chỉ được đưa vào plan nếu hồ sơ truy vấn *cho phép* — ngăn Planner gọi OCR
> cho query không có chữ (lãng phí), hay temporal_search cho query không có thứ tự. Đây là **tri thức miền
> được mã hoá tường minh**, không phó mặc LLM tự đoán.

### 3.2. Plan = chuỗi Step, không phải 1 Action đơn lẻ

```jsonc
{
  "plan_id": "A",
  "rationale": "Query có object nhỏ + binding thuộc tính -> ưu tiên thu hẹp bằng scene rồi verify vùng",
  "steps": [
    { "n": 1, "tool": "clip_search",    "args": {"q": "quầy bán hoa quả chín"}, "op": "search", "top_k": 1000 },
    { "n": 2, "tool": "object_filter",  "args": {"must_have": ["person"]},      "op": "filter" },
    { "n": 3, "tool": "region_search",  "args": {"region_hint": "auto", "q": "móc khóa gấu bông màu hồng"},
      "op": "rerank", "top_k": 100 },
    { "n": 4, "tool": "vlm_verify",     "args": {"question": "có ai đang đánh rơi chìa khóa không?"},
      "op": "rerank", "top_k": 20 }
  ],
  "est_cost": "medium",
  "est_time_s": 6.5
}
```

**Sinh 2-3 plan khác nhau về chiến lược** (không phải 3 biến thể na ná):
- Plan A — "thu hẹp bằng bối cảnh trước, verify chi tiết sau" (an toàn, recall cao).
- Plan B — "đánh thẳng vào chi tiết đặc trưng nhất trước" (nhanh nếu chi tiết đủ hiếm; rủi ro nếu detector miss).
- Plan C — "hỏi người dùng làm rõ trước rồi mới tìm" (khi ambiguity cao).

→ Người dùng thấy **cả 3 plan + lý do**, chọn/sửa/bỏ qua — đúng tinh thần SnapMind nhưng có **rationale tường
minh cho từng plan**, không chỉ là danh sách tham số.

### 3.3. Ba mức tự chủ (lấy nguyên từ SnapMind, đã có bằng chứng dùng được)

| Mức | Hành vi | Khi nào dùng |
|---|---|---|
| **Guide** | Hiện plan, người dùng tự bấm chạy từng bước | Người mới, hoặc query lạ cần quan sát kỹ |
| **Assist** | Người dùng duyệt plan, agent chạy hết, dừng ở checkpoint quan trọng | **Mặc định** — cân bằng tốc độ/kiểm soát |
| **Auto** | Approve 1 lần, agent chạy tới khi có kết quả hoặc kẹt | Khi tin tưởng + gấp thời gian |

---

## 4. Tầng 3 — CRITIC: kiểm chứng plan TRƯỚC khi chạy (phần SnapMind không có)

Đây là nơi thiết kế của mình đi xa hơn các hệ thống đã đọc. Chạy 3 kiểm tra **rẻ** trước khi tốn thời gian thật:

### 4.1. Ước lượng số kết quả (borrow từ VIREO's "recommendation shading")
Với mỗi bước `filter`, dùng **thống kê precompute** (đếm sẵn lúc indexing) để ước lượng **còn bao nhiêu ứng
viên sau bước đó** — *không cần chạy thật*.

```
Bước 2 (object_filter: person) -> ước lượng còn ~62% corpus  ⚠️ lọc quá yếu, gần như vô ích
Bước 3 (region_search)         -> ước lượng còn ~100 ứng viên ✓
```
→ Critic **cảnh báo bước lọc vô dụng** (giữ lại 62% = tốn thời gian không thu hẹp gì) và đề nghị bỏ/thay.

### 4.2. Kiểm tra "bước lọc chết" (dead-end detection)
Nếu ước lượng cho ra **0 ứng viên**, plan sẽ hỏng → Critic loại plan đó *trước khi* chạy, thay vì để người
dùng chờ 6 giây rồi nhận danh sách rỗng.

### 4.3. Ước lượng thời gian & đối chiếu ngân sách
KIS có đồng hồ. Critic cộng `est_time` từng bước, so với **thời gian còn lại**:
```
Plan A: 6.5s  ✓ (còn 4:12)
Plan B: 2.1s  ✓ nhanh nhất
Plan C: cần 2 lượt hỏi người dùng (~30s) ⚠️ chỉ dùng nếu còn nhiều thời gian
```
→ Khi thời gian còn ít, Critic **tự động ưu tiên plan rẻ**, không đề xuất plan cần hỏi han dài dòng.

---

## 5. Tầng 4 — EXECUTOR + MONITOR: phát hiện kẹt và replan

### 5.1. Fusion tích luỹ (SnapMind's incremental fusion)
Mỗi bước không vứt kết quả bước trước, mà **hợp nhất tích luỹ** qua rank aggregation:
- `CombSUM` / `CombMNZ` / **RRF** (Reciprocal Rank Fusion) — Exquisitor 2026 dùng RRF (k=60) làm tie-break.
- Giữ `source_contrib` cho mỗi item: *item này lên top nhờ CLIP 60%, OCR 30%, object 10%* → **giải thích được**.

### 5.2. Monitor — 4 tín hiệu "đang kẹt" (chưa hệ thống nào trong 29 paper làm tường minh)

| Tín hiệu | Cách đo | Hành động |
|---|---|---|
| **Ranking không đổi** | Jaccard(top-20 bước n, bước n-1) > 0.9 | Bước này vô dụng → bỏ, chuyển tool khác |
| **Điểm số bẹt (flat)** | (score_top1 − score_top20) / score_top1 < 0.05 | Không có ứng viên nào nổi bật → query quá mơ hồ, chuyển sang `ask_user` |
| **Tập ứng viên cạn** | len(candidates) == 0 sau 1 filter | Rollback bước đó, nới lỏng điều kiện |
| **Hết ngân sách thời gian** | elapsed > 0.7 × budget | Dừng tinh chỉnh, trả kết quả tốt nhất hiện có |

→ Khi kẹt: **replan** — quay lại tầng 2, sinh plan mới **có tính đến lịch sử đã thất bại** ("đã thử clip_search
với 'quầy hoa quả', ranking bẹt → thử diễn đạt khác hoặc đổi kênh").

### 5.3. Early stopping (SnapMind) — biết khi nào NÊN dừng
Dừng khi ranking đã ổn định: `Jaccard(top-K) > θ` **và** `ΔNDCG@K < ε` giữa 2 bước liên tiếp → thêm bước nữa
cũng không cải thiện, dừng lại tiết kiệm thời gian.

---

## 6. Vòng lặp học trong phiên (in-session learning)

Không cần train gì — chỉ cần **cập nhật trọng số tool theo phản hồi thực tế trong chính phiên đó**:

```
Nếu user click ảnh từ kết quả OCR nhiều lần  -> tăng trọng số ocr_search cho các bước sau
Nếu region_search liên tục trả về rác        -> giảm trọng số, Planner tránh dùng ở plan tiếp theo
```

Đây là bản nhẹ của **Rocchio/SVM online learning** mà PraK V4 dùng cho AVS — nhưng áp ở **cấp độ chọn tool**,
không chỉ ở cấp độ xếp hạng item. Chưa thấy hệ thống nào trong 29 paper làm điều này.

---

## 7. Ví dụ chạy đầy đủ — thấy rõ giá trị của Critic & Monitor

```
Query: "cảnh tôi đánh rơi chìa khóa có móc gấu bông hồng khi đi qua quầy hoa quả"

[1] ANALYZER -> profile: {small_object: T, binding: T, text_in_scene: F, temporal: F, ambiguity: 0.2}
    -> gated: ocr_search BỊ LOẠI (không có chữ), temporal_search BỊ LOẠI (không có thứ tự)

[2] PLANNER -> sinh 3 plan:
    A: clip(quầy hoa quả) -> object_filter(person) -> region(móc khóa hồng) -> vlm_verify(đánh rơi)
    B: region(móc khóa gấu bông hồng) trực tiếp -> clip rerank(quầy hoa quả)
    C: ask_user("bạn nhớ quầy hoa quả trong nhà hay ngoài trời?") -> rồi mới tìm

[3] CRITIC:
    A: bước object_filter(person) -> ước lượng giữ 62% corpus ⚠️ LỌC VÔ DỤNG -> đề nghị bỏ bước này
       (sửa thành: clip -> region -> vlm_verify, est 5.2s) ✓
    B: est 2.1s ✓ nhanh nhất, nhưng rủi ro nếu detector không bắt được móc khóa nhỏ
    C: ambiguity chỉ 0.2 -> KHÔNG CẦN hỏi ⚠️ loại plan C (lãng phí 1 lượt)
    -> Đề xuất: A (đã sửa) làm chính, B làm dự phòng

[4] EXECUTOR (mức Assist):
    Bước 1: clip_search -> 1000 ứng viên, score top1=0.34, top20=0.31 -> điểm BẸT ⚠️
    MONITOR: flat score -> "quầy hoa quả" không đủ phân biệt trong corpus này
    -> REPLAN: chuyển sang plan B (đánh thẳng vào chi tiết hiếm: móc khóa gấu bông)
    Bước 1': region_search(móc khóa gấu bông hồng) -> 40 ứng viên, top1=0.41 vs top20=0.22 -> PHÂN BIỆT TỐT ✓
    Bước 2': clip rerank(quầy hoa quả) trên 40 ứng viên -> top-5 rõ ràng
    -> show_image_grid(5 ảnh) + source_contrib: "lên top nhờ region 70% / scene 30%"
```

→ Nếu không có Critic, plan A đã tốn 1 bước lọc vô dụng. Nếu không có Monitor, agent đã đi tiếp với một
coarse recall bẹt và thất bại. **Đây chính là "thông minh" mà bản rule-based cũ không có.**

---

## 8. So sánh với thiết kế cũ và với SnapMind

| | Bản cũ (`kien-truc-2-tang`) | SnapMind (VBS26) | Bản nâng cấp này |
|---|---|---|---|
| Đơn vị quyết định | 1 Action | 1 Plan (chuỗi) | 1 Plan (chuỗi) + rationale |
| Chọn công cụ | if/else cứng | LLM tự chọn từ registry | LLM chọn, nhưng **gated bởi Query Profile** |
| Kiểm chứng trước khi chạy | ✗ | ✗ | ✓ **CRITIC** (ước lượng kết quả/thời gian, loại plan hỏng) |
| Phát hiện kẹt & replan | ✗ | ✗ | ✓ **MONITOR** (4 tín hiệu) |
| Giải thích | ✗ | ✓ source_contrib | ✓ source_contrib + rationale plan |
| Học trong phiên | ✗ | ✗ | ✓ trọng số tool cập nhật theo click |
| Mức tự chủ | ✗ | ✓ Guide/Assist/Auto | ✓ (giữ nguyên) |
| Dừng đúng lúc | ngưỡng entropy | ✓ early stopping | ✓ early stopping + ngân sách thời gian |

---

## 9. Thứ tự xây dựng (thực dụng — đừng làm hết cùng lúc)

1. **Query Analyzer + Tool Registry có `gated_by`** — rẻ, giá trị cao nhất, dùng được ngay cả khi chưa có LLM Planner.
2. **Critic: ước lượng số kết quả** (cần thống kê precompute lúc indexing — làm sớm cùng lúc build index).
3. **Monitor: 4 tín hiệu kẹt** — chỉ là vài phép đo trên ranking, rất rẻ, nhưng cứu được nhiều tình huống hỏng.
4. **LLM Planner sinh plan** — cần prompt engineering, làm sau khi 3 phần trên đã chạy.
5. **Fusion tích luỹ + source_contrib** — cần khi có nhiều kênh, chưa cần ở giai đoạn 1 kênh CLIP.
6. **Học trong phiên** — nice-to-have, làm cuối.

---

## 10. Câu hỏi còn mở

- **Critic ước lượng số kết quả cần thống kê precompute nào?** Với `object_filter` thì đếm sẵn số frame chứa
  mỗi label là đủ. Với `clip_search` thì khó ước lượng trước → có thể chỉ ước lượng cho các bước `filter`
  (rời rạc), không ước lượng cho `search` (liên tục).
- **Replan có nguy cơ lặp vô hạn** (A kẹt → B → B kẹt → A). Cần giới hạn số lần replan (vd tối đa 2) và giữ
  lịch sử plan đã thử để không lặp lại.
- **Ngưỡng cụ thể** (Jaccard > 0.9, flat < 0.05, θ, ε...) — phải tinh chỉnh bằng validation set thật
  (`bo-metric-va-validation-set.md`), không đoán.

---

## 11. Nối với các note khác

- Thay thế mục 4 ("Rule chọn Action") của `kien-truc-2-tang-agent-va-ui.md` — bảng rule cũ giờ chỉ còn là
  **fallback** khi LLM Planner không khả dụng.
- Tái sử dụng nguyên: schema `Predicate`, `evaluate_predicate`, `calibrate`, `combine_scores`
  (`phac-thao-compositional-scoring.md`); `BeliefState`, `update_belief` (`phac-thao-belief-state-kisc.md`).
- Tập Action UI (6 loại) giữ nguyên — Planner sinh Step ở tầng *logic*, Step nào cần hỏi người dùng thì
  ánh xạ ra Action UI tương ứng.
