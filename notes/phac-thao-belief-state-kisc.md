# Phác thảo — Belief-State & Chọn câu hỏi cho KISC (Lỗ hổng 5)

> Nối tiếp toàn bộ chuỗi note trước, đặc biệt tái sử dụng schema `Predicate` từ
> `phac-thao-compositional-scoring.md`. Đây là bản thiết kế cho Tầng 5 (Interaction) — vòng lặp thu hẹp
> tập ứng viên qua nhiều lượt, dùng chung cho cả KISC (hệ thống chủ động hỏi) lẫn UI người-lái thường
> (người dùng tự áp filter/feedback).
> Ngày: 2026-07-06.

---

## 1. Vị trí trong pipeline & vòng lặp tổng thể

```
Turn 0: query mơ hồ ban đầu → Filter + Coarse Recall (Tầng 1-2) → candidate set lớn (vài trăm-nghìn)
Turn i: chọn hành động kế tiếp (câu hỏi hệ thống hỏi, HOẶC filter người dùng áp)
        → nhận thông tin mới → parse thành Predicate → update belief-state → tập ứng viên co lại
...lặp đến khi: belief đủ chắc (top-1 tách biệt rõ) HOẶC hết lượt cho phép
Turn cuối: chạy Compositional Scoring Engine (Tầng 3, note trước) đầy đủ trên candidate set đã nhỏ → chốt kết quả
```

**Điểm quan trọng:** vòng lặp này **bọc bên ngoài** pipeline coarse/fine, không thay thế nó. Mỗi lượt chỉ làm việc
*rẻ* (filter/reweight); việc *đắt* (compositional scoring đầy đủ, LVLM verify) chỉ chạy **một lần cuối** khi
candidate set đã đủ nhỏ — đúng nguyên lý lỗ hổng 4.

---

## 2. Data model — Belief State

```python
@dataclass
class BeliefState:
    candidates: dict[frame_id, float]      # điểm/khối lượng xác suất hiện tại của từng ứng viên
    applied_predicates: list[Predicate]    # lịch sử toàn bộ ràng buộc đã áp (để tránh hỏi lặp)
    turn_count: int
    turn_budget: int                       # giới hạn lượt cho phép (luật chơi KISC)
```

`candidates` khởi tạo từ **kết quả Coarse Recall** (Tầng 2) sau khi qua Filter (Tầng 1) — không phải toàn bộ corpus.
Điểm ban đầu = điểm CLIP similarity, chuẩn hoá thành tổng = 1 (coi như một phân phối xác suất thô).

---

## 3. Cơ chế cập nhật — mọi thông tin mới đều đi qua cùng một hàm

**Ý tưởng cốt lõi:** dù thông tin đến từ *câu trả lời hội thoại* (KISC) hay *cú click filter* (UI thường) hay
*feedback "gần đúng"* (relevance feedback) — tất cả đều được quy về **một Predicate** (dùng lại schema ở note trước)
và áp vào belief state theo **một trong hai chế độ**:

```python
def update_belief(belief: BeliefState, predicate: Predicate, mode: str) -> BeliefState:
    if mode == "hard_filter":
        # ràng buộc tin cậy cao (thời gian, địa điểm rõ ràng, phủ định)
        belief.candidates = {
            f: s for f, s in belief.candidates.items()
            if evaluate_predicate(predicate, f) > HARD_THRESHOLD
        }
    elif mode == "soft_reweight":
        # tín hiệu mềm/nhiễu (mô tả cảnh mờ, thuộc tính khó chắc chắn)
        for f in belief.candidates:
            belief.candidates[f] *= calibrate(predicate.type, evaluate_predicate(predicate, f))
        renormalize(belief.candidates)

    belief.applied_predicates.append(predicate)
    belief.turn_count += 1
    return belief
```

**Quy tắc chọn `mode`** (đây chính là nguyên lý lỗ hổng 5.2 áp dụng cụ thể):

| Loại predicate | mode | Vì sao |
|---|---|---|
| `scene` mốc thời gian/địa điểm rõ (từ Metadata) | `hard_filter` | tin cậy cao, rẻ, an toàn để loại thẳng |
| phủ định bất kỳ | `hard_filter` | đã lập luận ở note trước — negation nên loại thẳng |
| `attribute`, `relation` mô tả từ hội thoại (màu áo, giới tính...) | `soft_reweight` | thông tin từ trí nhớ người dùng — có thể sai lệch nhẹ, không nên loại tuyệt đối |
| `scene` mô tả cảnh mờ ("không khí ấm cúng"...) | `soft_reweight` | tín hiệu yếu, để combiner sau xử lý mềm |

> Đây chính là bảng cụ thể hoá nguyên lý đã nêu ở lỗ hổng 5: **filter bằng thứ bạn chắc, rank bằng thứ bạn không chắc.**

**Feedback dạng "click gần đúng" (không qua ngôn ngữ):** không parse thành predicate symbolic, mà thành một
**dịch chuyển vector** kiểu Rocchio — mở rộng nhẹ schema để chấp nhận predicate "ẩn":
```python
{"type": "embedding_nudge", "vector_delta": liked_embedding - disliked_embedding, "alpha": 0.3}
```
Áp dụng: cộng `alpha * vector_delta` vào embedding truy vấn hiện tại rồi tính lại cosine similarity làm điểm reweight —
tái dùng đúng luồng `soft_reweight`, chỉ khác nguồn gốc predicate.

---

## 4. Chọn câu hỏi kế tiếp — thuật toán information gain

Đây là phần biến KISC từ "hỏi theo cảm tính" thành "hỏi theo tối ưu" (lỗ hổng 5.3).

### 4.1. Sinh ứng viên câu hỏi (candidate axes)

Từ chính `candidates` hiện tại, trích ra các **trục thuộc tính** có thể hỏi — không hỏi tuỳ tiện mà hỏi trên
những gì **dữ liệu đã có sẵn khả năng phân biệt** (Objects cho giới tính/số người, CLIP scene cho trong/ngoài,
Metadata cho khung giờ...):

```python
CANDIDATE_AXES = [
    {"axis": "indoor_outdoor", "question": "Không gian đó trong nhà hay ngoài trời?", "evaluator": "scene"},
    {"axis": "person_gender",  "question": "Người bạn đó là nam hay nữ?",              "evaluator": "attribute"},
    {"axis": "num_people",     "question": "Lúc đó có bao nhiêu người?",               "evaluator": "count"},
    {"axis": "time_of_day",    "question": "Đó là buổi sáng, chiều hay tối?",          "evaluator": "scene"},
    ...
]
```

### 4.2. Ước lượng phân phối câu trả lời (không hỏi thật, dự đoán trước)

Với mỗi trục, **dùng chính evaluator đã có** (từ note trước!) chấm điểm từng ứng viên theo từng giá trị-trả-lời
khả dĩ, để **dự đoán trước** nếu hỏi câu này, các ứng viên sẽ chia thành các nhóm cỡ bao nhiêu:

```python
def predict_answer_distribution(belief: BeliefState, axis) -> dict[answer_value, float]:
    buckets = defaultdict(float)
    for frame_id, weight in belief.candidates.items():
        best_answer = argmax_over_possible_answers(
            lambda a: evaluate_predicate(make_predicate(axis, a), frame_id)
        )
        buckets[best_answer] += weight
    return normalize(buckets)
```

### 4.3. Tính information gain kỳ vọng, chọn trục tối đa hoá nó

```python
def entropy(dist: dict) -> float:
    return -sum(p * math.log2(p) for p in dist.values() if p > 0)

def expected_info_gain(belief: BeliefState, axis) -> float:
    current_H = entropy(belief.candidates)          # entropy hiện tại của belief
    answer_dist = predict_answer_distribution(belief, axis)
    # entropy kỳ vọng SAU khi biết câu trả lời (điều kiện hoá theo từng nhánh trả lời)
    expected_post_H = 0.0
    for answer, p_answer in answer_dist.items():
        sub_belief = simulate_update(belief, axis, answer)
        expected_post_H += p_answer * entropy(sub_belief.candidates)
    return current_H - expected_post_H

def choose_next_question(belief: BeliefState, candidate_axes) -> Axis:
    unasked = [a for a in candidate_axes if a not in belief.applied_predicates]
    return max(unasked, key=lambda a: expected_info_gain(belief, a))
```

**Trực giác:** trục nào **chia belief ~50/50** cho gain gần 1 bit (tốt); trục nào 95% ứng viên rơi vào cùng
1 nhánh trả lời cho gain gần 0 bit (phí lượt hỏi) — khớp đúng nguyên lý "20 câu hỏi" đã nêu ở lỗ hổng 5.3.

### 4.4. Ràng buộc khả-trả-lời (answerability constraint)

Information gain lý thuyết cao vô nghĩa nếu người dùng **không nhớ nổi câu trả lời**. Cần lọc bớt trục trước khi
đưa vào bước 4.3:
- **Ưu tiên** trục thuộc phạm trù lớn, dễ nhớ: trong/ngoài, ngày/đêm, một mình/có người khác, địa điểm loại gì.
- **Tránh** trục quá chi tiết/không chắc trong trí nhớ: màu chính xác, số đếm chính xác lớn, chi tiết nhỏ.
- Có thể mã hoá đơn giản bằng một **danh sách trắng trục được phép hỏi** (curated theo tay), thay vì để LLM tự nghĩ
  câu hỏi tự do — an toàn hơn ở giai đoạn brainstorm/mới xây.

---

## 5. Điều kiện dừng (termination)

Dừng hỏi và chốt câu trả lời khi **một trong ba** điều kiện xảy ra:

```python
def should_terminate(belief: BeliefState) -> bool:
    sorted_scores = sorted(belief.candidates.values(), reverse=True)
    top1, top2 = sorted_scores[0], sorted_scores[1] if len(sorted_scores) > 1 else 0
    confident = (top1 > CONFIDENCE_THRESHOLD) and (top1 - top2 > MARGIN_THRESHOLD)
    small_enough = len(belief.candidates) <= SHOW_DIRECTLY_THRESHOLD  # vd <= 5, show cho user chọn tay
    budget_exhausted = belief.turn_count >= belief.turn_budget
    return confident or small_enough or budget_exhausted
```

Khi dừng vì `small_enough` (không phải vì đủ tự tin) → **không nên đoán bừa**, mà nên chạy full Compositional
Scoring Engine (note trước) trên tập nhỏ còn lại rồi chốt Top-1 — đây là lúc chuyển việc "đắt" vào, đúng như
mục 1 đã mô tả.

---

## 6. Hiển thị đa dạng — MMR (khi cần show ứng viên cho người dùng chọn/feedback)

Nếu top-K hiển thị đều là biến thể gần giống nhau của "một cách hiểu" query, người dùng không có gì mới để
feedback (lãng phí lượt). Dùng **Maximal Marginal Relevance** để chọn tập hiển thị vừa liên quan vừa đa dạng:

```python
def select_display_set(belief: BeliefState, k: int, mu: float = 0.7) -> list[frame_id]:
    selected = []
    remaining = list(belief.candidates.keys())
    while len(selected) < k and remaining:
        def mmr_score(f):
            relevance = belief.candidates[f]
            if not selected:
                return relevance
            max_sim = max(embedding_similarity(f, s) for s in selected)
            return mu * relevance - (1 - mu) * max_sim
        best = max(remaining, key=mmr_score)
        selected.append(best)
        remaining.remove(best)
    return selected
```

`mu` cao → ưu tiên relevance (an toàn); `mu` thấp → ưu tiên đa dạng (khai phá). Có thể **giảm dần `mu` qua các lượt**:
lượt đầu ưu tiên đa dạng để thu thập nhiều loại tín hiệu, lượt cuối ưu tiên relevance để hội tụ.

---

## 7. Ví dụ chạy tay — đúng kịch bản slide 15

**Turn 1.** Query: *"Tìm giúp tôi đoạn video tôi gặp một người bạn cũ vào tuần trước."*
→ `hard_filter` theo Metadata (`publish_date` trong tuần trước) + Coarse Recall trên "gặp bạn cũ" →
`candidates` còn ~vài trăm, entropy còn cao.

**Turn 2.** Hệ thống tính `expected_info_gain` cho các trục `{indoor_outdoor, person_gender, num_people, ...}`.
Giả sử `indoor_outdoor` chia belief ~55/45 (gain cao), `person_gender` chia ~50/50 (gain cao nhất) →
chọn hỏi **`person_gender`** trước (hoặc hỏi cả hai một lượt nếu luật cho phép nhiều câu/lượt — slide gộp cả hai
trong 1 lượt, mô hình ở đây coi là 2 predicate áp liên tiếp).
→ Trả lời: "ngoài trời, nam" → 2 predicate `soft_reweight` áp vào belief → `candidates` co mạnh, entropy giảm rõ rệt.

**Turn 3.** Người dùng cung cấp thêm (không cần hệ thống hỏi): *"quán cà phê ngoài trời, áo sơ mi xanh dương"*
→ parse thành `scene(outdoor cafe)` (`hard_filter` — đủ cụ thể) + `attribute(shirt, blue)` (`soft_reweight`)
→ `should_terminate` kiểm tra: nếu `candidates` đã ≤5 → chuyển sang Compositional Scoring Engine đầy đủ.

**Turn 4.** Compositional Scoring Engine (note trước) chạy full trên tập nhỏ còn lại, chốt Top-1.

---

## 8. Khung code tổng thể (pseudocode)

```python
def run_kisc_session(initial_query: str, turn_budget: int) -> frame_id:
    parsed = parse_query(initial_query)          # dùng lại parser của note Compositional Scoring
    hard_preds = [p for p in parsed.predicates if is_high_confidence(p)]   # thời gian, địa điểm rõ
    candidates = coarse_recall(parsed, filters=hard_preds)
    belief = BeliefState(candidates=normalize(candidates), applied_predicates=hard_preds,
                          turn_count=0, turn_budget=turn_budget)

    while not should_terminate(belief):
        axis = choose_next_question(belief, CANDIDATE_AXES)
        answer = ask_user_or_system(axis.question)              # KISC: hệ thống hỏi; UI: người dùng tự áp
        predicate = make_predicate(axis, answer)
        mode = "hard_filter" if is_high_confidence(predicate) else "soft_reweight"
        belief = update_belief(belief, predicate, mode)

    shortlist = top_n(belief.candidates, n=100)
    final_ranked = rerank(parsed, shortlist)       # gọi Compositional Scoring Engine (note trước)
    return final_ranked[0][0]
```

---

## 9. Kiểu lỗi cần tự cảnh giác

| Lỗi | Triệu chứng (xem note metric) | Cách giảm nhẹ |
|---|---|---|
| Hỏi câu không ai trả lời được ("màu chính xác") | Turn efficiency thấp, người dùng trả lời "không nhớ" nhiều | Answerability constraint (mục 4.4) — whitelist trục dễ nhớ |
| Trả lời "không chắc/không nhớ" bị xử lý như filter cứng | Belief sụp sai (loại oan ứng viên đúng) | Câu trả lời mơ hồ → **bỏ qua (skip)**, không update, không tính vào turn_count là thất bại |
| Predicate đối lập giữa các lượt (người dùng đổi ý/nhớ nhầm) | Belief co về tập rỗng | Cho phép "quên" (decay) các predicate cũ có trọng số thấp thay vì hard-filter tích luỹ vô hạn |
| Info gain ước lượng dựa trên evaluator yếu (lỗi lan từ lỗ hổng 1/2) | Câu hỏi "tối ưu" trên giấy nhưng không chia belief thật như dự đoán | Info gain chỉ tốt bằng evaluator bên dưới — cần Precision@1 của compositional scorer đã ổn trước khi tối ưu phần chọn câu hỏi |
| Hiển thị đa dạng quá mức làm mất relevance | người dùng thấy toàn kết quả không liên quan | giảm dần `mu` qua lượt, không cố định |

---

## 10. Điểm nối ngược lại note Compositional Scoring

Hai note này **dùng chung một hạ tầng**:
- Cùng schema `Predicate`, cùng registry `evaluate_predicate` + `calibrate`.
- Compositional Scoring Engine = "chấm điểm 1 lần, đầy đủ, trên shortlist nhỏ" (Tầng 3).
- Belief-State Loop = "chấm điểm nhiều lần, từng phần, trên tập lớn hơn, co dần" (Tầng 5), rồi **giao lại** cho
  Compositional Scoring Engine ở bước chốt cuối cùng.

Không cần xây hai hệ thống tách biệt — chỉ cần xây tốt bộ evaluator/calibrator **một lần**, cả hai note đều thừa hưởng.
