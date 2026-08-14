# Phác thảo — Compositional Scoring Engine (Lỗ hổng 2)

> Nối tiếp toàn bộ chuỗi note trước. Đây là bản thiết kế **đủ cụ thể để bắt đầu code** cho Tầng 3 (Fine Verify)
> của pipeline: nhận query + shortlist từ Coarse Recall, trả về danh sách đã xếp hạng lại theo đúng logic tổ hợp (AND).
> Ngày: 2026-07-06.

---

## 1. Vị trí trong pipeline & input/output

```
Input:  query_text, shortlist = [frame_1 ... frame_K]  (K ~ 100-1000, đã qua Coarse Recall + Filter)
Output: ranked_shortlist = [(frame_i, combined_score_i)]  đã sắp xếp giảm dần
```

Engine này **không chạy trên toàn corpus** — chỉ trên shortlist đã thu hẹp (đúng nguyên lý lỗ hổng 4:
việc đắt chỉ chạy trên tập nhỏ).

---

## 2. Schema dữ liệu — "Parsed Query"

Đây là cấu trúc trung tâm — mọi thứ khác (parser, evaluator, combiner) đều xoay quanh nó.
**Lưu ý quan trọng:** schema này sẽ được **tái sử dụng nguyên vẹn** ở note KISC (belief-state) — cùng một
loại "predicate" vừa dùng để chấm điểm, vừa dùng để cập nhật belief state khi có thông tin mới từ hội thoại.

```jsonc
{
  "raw_query": "Tôi vô tình đánh rơi chiếc chìa khóa có móc khóa hình gấu bông màu hồng khi đang đi bộ qua quầy bán hoa quả",
  "predicates": [
    { "id": "p1", "type": "scene",     "value": "fruit stand / market stall" },
    { "id": "p2", "type": "exists",    "object": "keychain",  "attributes": ["teddy bear shape", "pink"] },
    { "id": "p3", "type": "relation",  "subject": "person", "predicate": "dropping", "object": "keychain" },
    { "id": "p4", "type": "attribute", "object_ref": "p2",  "attribute": "pink color" }
  ],
  "negations": [],
  "temporal_constraints": [],
  "logic": "AND"
}
```

### Các loại predicate (predicate types) và evaluator tương ứng

| type | Ý nghĩa | Evaluator bản địa (route tới đâu) |
|---|---|---|
| `scene` | ngữ cảnh/toàn cảnh | CLIP toàn ảnh vs `"a photo of {value}"` |
| `exists` | có mặt 1 object | max theo box của Objects (nhãn khớp) HOẶC region-CLIP nếu object không nằm trong 600 category |
| `attribute` | thuộc tính gắn với 1 object cụ thể | Region-CLIP: crop box của object đó, so với `"{attribute} {object}"` — **luôn cần `object_ref`** để biết crop vùng nào |
| `count` | số lượng | đếm box cùng nhãn từ Objects, so với số yêu cầu |
| `text_in_frame` | chữ hiển thị trong khung hình | khớp mờ trên kênh OCR |
| `speech` | lời nói | khớp mờ trên kênh ASR (theo timestamp gần frame) |
| `relation` | quan hệ giữa 2 object | heuristic không gian từ box (near/left-of/above), hoặc gọi LVLM nếu là hành động (holding, dropping...) |
| *(trong `negations`)* | phủ định 1 predicate bất kỳ | chấm predicate gốc rồi **đảo dấu + chặn cứng**, không blend mềm |

> **Kỷ luật bắt buộc:** parser chỉ được sinh ra predicate mà evaluator registry **có cách chấm**.
> Nếu LLM trích ra thứ không route được (vd "cảm giác hoài niệm") → gắn `"evaluator": "none"` và **loại khỏi combine**,
> không được để nó lặng lẽ nhận điểm 0 (0 sẽ giết cả `min`).

---

## 3. Parser — text → Parsed Query

Thực dụng nhất ở giai đoạn brainstorm: dùng LLM với **structured output** (JSON schema ép buộc), few-shot bằng
chính các ví dụ trong slide tập huấn (kem+biển, túi tím/trắng, chìa khóa+quầy hoa quả).

**Nguyên tắc prompt (không phải prompt cụ thể, mà nguyên tắc thiết kế nó):**
- Liệt kê rõ **danh sách type hợp lệ** trong prompt (khớp đúng bảng ở mục 2) — không để LLM tự sáng tạo type mới.
- Yêu cầu LLM **giữ lại nguyên câu gốc** như một predicate `scene`-toàn-cảnh dự phòng (fallback baseline) — để nếu parse
  sai/thiếu, hệ thống vẫn thoái hoá mượt về hành vi CLIP-toàn-câu thay vì mất trắng.
- Với object được lặp lại có thuộc tính khác nhau ("túi tím" và "túi trắng") → gán `object_ref` khác nhau dù cùng
  `object` type "purse", đánh dấu instance khác nhau ngay từ lúc parse (chuẩn bị cho bước binding ở mục 5).

---

## 4. Calibration — bước hay bị bỏ quên nhưng quyết định thành/bại

Điểm thô từ các evaluator **khác thang hoàn toàn**:
- CLIP cosine similarity: thường nằm hẹp trong khoảng ~0.15–0.35, không phải [0,1] có ý nghĩa trực quan.
- Detector confidence: đã là [0,1] nhưng phân phối khác (thường lệch cao khi object rõ ràng).
- OCR: điểm khớp mờ (edit distance / embedding similarity) có thang riêng.

**Đưa `min`/`weighted` lên điểm chưa hiệu chỉnh là vô nghĩa** — kênh có thang điểm "co cụm" (như CLIP) sẽ luôn thua
kênh có thang điểm "trải rộng" (như detector confidence) trong phép `min`, bất kể độ tin cậy thực sự.

**Hai cách calibrate, chọn theo lượng dữ liệu có sẵn:**

1. **Relative calibration (không cần validation set, dùng ngay được):** với mỗi predicate, chuẩn hoá điểm
   *trong phạm vi shortlist hiện tại* — ví dụ percentile-rank hoặc min-max theo phân phối điểm của chính K ứng viên đó.
   Ý tưởng: ta chỉ cần biết "ứng viên nào tốt hơn ứng viên nào **cho query này**", không cần biết ý nghĩa tuyệt đối.
2. **Learned calibration (cần validation set từ note trước):** fit một hàm logistic đơn giản
   `P(predicate đúng | điểm thô)` riêng cho từng loại evaluator, dùng nhãn từ validation set (mục 3.3 của note metric).
   Tốt hơn nhưng cần dữ liệu gán nhãn.

→ **Gợi ý thực dụng:** bắt đầu bằng (1) vì rẻ và không cần dữ liệu, sau đó nâng cấp sang (2) khi đã tích luỹ đủ
validation set qua các lần eval.

---

## 5. Binding — nhóm predicate cùng object-type nhưng khác instance

Đây là phần **dễ làm sai nhất** nếu chấm độc lập. Với "túi tím VÀ túi trắng" (slide 39):

**Sai (chấm độc lập):** `score(p_tím) = max qua mọi vùng túi phát hiện được`, `score(p_trắng) = max qua mọi vùng túi`
→ nếu cùng 1 túi vừa hơi tím vừa hơi trắng thì có thể **cùng một vùng ăn điểm cho cả hai predicate** — vô lý.

**Đúng (group + assignment):**
1. Gom các predicate `attribute` cùng tham chiếu tới cùng `object` type nhưng khác `object_ref` thành 1 **binding group**.
2. Liệt kê tất cả instance đã detect được của object type đó trong frame (từ Objects — box list).
3. Với mỗi cặp (instance, predicate trong group), tính điểm attribute (region-CLIP trên crop của instance đó).
4. Giải bài toán **gán 1-1** (assignment): mỗi predicate phải được gán cho một instance *khác nhau*, tối đa tổng điểm.
   Với số instance nhỏ (thường ≤ chục cái/frame) — **thuật toán tham lam hoặc Hungarian** đều đủ nhanh.
5. Nếu số instance detect được **ít hơn** số predicate trong group (ví dụ chỉ thấy 1 túi mà cần 2 màu khác nhau)
   → group này **infeasible**, trả điểm rất thấp/loại — vì về mặt logic không thể thoả cả hai attribute trên 1 vật.

```python
def resolve_binding_group(group_predicates, instances, region_clip_score_fn):
    # group_predicates: list các predicate attribute cùng object type, khác object_ref
    # instances: list các box đã detect được của object type đó trong frame
    if len(instances) < len(group_predicates):
        return -math.inf   # không đủ instance để gán — infeasible
    cost_matrix = [[region_clip_score_fn(inst, pred) for inst in instances]
                   for pred in group_predicates]
    assignment, total_score = solve_assignment_max(cost_matrix)  # Hungarian hoặc greedy
    return total_score / len(group_predicates)
```

---

## 6. Combiner — hợp điểm theo đúng logic AND

```python
LAMBDA = 0.7   # tinh chỉnh trên validation set; gần 1 = nghiêm ngặt kiểu min, gần 0 = mềm kiểu mean

def combine_scores(calibrated_scores: list[float]) -> float:
    m = mean(calibrated_scores)
    mn = min(calibrated_scores)
    return (1 - LAMBDA) * m + LAMBDA * mn
```

- **`LAMBDA` nên khác nhau theo query_type:** KIS cần nghiêm ngặt (gần `min` thuần) vì chỉ có 1 đáp án đúng cần
  thoả *toàn bộ* điều kiện; AVS có thể mềm hơn một chút để không loại oan các kết quả "gần đúng" khỏi ranked list.
  → **tinh chỉnh `LAMBDA` bằng chính harness ở note metric** (grid search, đo Precision@1-có-điều-kiện theo từng giá trị).
- **Negation không đi qua combiner này** — xử lý riêng, **hard filter**:
  ```python
  for neg in parsed_query.negations:
      if calibrate(neg.type, evaluate(neg, frame)) > NEGATION_THRESHOLD:
          return -math.inf   # loại thẳng, không blend mềm
  ```
  Lý do: phủ định là ràng buộc cứng ("không liên quan" trong ví dụ hamburger tự nướng ở nhà, slide 13)
  — không nên để nó chỉ "kéo điểm xuống một chút" trong một phép trung bình.

---

## 7. Ví dụ chạy tay — "cầm kem VÀ ở biển" (slide 10-11)

Giả sử shortlist có 4 frame (đúng như minh hoạ slide 11), điểm calibrated giả định:

| Frame | scene(seaside) | exists(ice_cream) held-relation | combine (λ=0.7) |
|---|---|---|---|
| A (kem + biển, đúng) | 0.85 | 0.80 | `0.3×0.825 + 0.7×0.80 = 0.808` |
| B (chỉ có đường ven biển, không kem) | 0.80 | 0.05 | `0.3×0.425 + 0.7×0.05 = 0.163` |
| C (chỉ có biển đá, không kem) | 0.75 | 0.10 | `0.3×0.425 + 0.7×0.10 = 0.198` |
| D (biển, không rõ có kem) | 0.70 | 0.15 | `0.3×0.425 + 0.7×0.15 = 0.233` |

→ Frame A thắng áp đảo vì thoả *cả hai* điều kiện; B/C/D dù `scene(seaside)` cao ngang A nhưng bị `min` phạt nặng
vì thiếu điều kiện kem. **Đây chính là hành vi mà cosine similarity thuần trên cả câu không tự nhiên có được**
(vì "seaside" quá nổi bật trong câu, một CLIP thuần có thể xếp B/C/D gần A do đều "rất biển").

---

## 8. Ví dụ chạy tay — "túi tím VÀ túi trắng" (slide 39, số liệu đúng như slide)

3 vùng túi detect được, điểm attribute "purple purse" trên từng vùng (đúng số liệu slide): `0.233, 0.251, 0.224`.
Cần **2 predicate** trong group: `attribute(purse_A, purple)` và `attribute(purse_B, white)`, với `purse_A ≠ purse_B`.

Nếu chỉ có điểm "purple" mà chưa chạy "white" cho từng vùng, ta chưa đủ để giải assignment — cần chấm **ma trận đầy đủ**
(mỗi vùng × mỗi attribute cần), rồi mới gán. Đây là lý do binding **đắt hơn** attribute đơn: số lần gọi region-CLIP
= (số instance) × (số attribute trong group), không phải chỉ (số instance).

---

## 9. Khung code tổng thể (pseudocode)

```python
def score_frame(parsed_query: ParsedQuery, frame) -> float:
    scores = []

    # 1. Predicate đơn (không cần binding)
    for p in parsed_query.predicates:
        if is_in_binding_group(p, parsed_query):
            continue
        raw = evaluate_predicate(p, frame)      # route theo p.type
        scores.append(calibrate(p.type, raw))

    # 2. Binding groups
    for group in get_binding_groups(parsed_query):
        instances = detect_instances(frame, group.object_type)
        binding_score = resolve_binding_group(group.predicates, instances, region_clip_score)
        if binding_score == -math.inf:
            return -math.inf
        scores.append(binding_score)

    # 3. Negation — hard filter
    for neg in parsed_query.negations:
        if calibrate(neg.type, evaluate_predicate(neg, frame)) > NEGATION_THRESHOLD:
            return -math.inf

    # 4. Fallback baseline (luôn có mặt, phòng parse thiếu)
    scores.append(calibrate('scene', clip_whole_query_score(parsed_query.raw_query, frame)))

    return combine_scores(scores)


def rerank(parsed_query, shortlist):
    scored = [(f, score_frame(parsed_query, f)) for f in shortlist]
    return sorted(scored, key=lambda x: x[1], reverse=True)
```

---

## 10. Kiểu lỗi cần tự cảnh giác (nhắc lại có bổ sung thực dụng)

| Lỗi | Triệu chứng đo được (xem note metric) | Cách giảm nhẹ |
|---|---|---|
| Parse sai/thiếu | Precision@1-có-điều-kiện thấp đều ở mọi nhóm | luôn giữ fallback baseline `scene`-toàn-câu trong combine |
| Over-decomposition | thêm predicate không route được → nhiễu | chỉ giữ predicate có evaluator; log cảnh báo khi loại bỏ |
| Calibration lệch kênh | Precision thấp riêng ở nhóm có nhiều loại evaluator trộn (vd binding, relation) | chuyển sang relative calibration nếu learned calibration chưa đủ data |
| Binding infeasible bị chấm nhầm thành điểm thấp-nhưng-khác-âm-vô-cực | frame sai vẫn lọt top vì điểm binding "thấp" chứ không "loại hẳn" | đảm bảo trả `-inf` tường minh khi thiếu instance, không phải một số nhỏ |
| LAMBDA cố định cho mọi query_type | AVS bị "mất" các kết quả gần đúng do quá nghiêm | tách LAMBDA riêng theo query_type, tinh chỉnh bằng harness |

---

## 11. Điểm nối sang note KISC

Schema `Predicate` ở mục 2 sẽ được **dùng lại y nguyên** làm đơn vị cập nhật `belief-state` trong note KISC:
mỗi câu trả lời của người dùng trong hội thoại được parse thành đúng loại predicate này, rồi áp vào tập ứng viên
bằng cùng bộ `evaluate_predicate` + `calibrate` đã xây ở đây — **không xây hai hệ thống tách biệt**.
