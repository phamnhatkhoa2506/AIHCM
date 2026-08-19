# Đào sâu Lỗ hổng 2 & 5 — từ số 0

> Tiếp nối `brainstorm-retrieval-tu-ban-chat.md`. Hai lỗ hổng khó nhất về *tư duy*:
> #2 Compositional scoring và #5 Iterative refinement. Ngày: 2026-07-05.

---

# LỖ HỔNG 2 — COMPOSITIONAL SCORING

## 2.1. Vì sao một tích vô hướng *cấu trúc* không làm được AND (phần toán)

Mục tiêu huấn luyện CLIP là **contrastive**: kéo cặp (ảnh, caption) khớp lại gần, đẩy cặp lệch ra xa.
Caption huấn luyện đa số là mô tả **ngắn, tổng thể** ("a dog on a beach"). Hệ quả: text encoder học cách ánh xạ
một caption về *một điểm gần "trọng tâm" các khái niệm của nó*.

Điểm số `s(q,x) = <f_text(q), f_img(x)>` do đó hành xử xấp xỉ như **tổng có trọng số các tín hiệu "có mặt khái niệm"**:

```
s(q, x) ≈ Σ_c  w_c · presence_c(x) · relevance_c(q)
```

Đây là **cộng tính** → về bản chất là **OR / tổng mềm**, *không phải* AND. Bốn hệ quả trực tiếp:

- **Thiếu một điều kiện chỉ bị phạt một phần, không bị phủ quyết.** Frame khớp *cực mạnh* A có thể vượt frame khớp *vừa* cả A,B,C.
- **Thuộc tính trôi tự do khỏi object (binding failure):** "red cube and blue sphere" ≈ "blue cube and red sphere" — vì túi khái niệm `{red, cube, blue, sphere}` giống hệt.
- **Đếm hỏng:** "two dogs" ≈ "dogs".
- **Phủ định hỏng:** "a street with no cars" thường trả về đường *có* xe (vì "car" có trong text, kéo về ảnh có xe).

→ Đây **không phải bug tinh chỉnh được**; nó là *hệ quả của mục tiêu huấn luyện + việc pool về 1 vector*.
Đó chính là gốc rễ vì sao lỗ hổng 2 tồn tại.

## 2.2. Reframe: query là **vị từ (predicate)**, không phải điểm

Một query `Q` thực chất là một **công thức logic** trên cảnh:

```
Q = p_1 ∧ p_2 ∧ ... ∧ p_k     (có thể kèm ∨, ¬, đếm, quan hệ không gian/thời gian)
```

với mỗi `p_i` là một vị từ nguyên tử, ví dụ query "cầm kem VÀ ở biển":

```
exists(ice_cream)
relation(person, holding, ice_cream)   ← quan hệ
scene(seaside)
```

Hay "túi tím và túi trắng" (slide 39):
```
exists(purse) ∧ attribute(purse_A, purple) ∧ attribute(purse_B, white) ∧ purse_A ≠ purse_B   ← binding
```

Retrieval = tìm `x` tối đa/thoả `Q(x)`. Toàn bộ khó khăn: CLIP đánh giá `Q` như **một điểm**, làm sập cấu trúc.

## 2.3. Từ số 0: bộ chấm điểm tổ hợp cần làm 4 việc

**(A) Parse** `Q` thành cấu trúc (vị từ + toán tử logic).
**(B) Evaluate** từng vị từ nguyên tử lên biểu diễn (nối lỗ hổng 1: mỗi vị từ ở một *mức hạt* khác nhau).
**(C) Combine** điểm các vị từ theo đúng logic.
**(D) Xử lý** binding / quan hệ / đếm / phủ định — phần CLIP *về cấu trúc* không làm được.

### (A) Parse — phổ mức độ cấu trúc
1. **Không parse** (CLIP baseline): `Q` là chuỗi, 1 vector. Rẻ, zero cấu trúc.
2. **Trích khái niệm** (noun/adj/verb): có khái niệm, chưa có logic.
3. **LLM parse có cấu trúc** → xuất JSON: `{objects:[{name,attributes,count}], relations, scene, negations, time, place}`. **Điểm ngọt thực dụng 2026.**
4. **Parse ngữ nghĩa hình thức** (scene graph / logical form): giàu nhất nhưng giòn, cần matcher nói cùng ngôn ngữ.

> **Kỷ luật quan trọng:** độ sâu parse phải bị chặn bởi *thứ bạn thật sự evaluate được*.
> Một vị từ parse ra nhưng không có evaluator = vô dụng (thậm chí thêm nhiễu). **Đừng trích "vẻ mặt hoài niệm" nếu không có cách chấm nó.**

### (B) Evaluate — mỗi loại vị từ có "evaluator bản địa" riêng
| Vị từ | Evaluator bản địa |
|-------|-------------------|
| `scene(X)` | CLIP toàn ảnh vs "a photo of X" |
| `exists(O)` | nhãn detector (kênh Objects) HOẶC region-CLIP: max theo vùng của CLIP(vùng, "O") |
| `attribute(O, A)` | region-CLIP trên **box của O** vs "A O" — **phải chấm trên vùng, không phải cả ảnh**, nếu không binding sẽ rò |
| `count(O)=n` | đếm box từ detector (Objects). CLIP không làm được; detector làm được |
| `text_in_frame(S)` | kênh **OCR** khớp mờ |
| `speech(S)` | kênh **ASR** khớp |
| `relation(O1,R,O2)` | khó nhất: heuristic không gian từ box (trái/phải/trên/gần), HOẶC gọi **LVLM/VQA** ("người có đang cầm kem không?") |
| `¬P` | chấm `P` rồi **phạt/chặn** điểm cao |

> **Nguyên lý sâu (B):** *CLIP chỉ là evaluator bản địa cho vị từ mức-cảnh (gist).* Ép mọi thứ qua CLIP là
> **sai lầm gốc**. Hệ thống nên **định tuyến (route)** mỗi vị từ tới evaluator tốt nhất của nó.

### (C) Combine — toán tử cho AND
Cho điểm từng vị từ `s_1..s_k ∈ [0,1]` (**phải calibrate trước!**):

| Toán tử | Bản chất | Ghi chú |
|---------|----------|---------|
| sum / mean | OR-ish | chính là lỗi mặc định của CLIP |
| product `Πs_i` | AND-ish | một số 0 giết cả tích → giòn với 1 evaluator miss |
| **min** `min_i s_i` | AND nghiêm | "mắt xích yếu nhất"; frame chỉ tốt bằng điều kiện *tệ nhất*. **Hợp KIS** |
| soft-min / log-sum-exp | giữa min↔mean | khả vi, tinh chỉnh được |
| weighted / learned | AND có trọng số | vài điều kiện quan trọng hơn; học được nếu có data |

> **Bước quyết định thường bị bỏ — CALIBRATION.** Điểm CLIP, confidence detector, khớp OCR *khác thang/phân phối*.
> `min`/`product` trên điểm chưa calibrate là **vô nghĩa**. Đưa mỗi kênh về thang so sánh được (rank per-channel,
> z-score, hoặc map sang `P(vị từ đúng | điểm)` bằng tập calibrate nhỏ). Rất nhiều "multi-modal fusion" ngây thơ
> *chết chỉ vì bỏ bước này*.

Thực dụng: `min` thuần khá khắc nghiệt (1 evaluator dở là tụt cả true positive). Một **tổng hợp mềm phạt nặng min thấp**,
ví dụ `mean − λ·(1 − min)`, cân bằng robust ↔ nghiêm. Tinh chỉnh trên validation.

### (D) Binding / quan hệ / đếm / phủ định
Nguyên lý: **phân rã xuống tới mức có evaluator, rồi đẩy phần tổ hợp vào combiner tường minh hoặc model suy luận.**
- **Binding:** chấm attribute trên *vùng* (cần box). "túi tím ∧ túi trắng" → tìm vùng A = purse∧purple VÀ vùng B = purse∧white, A≠B → đây là **bài toán gán (assignment)** trên các vùng, *không phải* một điểm số. (Slide 39 chính là chấm "purple purse" lên từng vùng túi.)
- **Quan hệ:** heuristic không gian (rẻ, từ box) hoặc gọi LVLM verify (đắt, chỉ ở verify-stage).
- **Đếm:** detector, không phải CLIP.
- **Phủ định:** gate/penalty, hoặc **lọc cứng** (bỏ frame mà P có mặt mạnh).

## 2.4. Nó chạy ở đâu (nối lỗ hổng 4)

Chấm điểm tổ hợp **đắt** (nhiều evaluator, có thể gọi LVLM, thao tác vùng) → sống ở **verify stage** trên shortlist:
1. **Coarse:** CLIP trên *query tổng thể* → top-K (recall). CLIP-như-OR ổn ở đây vì bạn *muốn* recall, thừa còn hơn thiếu.
2. **Fine:** chấm lại tổ hợp trên K đó → precision. Phân rã, evaluate từng vị từ, combine min/weighted, verify binding/quan hệ/phủ định.

→ Giải nghịch lý "CLIP dở AND thì dùng làm gì": vì ở coarse ta *muốn* over-recall kiểu OR; kỷ luật AND áp *sau*, nơi kham nổi.

## 2.5. Điểm tinh tế: gửi sub-query nào vào coarse recall

Nếu đáp án cần `A∧B∧C` mà bạn chỉ coarse-search bằng query gộp, có thể *lọt* frame mà A nổi bật còn B,C tinh vi.
Chiến lược recall mạnh hơn: **coarse recall từng vị từ con riêng, rồi HỢP các pool ứng viên** (recall-by-decomposition),
sau đó áp AND ở verify.

> **Nguyên lý sâu:** *phân rã để recall (hợp các OR)* VÀ *phân rã để precision (verify cái AND)* —
> decomposition giúp ở **cả hai** stage, vì hai lý do ngược nhau.

## 2.6. Kiểu lỗi cần tự cảnh giác
- **Parse sai lan truyền** → sai evaluator. Giảm nhẹ: giữ *điểm CLIP tổng thể* như **một vị từ** trong hỗn hợp, để parse dở thì thoái hoá mượt về baseline.
- **Over-decomposition:** trích vị từ không chấm được = thêm nhiễu. Kỷ luật: chỉ trích thứ chấm được.
- **Lệch calibration** giữa các kênh.

---

# LỖ HỔNG 5 — ITERATIVE REFINEMENT / TƯƠNG TÁC LÀ THUẬT TOÁN

## 5.1. Lõi lý thuyết thông tin

Coi corpus có `N` item. Để cô lập đúng **một** item (KIS) cần `log2(N)` bit. `N` = 1 triệu keyframe → ~**20 bit**.
Một query mơ hồ ("tuần trước tôi gặp bạn cũ") có thể chỉ mang vài bit *phân biệt* → hậu nghiệm trải trên hàng nghìn ứng viên.

> **Không model nào chế ra ~15 bit còn thiếu; chúng phải được *cung cấp*.** One-shot retrieval **thiếu xác định
> vì ngân sách thông tin**, không phải vì model yếu.

→ Reframe retrieval thành **thu nhận thông tin tuần tự**: mỗi tương tác (filter người dùng áp, câu trả lời cho câu hỏi
làm rõ, cú click relevance-feedback) **bơm bit** và co tập ứng viên / làm sắc hậu nghiệm.

**Belief state.** Giữ một *phân phối (hoặc tập ứng viên có điểm)* trên các item theo mọi thứ đã biết. Mỗi lượt:
1. Trình bày gì đó (top ứng viên, hoặc một câu hỏi).
2. Nhận thông tin (ràng buộc / câu trả lời / feedback).
3. Cập nhật belief state (kiểu Bayes: reweight/lọc).
4. Chọn *hành động kế tiếp* để tối đa tiến độ kỳ vọng.

Đây đúng cấu trúc **active learning / thiết kế thí nghiệm tối ưu / POMDP**. **KISC** là bản tường minh nơi *hệ thống*
chọn câu hỏi; **UI người-lái** là bản nơi *người dùng* chọn ràng buộc kế. *Cùng toán, khác tác nhân chọn hành động.*

## 5.2. Bơm bit hiệu quả — thu hẹp theo độ chắc chắn

Không phải ràng buộc nào cũng bằng nhau. Hai trục:
- **Độ tin cậy** (ít nhiễu tới đâu): metadata thời gian *chính xác*; mô tả cảnh mờ thì *nhiễu*.
- **Độ chọn lọc / information gain** (co tập bao nhiêu): "thứ Ba tuần trước" có thể cắt 90%; "ngoài trời" cắt ~50%.

"Thu hẹp dần theo độ chắc chắn" (slide 40) = áp filter **tin cậy cao + chọn lọc cao + rẻ** *trước*. Vì sao trước:
chúng co không gian *trước khi* xài similarity CLIP nhiễu/đắt, và *không rủi ro làm rớt đáp án* (tin cậy cao).
Thứ tự đại khái: **thời gian → địa điểm → object có mặt (detector, khá tin) → mô tả cảnh mờ (kém tin, áp cuối, như *ranker* không phải filter cứng)**.

> **Nguyên lý sâu:** *ràng buộc cứng & tin cậy → dùng làm FILTER (phép tập hợp); tín hiệu mềm & nhiễu → dùng làm RANKER (điểm số).*
> Nhầm hai cái này — lấy điểm CLIP mờ làm ngưỡng cứng, hoặc lấy timestamp chính xác làm nudge mềm — là **lỗi thiết kế phổ biến**.
> **Lọc bằng thứ bạn chắc; xếp hạng bằng thứ bạn không chắc.**

## 5.3. Nước đi của hệ thống — chọn câu hỏi theo information gain (KISC)

Khi *hệ thống* chọn câu hỏi làm rõ kế tiếp (KISC), chọn câu nào? Câu **tối đa information gain kỳ vọng** = giảm entropy
belief state nhiều nhất. Trực giác: hỏi câu mà câu trả lời **chia tập ứng viên đều nhất (~đôi)** — chia 50/50 cho ~1 bit;
câu mà 99% ứng viên trả lời giống nhau cho ~0 bit.

Cụ thể: tập ứng viên `C`. Với câu hỏi `q` có các đáp án `{a_1..a_m}`, phân hoạch `C` theo đáp án dự đoán → cỡ `{|C_1|..|C_m|}`.
Information gain ≈ **entropy của phân hoạch đó**. Chọn `q` tối đa nó (chiến lược cây quyết định / "20 câu hỏi").
"Trong nhà hay ngoài trời?" là câu đầu tốt vì hay chia ~đôi. "Có mưa không?" là câu đầu tệ nếu 95% ứng viên không mưa.

> Biến KISC từ *cảm tính* → *tối ưu*: **tối đa bit trên mỗi câu hỏi.**
> Bản thực dụng: cluster tập ứng viên theo vài trục thuộc tính (địa điểm, trong/ngoài, số người, màu chủ đạo, thời điểm),
> rồi hỏi về **trục có entropy cụm cao nhất**.

**Điều kiện đi kèm:** cân thêm *chi phí cho người dùng* và *độ tin của câu trả lời*. Hỏi chi tiết họ *không nhớ* = phí một lượt.
Vậy: **tối đa (info gain kỳ vọng) với ràng buộc (người dùng trả lời được đáng tin).** Info gain lý thuyết khổng lồ mà người không trả lời được = 0 gain thật.

## 5.4. Relevance feedback — bit từ "cái này gần đúng"

Khi người dùng nói "kết quả #7 gần đúng nhưng chưa phải", đó là một **hướng**, không phải ràng buộc cứng. Các cách kinh điển:
- **Dời điểm query (kiểu Rocchio):** `q' = q + α·liked − β·disliked` (dời embedding về mẫu thích, xa mẫu ghét).
- **"More like this" ảnh↔ảnh:** search theo tương đồng hình ảnh với frame được thích.
- **Reweight vị từ:** suy ra điều kiện nào mẫu-thích thoả rồi tăng trọng số.

> **Nguyên lý sâu:** *feedback dương/âm là **thông tin gradient** trên embedding query / trọng số vị từ.*
> Người dùng đang **gradient-descent hộ bạn**, rẻ, trong không gian ngữ nghĩa.

**Bonus riêng cho video/lifelog:** frame được thích có **hàng xóm thời gian** — đáp án thật thường cách vài giây.
Nên "**browse quanh hit này trên timeline**" là hành động *cực kỳ giá trị, gần như miễn phí*. (Đây là lý do UI thắng
VBS/AIC *luôn* có browse thời gian.)

## 5.5. Query expansion — nâng RECALL, không phải precision

Mặt còn lại: đôi khi vấn đề không phải quá nhiều ứng viên, mà là *đáp án đúng không nổi lên* (lệch từ vựng/cách diễn đạt,
hoặc kịch bản hình ảnh đúng khác với câu chữ đơn lẻ của bạn). LLM query expansion sinh nhiều cách diễn đạt / kịch bản
hình ảnh cụ thể của *cùng ý định* → hợp recall của chúng. Việc này nâng **trần (recall)** để refinement có cái mà hội tụ về.

> Đây là **cần recall**; còn thu-hẹp-theo-độ-chắc-chắn và feedback là **cần precision**.
> Vòng refinement cần *cả hai*: **expand** để chắc đáp án *nằm trong* pool, **narrow** để *tìm ra* nó.

## 5.6. Tổng hợp — vòng lặp như một thuật toán

Cả hệ là một **vòng điều khiển trên belief state**:
- **Cần recall** (expansion, hợp-các-decomposition): đảm bảo đáp án nằm trong pool.
- **Cần precision** (filter tin cậy trước, rồi ranker mềm): co dần về nó.
- **Thu nhận thông tin** (hệ hỏi theo info-gain, hoặc feedback người dùng như gradient): bơm bit còn thiếu.
- **Khai thác cấu trúc** (hàng xóm thời gian): bit gần-miễn-phí.

> **Meta-nguyên lý nối cả cuộc thi:** với retrieval có human-in-the-loop, thứ bạn tối ưu **không phải "độ chính xác một query"**
> mà là **"số lượt tương tác (hoặc số giây) kỳ vọng để chạm đích".**

Reframe này đổi mọi thứ:
- Một ranker *hơi kém* nhưng nổi lên ứng viên **đa dạng & giàu thông tin** (để cú click kế của người dùng *tối đa thông tin*)
  có thể **thắng** một ranker tốt hơn nhưng nổi top-5 gần trùng nhau (phí click của người dùng).
  → **Đa dạng kết quả** (vd MMR — maximal marginal relevance, để tập ứng viên *trải* trên belief state thay vì dồn 1 cách hiểu),
  **khả năng browse**, và **vòng feedback nhanh** trở thành **mục tiêu hạng nhất** — thứ mà tư duy "maximize top-1 similarity" bỏ qua.
- **Độ trễ UI là một phần của metric:** mỗi vòng lặp tốn thời gian. Search 2 giây cho làm 30 vòng **thắng** search 20 giây chỉ làm 3 vòng.

---

## Nối hai lỗ hổng với nhau
- **Lỗ hổng 2** cho bạn *một lượt chấm điểm sắc hơn* (đánh giá đúng công thức logic của query).
- **Lỗ hổng 5** thừa nhận *một lượt là không đủ* và biến bài toán thành *chuỗi lượt rẻ*.
- Chúng bổ trợ: decomposition ở #2 chính là thứ cho phép **reweight vị từ** khi feedback (#5), và cho phép
  **hỏi về một vị từ cụ thể** trong KISC. Cùng một "cấu trúc query" nuôi cả hai.
