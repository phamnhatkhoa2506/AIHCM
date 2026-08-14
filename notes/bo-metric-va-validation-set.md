# Bộ Metric & Validation Set — tự chấm hệ thống trước khi thi

> Nối tiếp `brainstorm-retrieval-tu-ban-chat.md`, `lo-hong-2-va-5-chi-tiet.md`, `ban-do-du-lieu-pipeline.md`.
> Mục tiêu: có cách **đo tại từng tầng pipeline** (không chỉ end-to-end) để biết chính xác nên đổ công sức
> vào đâu — thực thi hoá nguyên lý "thiết kế ngược từ metric" (lỗ hổng 4) bằng số liệu thật.
> Ngày: 2026-07-06.

---

## 0. Vì sao *không thể* chỉ đo end-to-end

Nếu chỉ đo "Top-1 accuracy cuối cùng = 40%", bạn biết hệ thống *có vấn đề* nhưng không biết **vấn đề ở đâu**:
có thể coarse recall đã làm rớt đáp án từ vòng đầu (không tầng nào cứu được), hoặc đáp án nằm trong top-100 nhưng
fine verify xếp hạng sai, hoặc temporal logic đảo lộn thứ tự, hoặc UI không cho người dùng đủ vòng lặp để tìm ra.

Bốn nguyên nhân này **cần bốn cách sửa hoàn toàn khác nhau**. Đo end-to-end không tách được chúng.

> **Nguyên lý dựng bộ metric:** với mỗi tầng trong pipeline (Filter → Coarse recall → Fine verify → Temporal → Interaction),
> cần **một metric riêng đo đúng việc tầng đó phải làm**, cộng với **một metric end-to-end** đo kết quả cuối cùng theo đúng
> cách cuộc thi chấm. Không có cái đầu, bạn mù về nguyên nhân. Không có cái sau, bạn không biết tổng thể có ổn không.

---

## 1. Metric end-to-end — theo đúng cách cuộc thi thật sự chấm

### KIS (Known-Item Search)
Cuộc thi kiểu VBS/AIC thường chấm KIS bằng **hạng của đáp án đúng + thời gian tìm ra**, không chỉ đúng/sai đơn thuần.
Ba con số nên theo dõi song song:

| Metric | Công thức / cách đo | Ý nghĩa |
|---|---|---|
| **Recall@K** (K=1,5,10,100) | % số query mà đáp án đúng nằm trong top-K kết quả cuối | Càng K nhỏ càng khắt khe; Top-1 là con số ăn điểm thật |
| **MRR** (Mean Reciprocal Rank) | trung bình của `1/rank_đáp_án_đúng` trên toàn bộ query | Phạt mượt theo hạng, tốt hơn Recall@K khi so sánh hai hệ thống gần nhau |
| **Time-to-find** | số giây/số vòng tương tác đến khi người dùng chốt đúng | Mô phỏng đúng áp lực thời gian thật của cuộc thi (VBS chấm cả điểm theo thời gian) |

> Gợi ý công thức điểm mô phỏng kiểu VBS (không bắt buộc, nhưng giúp bạn có 1 con số tổng để so sánh hệ thống theo thời gian):
> `score = max(0, 100 − α·rank_penalty − β·time_penalty)` nếu tìm đúng trong giới hạn thời gian, ngược lại `0`.
> Không cần chính xác công thức BTC dùng — mục đích là **có một con số phạt cả hạng lẫn thời gian** để so sánh giữa các phiên bản hệ thống của bạn.

### AVS (Ad-hoc Video Search)
Đây là bài toán ranking trên toàn bộ tập liên quan, không phải 1 đáp án:

| Metric | Công thức / cách đo | Ý nghĩa |
|---|---|---|
| **AP** (Average Precision) mỗi query, rồi **mAP** trung bình | diện tích dưới đường precision-recall của danh sách trả về | Chuẩn TRECVID AVS — thưởng việc đẩy các kết quả đúng lên đầu |
| **nDCG@K** | có tính đến thứ hạng và mức độ liên quan (nếu bạn gán độ liên quan nhiều mức thay vì nhị phân) | Phù hợp khi một số kết quả "đúng nhưng yếu" (khớp một phần điều kiện) |
| **Recall@K toàn cục** | % tổng số item liên quan thật sự được tìm thấy trong K kết quả đầu | Đo trần recall của hệ thống cho loại truy vấn rộng |

### VQA
| Metric | Công thức / cách đo | Ý nghĩa |
|---|---|---|
| **Answer accuracy** | so khớp câu trả lời với đáp án đúng — dùng **LLM-as-judge** để so khớp ngữ nghĩa (không phải exact string match, vì câu trả lời tự nhiên có nhiều cách diễn đạt đúng) | Metric cuối cùng |
| **Localization accuracy** (metric phụ, chẩn đoán) | % câu hỏi mà đoạn clip được cắt ra *có chứa* đủ thông tin trả lời đúng | Tách lỗi "định vị sai đoạn" khỏi lỗi "LVLM suy luận sai dù đã có đúng đoạn" |

> Tách hai dòng trên **quan trọng**: nếu answer accuracy thấp nhưng localization accuracy cao → vấn đề nằm ở LVLM suy luận,
> không phải retrieval. Nếu cả hai đều thấp → vấn đề nằm ở tầng định vị (Coarse recall + Temporal), sửa LVLM vô ích.

### KISC (Conversational KIS)
| Metric | Công thức / cách đo | Ý nghĩa |
|---|---|---|
| **Success rate trong N lượt** | % phiên hội thoại tìm ra đáp án đúng trong giới hạn lượt cho phép | Metric cuối cùng, mô phỏng luật chơi |
| **Turn efficiency** | số lượt trung bình cần để thành công (chỉ tính các phiên thành công) | Đo hiệu quả hỏi — càng ít lượt càng tốt |
| **Information gain / lượt** (chẩn đoán) | tốc độ co tập ứng viên: `log2(|C_trước|) − log2(|C_sau|)` mỗi lượt | Đo trực tiếp chất lượng câu hỏi được chọn (lỗ hổng 5.3) — tách lỗi "chọn câu hỏi dở" khỏi lỗi "retrieval nền yếu" |

---

## 2. Metric per-stage (chẩn đoán) — đây là phần thường bị bỏ qua nhưng quan trọng nhất

### Tầng 1 — Filter (Metadata)
**Filter recall** = % query mà đáp án đúng **không bị loại oan** bởi bộ lọc cứng (thời gian/kênh/địa điểm).
> **Ngưỡng phải đạt: gần 100%.** Đây là filter *cứng* — nếu nó lỡ tay loại đáp án đúng, không tầng nào phía sau cứu được
> (đúng nguyên lý "lọc bằng thứ bạn chắc" — nếu bạn lọc sai, chứng tỏ độ tin cậy giả định của trục đó không đúng như nghĩ).
> Nếu filter recall < 100%, việc cần làm là **nới lỏng điều kiện lọc** hoặc chuyển trục đó từ filter sang ranker (soft).

### Tầng 2 — Coarse recall (CLIP ANN)
**Recall@K của coarse recall** (K = kích thước shortlist đưa sang fine verify, ví dụ K=100).
> Đây chính là **cái trần** đã nói ở lỗ hổng 1 & 4: nếu đáp án đúng không nằm trong top-K ở tầng này, mọi nỗ lực fine verify
> phía sau đều vô nghĩa cho query đó. **Đây là con số quan trọng nhất để theo dõi riêng** — vì end-to-end Top-1 thấp có thể
> chỉ vì cái trần này thấp, chứ không phải vì fine verify dở.
>
> Cách chẩn đoán: nếu Recall@100 ở tầng này đã thấp (ví dụ 70%) trong khi end-to-end Top-1 chỉ 40% — thì **30 điểm phần trăm
> đầu là do coarse recall**, **phần còn lại (70% → 40%, tức 30 điểm nữa) là do fine verify xếp hạng dở**. Hai vấn đề, hai cách sửa khác hẳn.

### Tầng 3 — Fine verify / Compositional scoring
**Precision@1 có điều kiện** = trong các query mà đáp án đúng **đã có mặt** trong shortlist (từ Tầng 2), % mà fine verify
xếp nó lên đúng vị trí Top-1.
> Đây là metric "sạch" của riêng compositional scoring — loại bỏ nhiễu từ lỗi coarse recall. Nếu con số này thấp,
> vấn đề nằm ở cách parse query / chọn evaluator / combine operator (lỗ hổng 2), không phải ở embedding.

**Bổ sung khi có ground truth cấu trúc (structured):** với các query đã được gán nhãn *loại lỗi tiềm ẩn* (xem mục 3.3),
đo riêng theo từng loại: precision cho query có **binding** (thuộc tính gắn đối tượng), có **đếm**, có **phủ định**, có **quan hệ**.
→ biết chính xác evaluator/combine nào đang yếu.

### Tầng 4 — Temporal
**Temporal accuracy** = trong các query có yêu cầu thứ tự trước/sau tường minh (tập con nhỏ, gán nhãn riêng),
% mà hệ thống trả về đúng thứ tự, không bị đảo.
**Duplicate rate cho AVS** = % kết quả trong top-K bị trùng shot/scene với một kết quả xếp hạng cao hơn — đo hiệu quả
của bước gom shot/scene (dedup); tỷ lệ cao nghĩa là đang lãng phí thứ hạng vào các bản sao gần giống nhau.

### Tầng 5 — Interaction
Ngoài info-gain/lượt đã nêu ở KISC, với UI người-lái nên đo:
**Diversity@K** (chẩn đoán) = độ đa dạng ngữ nghĩa của K kết quả hiển thị đầu tiên (ví dụ trung bình khoảng cách cặp đôi
giữa các embedding được hiển thị). Diversity thấp = hệ thống đang chỉ dội một "cách hiểu" của query, lãng phí cơ hội
cho người dùng cung cấp feedback phân biệt (đúng nguyên lý MMR ở lỗ hổng 5.6).

### Bảng tóm tắt — mỗi tầng đo gì, ngưỡng kỳ vọng

| Tầng | Metric chẩn đoán | Vì sao chỉ riêng tầng này |
|---|---|---|
| Filter | Filter recall | phải ~100%, nếu không đang lọc sai |
| Coarse recall | Recall@K (K=shortlist size) | **cái trần** — quyết định giới hạn trên của mọi thứ sau |
| Fine verify | Precision@1 có điều kiện (đã lọt shortlist) | đo sạch compositional scoring, tách khỏi lỗi tầng trước |
| Temporal | Temporal accuracy, Duplicate rate | đo riêng phần thứ tự & gom trùng |
| Interaction | Info-gain/lượt, Diversity@K, Turn efficiency | đo chất lượng vòng lặp, không phải 1 lần truy vấn |

---

## 3. Dựng Validation Set — vì chưa có ground truth chính thức

BTC chỉ công bố đề thi thật vào ngày thi. Trước đó bạn **phải tự tạo** một tập câu hỏi có đáp án đã biết,
từ chính dữ liệu được phát (Videos/Keyframes/Metadata) để tự chấm.

### 3.1. Chiến lược tạo query có đáp án biết trước (reverse-engineering)

**Từ Metadata (nhanh nhất, số lượng lớn):**
1. Chọn ngẫu nhiên N video từ tập được phát.
2. Đọc `title`/`description` — **tự viết lại** thành một câu mô tả kiểu KIS ("tìm khoảnh khắc...") *bằng lời của mình*,
   không copy nguyên văn title (để giả lập đúng khoảng cách ngôn ngữ tự nhiên ↔ nội dung thật — nếu copy nguyên văn title,
   bạn đang test khả năng khớp text-text chứ không phải text-hình ảnh).
3. Ground truth = chính video đó + một khung thời gian cụ thể (tự chọn 1 keyframe làm mốc "đúng").

**Từ xem trực tiếp Video/Keyframes (chậm hơn nhưng chất lượng cao, mô phỏng đúng KIS thật):**
1. Random sample một số video, xem lướt qua.
2. Chọn 1 khoảnh khắc bất kỳ (ưu tiên khoảnh khắc có **tổ hợp điều kiện** — 2-3 chi tiết cùng xảy ra, giống ví dụ
   "kem + biển" ở slide 10) → viết mô tả **như đang nhớ lại**, không nhìn hình mà mô tả máy móc.
3. Ground truth = video + timestamp/keyframe chính xác.

**Từ Objects (tạo AVS quy mô lớn, bán tự động):**
1. Chọn 2-3 nhãn object cùng xuất hiện trong 1 frame (ví dụ `Person + Bicycle`) → sinh query "tìm cảnh có người đi xe đạp".
2. Ground truth = tập hợp mọi frame thoả điều kiện label (có thể tính tự động từ dữ liệu Objects đã có, không cần gán tay).
3. Có thể sinh **hàng trăm query AVS** theo cách này gần như miễn phí — vì Objects đã cho sẵn nhãn.
4. **Cảnh báo:** cách này tạo query "dễ" (đúng bản chất công việc Objects làm tốt) — cần bù bằng nhóm query khó ở 3.3.

### 3.2. Quy tắc tách biệt vai trò — tránh tự lừa mình

> **Nguyên lý quan trọng:** người **viết query** và người **cải tiến hệ thống** nên là hai người khác nhau
> (hoặc ít nhất, viết query *trước khi* biết hệ thống sẽ được sửa thế nào). Nếu người code tự viết query,
> họ sẽ vô thức chọn câu mà họ *biết hệ thống xử lý được* — validation set sẽ đánh giá quá cao khả năng thật.

Nếu đội chỉ có 1-2 người: viết toàn bộ query **một lần, chốt lại**, tránh sửa query sau khi thấy hệ thống sai
(dễ dẫn tới "sửa đề cho khớp đáp án" trong vô thức).

### 3.3. Nhóm khó — bài test riêng cho từng lỗ hổng

Tập dễ (từ Metadata/Objects ở trên) không đủ để phát hiện các lỗi tinh vi đã phân tích. Cần chủ đích tạo các nhóm nhỏ:

| Nhóm | Cách tạo | Test cho lỗ hổng nào |
|---|---|---|
| **Object nhỏ / chi tiết hiếm** | chọn khoảnh khắc có vật thể tí xíu không nằm trong 600 category của Objects | #1 Representation |
| **Compositional AND** | mô tả bắt buộc ≥2 điều kiện độc lập cùng xảy ra (như "kem + biển"), kèm ít nhất 1 "near-miss distractor" (chỉ thoả 1 điều kiện) trong tập | #2 Compositional scoring |
| **Binding** | mô tả 2 vật cùng loại khác thuộc tính ("túi tím" và "túi trắng") | #2 (binding cụ thể) |
| **Phủ định** | mô tả có "không", "trừ", "ngoại trừ" | #2 (negation) |
| **Temporal order** | mô tả tường minh trước/sau ("...trước khi...", "...rồi mới...") | #3 Temporal |
| **Mơ hồ cố ý** | mô tả thiếu thông tin, chỉ đủ để test vòng lặp làm rõ (KISC) | #5 Interaction |
| **Text-in-frame** | khoảnh khắc mà chi tiết phân biệt chính là chữ hiển thị trên màn hình (banner/số liệu bản tin) | kênh OCR (bản đồ dữ liệu) |

**Số lượng gợi ý:** tối thiểu ~20-30 query/nhóm để có ý nghĩa thống kê tối thiểu (đủ để thấy chênh lệch rõ giữa
các phiên bản hệ thống, dù chưa "chuẩn khoa học"). Nhóm dễ (Metadata/Objects) có thể để 50-100+ vì tạo gần như miễn phí.

### 3.4. Định dạng lưu trữ (thực dụng)

Một dòng JSONL cho mỗi query, tối thiểu các trường:
```
{
  "query_id": "...",
  "query_text": "...",
  "query_type": "KIS_textual | KIS_video | AVS | VQA | KISC",
  "difficulty_tag": "easy | small_object | compositional | binding | negation | temporal | ambiguous | ocr",
  "ground_truth_video_id": "...",
  "ground_truth_timestamp_or_frame": "...",
  "ground_truth_relevant_set": ["...", "..."]   // dùng cho AVS, danh sách tất cả các frame/video liên quan
}
```
Có trường `difficulty_tag` để lúc chấm có thể **cắt lát theo nhóm** — đây chính là thứ cho phép mục 2 (per-stage/per-loại-lỗi)
hoạt động được, không chỉ có một con số tổng.

---

## 4. Ràng buộc thời gian/latency — vì cuộc thi có đồng hồ

Ngoài độ chính xác, cần log **thời gian chạy mỗi tầng** (coarse recall mất bao lâu, fine verify mất bao lâu cho K ứng viên,
LVLM verify mất bao lâu...). Vì:
- KIS thường bị giới hạn thời gian cứng mỗi câu (kiểu VBS) → hệ thống đúng nhưng chậm vẫn thua.
- Có sự đánh đổi rõ ràng: K càng lớn ở coarse recall (an toàn cho recall) thì fine verify càng chậm (chạy trên nhiều ứng viên hơn).

**Việc cần làm:** đo và ghi lại `latency` theo từng tầng như một cột riêng trong kết quả eval, không chỉ độ chính xác.
Khi tối ưu, luôn nhìn cặp (metric chính xác, latency) cùng lúc — cải thiện Recall@100 bằng cách tăng K từ 100 lên 1000
có thể "ăn gian" điểm nhưng phá luật chơi thời gian thật.

---

## 5. Harness đánh giá — chạy lại mỗi khi đổi bất kỳ thành phần nào

**Nguyên lý:** trực giác về "cải tiến này chắc sẽ tốt hơn" **thường sai** — nhất là với các quyết định như đổi combine
operator, đổi ngưỡng calibration, thêm 1 evaluator mới. Phải có một script **chạy toàn bộ validation set và in ra bảng
metric theo `difficulty_tag`** mỗi khi có thay đổi, giống một bộ regression test.

Luồng harness (không cần phức tạp, một script Python đơn giản là đủ):
1. Đọc file JSONL validation set.
2. Với mỗi query: chạy pipeline, log lại **kết quả trung gian ở mỗi tầng** (không chỉ kết quả cuối) — tức là log cả
   "top-K sau coarse recall" lẫn "top-K sau fine verify" cho cùng 1 query, để tính được cả metric end-to-end lẫn per-stage.
3. Tính toán các metric ở mục 1 và 2, in bảng theo `query_type` × `difficulty_tag`.
4. So sánh với lần chạy trước (giữ lịch sử) → biết thay đổi vừa rồi **thật sự** cải thiện cái gì, làm xấu đi cái gì.

> **Điểm mấu chốt của bước 2 (log kết quả trung gian):** nếu chỉ log kết quả cuối cùng, bạn *mất khả năng* tính
> Recall@K của riêng coarse recall — tức là mất luôn khả năng chẩn đoán "cái trần" đã nói ở mục 2. Ngay từ đầu,
> hãy thiết kế code sao cho mỗi tầng trả về (và log lại) danh sách ứng viên + điểm số của chính nó, đừng chỉ trả về câu trả lời cuối.

---

## 6. Tổng kết — bộ metric này đóng vòng lặp với 5 lỗ hổng thế nào

Toàn bộ mục đích của bộ metric là biến "nghi ngờ" thành "bằng chứng":

- Recall@K của Coarse recall thấp → xác nhận vấn đề ở **lỗ hổng 1** (representation) hoặc tham số ANN, không phải fine verify.
- Precision@1 có điều kiện thấp, đặc biệt lệch nặng ở nhóm `compositional`/`binding`/`negation` → xác nhận vấn đề ở **lỗ hổng 2**.
- Temporal accuracy thấp ở nhóm `temporal` riêng, trong khi các nhóm khác ổn → xác nhận vấn đề ở **lỗ hổng 3**, khoanh vùng rõ không cần đụng tới representation hay compositional scoring.
- Latency vượt ngưỡng ở một tầng cụ thể → xác nhận cần tối ưu **lỗ hổng 4** (cắt K, hoặc chuyển việc nặng sang ít query hơn).
- Turn efficiency/info-gain thấp ở nhóm `ambiguous` (KISC) dù retrieval nền tốt → xác nhận vấn đề ở **lỗ hổng 5** (chọn câu hỏi/UI), không phải ở model.

> **Không có bộ metric cắt lát theo tầng + theo loại khó, mọi buổi "brainstorm cải tiến" sẽ dựa trên cảm giác.
> Có nó, mỗi thay đổi trở thành một thí nghiệm có thể đo được — đúng tinh thần "thiết kế ngược từ metric" đã đặt ra từ lỗ hổng 4.**
