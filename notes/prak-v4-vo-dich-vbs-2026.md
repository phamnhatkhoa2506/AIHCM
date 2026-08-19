# PraK V4 — Vô địch VBS 2026 (bản đầy đủ, từ chính bài báo)

> Bản sửa lại sau khi có full-text từ chính user (Jäckl, Verner, Stroh, Kloda, Nagy, Deussen, Keim, Lokoč —
> Univ. Konstanz + Charles University, MMM 2026 proceedings). **Một số điều mình suy đoán ở bản trước (dựa
> trên tóm tắt tìm kiếm) là sai hoặc thiếu — đính chính rõ ràng ở mục 3.** Nối tiếp
> `doi-chieu-thuc-te-cac-doi-vo-dich.md`, `phac-thao-compositional-scoring.md`,
> `lop-tuong-tac-thuc-dung-kiem-soat-duoc.md`. Ngày: 2026-07-07.

---

## 0. Bối cảnh cuộc thi — một biến số mới chưa từng bàn: dữ liệu đồng nhất (homogeneous)

VBS 2026 dùng 3 bộ dữ liệu: **V3C** (cảnh đời thường, đa dạng), **MVK** (Marine Video Kit — quay dưới nước),
**LHE** (video phẫu thuật nội soi phụ khoa — GynSurg). Điểm mấu chốt bài báo nêu ngay từ đầu:

> Text query CLIP hoạt động tốt trên V3C (cảnh đa dạng, phản hồi tốt với text prompt), nhưng **MVK và LHE có
> tính đồng nhất cao (homogeneous)** — mọi cảnh trông khá giống nhau (toàn nước + cá, hoặc toàn dụng cụ phẫu
> thuật) — khiến CLIP text-query **rất kém hiệu quả**, buộc người dùng phải **duyệt tay diện rộng**, tăng tải
> nhận thức và tăng rủi ro bỏ sót.

→ Đây là một khía cạnh **mới, chưa từng đưa vào khung 5 lỗ hổng**: semantic gap (gốc rễ #1) biểu hiện khác
nhau tuỳ **độ đa dạng thị giác của corpus**. Trên dữ liệu HTV/tin tức (đa dạng: nhiều chủ đề, bối cảnh), CLIP
text-query nhìn chung ổn; nhưng nếu trong dữ liệu AIC có phân đoạn nào lặp lại khung cảnh nhàm chán (vd nhiều
đoạn news-desk giống hệt nhau), đúng lúc đó semantic gap sẽ trầm trọng hơn hẳn — **đáng để kiểm tra sớm** bằng
cách nhìn qua độ đa dạng thị giác của keyframe corpus AIC (cluster CLIP embedding, xem có mảng nào quá đặc/đồng
nhất không) trước khi giả định "CLIP baseline luôn đủ tốt".

---

## 1. Kiến trúc nền — chính xác lại 4 phần

PraK kế thừa & mở rộng ý tưởng từ VIRET, SOMHunter, CVHunter — gồm 4 phần:

1. **Tiền xử lý → keyframe features:** phát hiện shot bằng **TransNet V2**, trích keyframe bằng
   **agglomerative clustering** (khác cách MERVIN làm — MERVIN lấy 3 điểm cố định 0.15/0.5/0.85 mỗi shot;
   PraK dùng phân cụm để chọn keyframe đại diện — thêm 1 lựa chọn cụ thể nữa cho việc "nên trích keyframe
   thế nào" ngoài I-frame gốc BTC phát). Mỗi keyframe kèm **CLIP feature đã fine-tune riêng cho image-image
   search** (không phải CLIP gốc — trích dẫn Schall et al., SISAP 2024, "Optimizing CLIP models for image
   retrieval with maintained joint-embedding alignment") + **cùng loại feature cho 5 vùng lưới tĩnh** (4 góc +
   giữa) dùng cho MVK/LHE localized query.
2. **Data service (stateless, in-memory):** trả lời truy vấn (text query...).
3. **Backend (stateful):** quản lý trạng thái phiên (candidate hiện tại), gọi data service khi cần.
4. **Frontend:** lưới 4 cột xếp hạng, tuỳ chọn nhóm theo video (tối đa 1 keyframe/video), hover xem trước
   video gốc/toàn bộ keyframe. Panel điều khiển hỗ trợ text/temporal/localized-text query (suffix vị trí, vd
   `br` = bottom-right). **Bayesian Updates**: sắp xếp lại tập kết quả hiện tại dựa trên các ví dụ dương đã
   click — keyframe tương tự trồi lên trên.

---

## 2. Năm cải tiến của V4 — đầy đủ (bản trước chỉ có 1/5)

Động lực (từ log analysis + phỏng vấn chuyên gia): **PraK yếu ở AVS**, người dùng **tốn quá nhiều thời gian
tìm chi tiết trong 1 video đã xác định đúng**, và V3 thiếu khả năng **kết hợp nhiều vùng cùng lúc**.

### 2.1 Spatial Conjunction of Localized Queries — tính năng đầu bảng, có nghiên cứu thực nghiệm hẳn hoi

**V3** đã có: suffix vị trí trên lưới tĩnh 5 vùng (theo cách SOMHunter làm). Nghiên cứu trước đó (Jäckl et al.)
so sánh vùng **tĩnh** (static grid) với vùng **động** (dựa trên object detector) trên MVK — vẽ 1 box + text +
khoảng cách hình học cho kết quả tốt nhất — nhưng nghiên cứu đó **chỉ giới hạn ở ảnh đơn (keyframe)**, chưa
thử trên shot (đoạn video ngắn), và chưa thử **kết hợp nhiều vùng cùng lúc**.

**V4 mở rộng:** áp dụng cho **shot** (không chỉ 1 keyframe) + cho phép **truy vấn liên hợp nhiều vùng** (nhiều
box khớp đồng thời trong 1 shot).

**Phương pháp thực nghiệm (nghiêm túc, đáng học theo):**
- 5 annotator có kinh nghiệm gán nhãn ngẫu nhiên các shot MVK (trung bình 12s, khoảng 8-16s), bỏ qua shot mà
  sub-region không liên quan.
- Mỗi shot: 1 câu query toàn ảnh + ít nhất 1 câu query cục bộ (box + text).
- Tổng cộng **212 annotation**, so sánh 5 biến thể:
  1. **Whole-Image** — baseline, query toàn ảnh.
  2. **Static** — khớp bounding box với vùng tĩnh tốt nhất theo IoU, cộng dồn khoảng cách qua nhiều box gán nhãn.
  3. **Dynamic** — giống Static nhưng dùng vùng động (object detector: **Grounded-SAM**).
  4. **Dynamic with CD** (centroid distance) — tính khoảng cách giữa **mọi** sub-region đã trích với text query
     cục bộ; đồng thời tính khoảng cách Euclid L2 giữa **tâm** box gán nhãn và tâm box detector trích được; cả
     hai khoảng cách được **chuẩn hoá (z-score theo mean/std tính trước)** rồi **cộng lại**; sau đó "phân tầng"
     (stratify) về khoảng cách theo từng keyframe.
  5. **Dynamic with IoU** — giống CD nhưng dùng IoU thay vì khoảng cách tâm.

**Kết quả (Recall@K trung bình trên toàn bộ annotation liên hợp):**

| Phương pháp | R@1 | R@10 | R@100 | R@1000 |
|---|---|---|---|---|
| Whole-Image | 0.04 | 0.19 | 0.46 | 0.73 |
| Static | 0.07 | 0.28 | 0.59 | 0.83 |
| Dynamic | 0.04 | 0.21 | 0.43 | 0.74 |
| Dynamic + CD | **0.13** | **0.31** | 0.56 | 0.81 |
| Dynamic + IoU | 0.11 | 0.29 | **0.59** | **0.83** |

**Phát hiện quan trọng (ngược lại kỳ vọng ban đầu):**
- Static và cả 2 biến thể "Dynamic + fusion khoảng cách" **cải thiện đáng kể** so với Whole-Image (R@1 tăng
  từ 0.04 lên tới 0.13 — hơn 3 lần).
- **Dynamic đơn thuần (không fusion) lại KHÔNG cải thiện** — trái ngược với nghiên cứu trước đó của chính nhóm
  tác giả. Lý do họ tự phân tích:
  1. Bounding box của annotator **kém chính xác hơn** (annotator vẽ trên khung riêng, không vẽ trực tiếp lên
     keyframe).
  2. Annotator được yêu cầu mô tả **"vùng đáng chú ý"** chứ không bắt buộc là **"object"** — nhiều query mô tả
     kiểu "một đàn cá" (school of fish) không được detector nhận ra như **1 vùng liền mạch**, mà tách thành
     **nhiều object riêng lẻ** → mismatch giữa ý định người dùng và output của detector.
- Hiệu năng phụ thuộc mạnh vào **loại annotation** (không đồng đều) → PraK V4 quyết định **giữ cả static lẫn
  dynamic** làm 2 lựa chọn song song, không chọn 1 cái duy nhất.

**Về lợi ích của việc liên hợp (conjunction) nhiều vùng (bảng riêng, dùng Static):**

| Số box | Biến thể | R@1 | R@10 | R@100 | R@1000 |
|---|---|---|---|---|---|
| 1 box | Full/Random | 0.06 | 0.25 | 0.68 | 0.87 |
| 2 box | Full (dùng cả 2) | 0.10 | 0.32 | 0.62 | 0.86 |
| 2 box | Random (chỉ chọn 1/2, TB 5 lần) | 0.05±0.02 | 0.20±0.03 | 0.44±0.03 | 0.72±0.03 |
| 3 box | Full (dùng cả 3) | 0.05 | 0.26 | 0.32 | 0.68 |
| 3 box | Random (chỉ chọn 1/3) | 0.01±0.02 | 0.04±0.04 | 0.11±0.06 | 0.55±0.03 |

**Đọc bảng này đúng cách (điểm dễ hiểu nhầm):** recall tổng thể của "dùng đủ cả 2-3 box" không nhỉnh hơn nhiều
so với 1 box — **nhưng khi so Full với Random cùng số box**, chênh lệch rất lớn (vd 3-box: R@1000 Full=0.68 vs
Random=0.55). Kết luận của nhóm tác giả: **một số shot buộc phải có nhiều box cùng lúc mới tìm ra được** (vì
có ≥2 đối tượng tách biệt về mặt không gian) — dùng thiếu 1 điều kiện thì bỏ sót hẳn, không phải chỉ giảm nhẹ
độ chính xác. → Đây là **bằng chứng thực nghiệm trực tiếp** cho đúng luận điểm gốc của lỗ hổng 2 (compositional
scoring): với truy vấn hợp (AND) nhiều điều kiện, **thiếu 1 điều kiện không phải "giảm điểm nhẹ" mà là "sai
hẳn"** — khớp đúng lý do vì sao dùng phép trung bình (mean/OR) là sai nguyên tắc, cần một cơ chế tôn trọng AND.

### 2.2 Intra-Video Querying — giải quyết đúng vấn đề "tìm đúng video rồi mà vẫn mất thời gian"
- Video player: cho tốc độ phát tuỳ chỉnh + hiển thị **ranh giới cảnh (scene boundary)** tường minh để nhảy
  nhanh giữa các cảnh.
- Mở rộng khả năng truy vấn: cho phép **xếp hạng lại chính các keyframe trong 1 video cụ thể** (giống cách
  CVHunter làm) — tức là chạy lại 1 vòng tìm kiếm thu nhỏ, chỉ trong phạm vi 1 video đã xác định đúng.

→ Đây là một **sub-problem mới, đáng bổ sung vào khung ban đầu**: ngoài "tìm đúng video trong toàn corpus"
(lỗ hổng 1-4 đã bàn), còn có **"tìm đúng khoảnh khắc trong 1 video đã biết chắc là đúng"** — một bài toán nhỏ
hơn, cần UI/thuật toán riêng (browse theo scene boundary, rerank cục bộ), không tự động được giải quyết chỉ
bằng làm tốt hơn recall toàn corpus.

### 2.3 Parallelized Backend
Backend đơn luồng đồng bộ trước đây gây nghẽn khi nhiều người dùng cùng lúc. V4: nhiều instance backend
bất đồng bộ sau NGINX, data service nhân bản (replicated), cân bằng tải kiểu least-connections. Kết quả nội
bộ: **25 người dùng đồng thời, < 3 giây/truy vấn**.

### 2.4 Online Learning — giải quyết đúng điểm yếu AVS
Ngoài Bayesian Update do người dùng chủ động click, thêm **rerank tự động nhẹ** cho tác vụ AVS, kết hợp
Bayesian model với 1 trong 2 lựa chọn:
- **Rocchio algorithm** (thuật toán cổ điển, dùng hiệu quả trong dense retrieval) — **khớp đúng** ý tưởng
  "Rocchio-style relevance feedback" đã đề xuất ở `phac-thao-belief-state-kisc.md` mục 3 — xác nhận lựa chọn
  đó là hợp lý, không phải phỏng đoán suông.
- **SVM huấn luyện tăng dần (incremental)** theo từng batch frame liên quan đã nộp — một lựa chọn **mới**,
  đáng cân nhắc bổ sung vào note belief-state như phương án thay thế Rocchio.

### 2.5 Keyframe Layouts — nguyên lý UX mới đáng giá, chưa từng nghĩ tới
Lưới 4 cột xếp hạng thuần tuý: nhanh khi ứng viên mạnh nằm gần đầu, nhưng **kém hiệu quả khi cần loại trừ**
(xác nhận rằng "mục tiêu chắc chắn không có trong khu vực này") trên tập đồng nhất như MVK. **Giải pháp lai:**
top-16 vẫn xếp hạng 4 cột như cũ; phần còn lại xếp theo **FLAS** (thuật toán layout lưới bảo toàn khoảng cách,
dùng CLIP feature) để **quét loại trừ nhanh hơn**.

→ **Nguyên lý đáng tách riêng, tổng quát:** **layout xếp hạng (ranked list) tối ưu cho việc XÁC NHẬN TRÚNG**
(ứng viên đúng nằm gần đầu, dễ thấy); **layout bảo toàn khoảng cách (similarity-preserving spatial layout)
tối ưu cho việc XÁC NHẬN TRẬT** (mắt người quét nhanh một vùng, thấy "không có gì giống" và bỏ qua cả cụm cùng
lúc, thay vì phải nhìn từng cái theo thứ tự rank rời rạc). Đây là 2 tác vụ thị giác khác nhau, cần 2 cách trình
bày khác nhau — nguyên lý này đáng thêm vào tư duy thiết kế UI ở `lop-tuong-tac-thuc-dung-kiem-soat-duoc.md`.

### 2.6 Thay đổi khác — chi tiết nhỏ nhưng quan trọng cho "kiểm soát được"
> *"Bayesian updates now only affect the on-screen ordering, not underlying scores, ensuring that each update
> visibly prioritizes the selected positives **without accumulating hidden history**."*

Đây là xác nhận trực tiếp, bằng chính lời tác giả, cho đúng nguyên lý "kiểm soát được" đã đặt ra: **feedback
chỉ nên đổi thứ tự hiển thị hiện tại, không âm thầm tích luỹ vào một điểm số ẩn** mà người dùng không thấy/không
gỡ được. Nếu để điểm số ẩn tích luỹ qua nhiều lượt, người dùng mất khả năng hiểu "vì sao kết quả này đang xếp
hạng như vậy" — đúng rủi ro đã cảnh báo ở lỗ hổng 5.

Cũng đáng chú ý: PraK V4 **bỏ hẳn texture/color patch query** (từng có ở V3) vì localized text query hiệu quả
hơn — một tín hiệu cho thấy không phải mọi kênh bổ sung đều đáng giữ, cần đo và sẵn sàng loại bỏ.

---

## 3. Đính chính lại so với bản trước (do trước đó chỉ có tóm tắt tìm kiếm, không có full text)

| Điều mình nói sai/thiếu trước đó | Sự thật theo bài báo |
|---|---|
| "PraK không cần Object Detection, chỉ dùng lưới tĩnh" | **Sai.** PraK V4 dùng **cả** static grid **và** dynamic region (Grounded-SAM object detector) — họ so sánh thực nghiệm và giữ **cả hai** vì tuỳ tình huống mà cái nào tốt hơn (không có 1 câu trả lời chung) |
| "Cơ chế AND chỉ là intersect tập hợp đơn giản, không cần calibration/combiner" | **Sai.** Cơ chế thật: **chuẩn hoá (z-score) từng loại khoảng cách rồi cộng lại** (centroid distance hoặc IoU, chuẩn hoá theo mean/std tính trước) — **chính là pattern "calibrate rồi combine"** đã thiết kế ở `phac-thao-compositional-scoring.md`, chỉ khác là áp cho khoảng cách hình học (box-matching) thay vì cho điểm semantic đầy đủ. Thiết kế cũ **đúng hướng hơn mình tưởng**, không cần "hạ vai trò" nhiều như bản note trước đã viết. |
| Không biết PraK yếu ở đâu | **Biết rõ:** PraK tự nhận yếu ở **AVS** và ở **tìm chi tiết trong 1 video đã biết đúng** — hai điểm yếu rất cụ thể, không phải điểm yếu chung chung |
| Chỉ biết 1/5 cải tiến | Nay có đủ 5/5, xem mục 2 |

---

## 4. Việc cần làm — bổ sung cụ thể vào các note trước

1. **Thêm bước kiểm tra độ đồng nhất thị giác của corpus AIC** (mục 0) trước khi giả định CLIP baseline luôn đủ.
2. **Không bỏ qua Object Detection cho binding** — dùng **cả 2** (static grid rẻ + dynamic dựa trên Objects đã
   có sẵn trong dữ liệu AIC) song song, đúng như PraK làm, thay vì chọn 1.
3. **Cơ chế combine cho binding nên là "z-score rồi cộng"** (đã có PraK xác nhận hiệu quả bằng số liệu thật),
   khớp đúng, không cần thiết kế lại — chỉ áp dụng đúng công thức calibrate+combine đã có ở note compositional
   scoring, cho cả trường hợp box-matching.
4. **Bổ sung mục "Intra-video Querying"** như 1 mode riêng trong multi-mode browsing (`lop-tuong-tac-thuc-dung-kiem-soat-duoc.md`
   Lớp 1) — tìm trong 1 video đã xác định đúng là bài toán khác, cần UI riêng (scene boundary, rerank cục bộ).
5. **Bổ sung nguyên lý layout kép** (ranked-list để confirm-hit, FLAS/similarity-layout để confirm-miss) vào
   thiết kế UI.
6. **Thêm SVM incremental** như phương án thay thế/bổ sung Rocchio cho online learning trong AVS.
7. **Bayesian update chỉ nên đổi thứ tự hiển thị, không tích luỹ điểm ẩn** — áp dụng nguyên tắc này khi cài
   `update_belief` (đảm bảo luôn có thể trace lại "vì sao thứ tự này", không có state ẩn không kiểm soát được).

---

## 5. Tổng kết

Bản đầy đủ này **củng cố** (không đảo ngược) toàn bộ hướng đã đi: bài toán binding/compositional AND (lỗ hổng
2) có lời giải thực tế, đã vô địch, dùng **đúng nguyên lý calibrate+combine** đã thiết kế — chỉ khác PraK áp
dụng nó ở mức "khoảng cách hình học box" thay vì "điểm semantic đầy đủ", và họ **không chọn 1 giải pháp duy
nhất (static hoặc dynamic)** mà cung cấp cả hai vì hiệu quả phụ thuộc ngữ cảnh — bài học "đo đạc thực nghiệm
thay vì đoán cái nào tốt hơn" cũng đáng ghi nhớ cho chính hệ thống của mình.

---

## Sources

- Jäckl, B., Verner, B., Stroh, M., Kloda, V., Nagy, L., Deussen, O., Keim, D.A., Lokoč, J.: *PraK V4 at the
  Video Browser Showdown 2026*. In: MMM 2026, LNCS 16415, pp. 230–237. (bản đầy đủ do user cung cấp trực tiếp)
