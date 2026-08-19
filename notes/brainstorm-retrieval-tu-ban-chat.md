# Brainstorm AIC 2026 — Hệ thống Retrieval nhìn từ bản chất

> Mục tiêu của note này: hiểu **tại sao** một hệ thống retrieval cần những thành phần nó cần,
> bằng cách suy ngược từ bản chất bài toán — thay vì copy công thức "CLIP + Faiss + reranker" từ các paper.
> Ngày: 2026-07-05.

---

## Phần 0 — Vấn đề vừa đặt ra (framing)

### Bản chất của retrieval
Retrieval về cơ bản là bài toán **xây một không gian đo khoảng cách** sao cho:

> "khoảng cách ngữ nghĩa theo cảm nhận con người" ≈ "khoảng cách trong không gian biểu diễn của máy"

Mọi khó khăn đều sinh ra từ chỗ hai khoảng cách này bị **lệch** (semantic gap).

### 3 gốc rễ của semantic gap
1. **Con người mô tả bằng khái niệm trừu tượng và không đầy đủ** (trí nhớ mờ, thiếu chi tiết), trong khi dữ liệu là pixel thô.
2. **Một câu truy vấn thường là phép HỢP (AND) của nhiều điều kiện độc lập** phải xảy ra *đồng thời* — không phải một điểm duy nhất trong không gian embedding.
3. **Ý nghĩa phụ thuộc thứ tự thời gian**, còn hầu hết biểu diễn hình ảnh/video là *bất biến theo thời gian*.

### 5 lỗ hổng suy ra từ 3 gốc rễ (checklist làm việc)
| # | Lỗ hổng | Bài toán buộc ta cần gì |
|---|---------|--------------------------|
| 1 | **Representation granularity** | Vector toàn cục làm loãng object nhỏ/chữ/âm thanh → cần biểu diễn đa mức độ |
| 2 | **Compositional scoring** | 1 điểm cosine không diễn tả được AND nhiều điều kiện → phân rã rồi hợp lại |
| 3 | **Temporal modeling** | Embedding từng frame không mã hoá trước/sau → cần tầng xử lý chuỗi riêng |
| 4 | **Coarse-recall vs fine-verify** | Quy mô lớn cấm chạy model nặng toàn corpus → lọc thô rẻ + verify đắt trên shortlist |
| 5 | **Iterative refinement** | Query của người dùng vốn thiếu bit → retrieval không thể one-shot; tương tác là một phần thuật toán |

**Cách dùng checklist:** với bất kỳ thiết kế nào, đừng hỏi "mình đã dùng CLIP/Faiss chưa" mà hỏi
"hệ thống của mình có đóng được 5 lỗ hổng này không, và đóng ở đâu".

---

## Phần 1 — Đi sâu 5 lỗ hổng

### Lỗ hổng 1 — Representation granularity (độ hạt của biểu diễn)

**Gốc rễ.** Một global embedding là một phép **nén mất mát** (lossy compression): bạn ép hàng triệu pixel
xuống 512–768 chiều. Cái sống sót là ngữ nghĩa *chủ đạo* của ảnh; object nhỏ, chữ, thuộc tính tinh vi bị
"trung bình hoá" và biến mất. Đây chính là lý do "móc khóa gấu bông màu hồng" (vật thể tí xíu) tan biến khi
bạn chỉ so khớp bằng 1 vector toàn ảnh.

**Suy từ số 0.** Bạn không biết trước câu hỏi nhắm vào *mức độ nào*. Nó có thể hỏi về:
- toàn cảnh (gist): "bãi biển lúc hoàng hôn"
- một object: "chiếc chìa khoá"
- thuộc tính của object: "móc khoá **màu hồng**"
- chữ trong ảnh (OCR): dòng chữ chạy trên bản tin
- âm thanh (ASR): lời người dẫn chương trình

→ Mỗi mức là một **kênh thông tin** riêng, cần cách trích khác nhau. Vì không biết trước, bạn cần
biểu diễn **đa mức độ (multi-granularity)**.

**Không gian thiết kế.**
- Global CLIP embedding — rẻ, **đã được cung cấp sẵn** (CLIPFeatures).
- Region/object embeddings — crop các bounding box (Objects đã cho sẵn!) rồi CLIP-encode từng crop; hoặc dùng nhãn object làm **inverted index** rời rạc.
- **OCR** (chữ trong khung hình) — với dữ liệu HTV/bản tin thì cực kỳ giàu chữ (banner, dòng tin chạy, "Chính phủ đồng ý giảm 10% thuế nhập khẩu xăng"...).
- **ASR** (speech → text) — lời dẫn, phụ đề.
- Metadata YouTube (title/description/keywords) — ngữ cảnh ở mức *video*.

**Điểm mấu chốt cho dữ liệu AIC 2026.** CLIPFeatures + Objects là baseline **ai cũng có**. Yếu tố tạo khác biệt
là **OCR và ASR** — không nằm trong feature được phát, mà lại đặc biệt có giá trị với dữ liệu bản tin tiếng Việt
(chữ trên màn hình + lời dẫn mang rất nhiều thông tin định danh).

**Nguyên lý sâu.**
> Chất lượng retrieval bị **chặn trên** bởi việc thông tin phân biệt có *sống sót vào index* hay không.
Nếu hai frame chỉ khác nhau ở dòng chữ OCR mà bạn không index chữ đó, thì *không reranker nào cứu được*.
Representation là **trần**; mọi bước sau chỉ sắp xếp lại bên dưới cái trần đó.

**Tự kiểm.** Biểu diễn của bạn có giữ chi tiết ở nhiều mức hạt không, hay chỉ 1 vector toàn cục?

---

### Lỗ hổng 2 — Compositional scoring (chấm điểm theo tổ hợp)

**Gốc rễ.** `cosine(query, frame)` cho ra **một số vô hướng duy nhất**. Nhưng câu hỏi thật là tổ hợp nhiều
mệnh đề con (entity + attribute + action + context) phải *cùng đúng*. CLIP text encoder nén cả câu thành 1 vector,
và tích vô hướng hành xử gần như **trung bình mềm** các khái niệm khớp — nên một frame khớp mạnh điều kiện A
nhưng thiếu B, C vẫn có thể *ăn điểm cao hơn* frame khớp vừa phải cả A, B, C. CLIP nổi tiếng hành xử như
"túi khái niệm" (bag of concepts): yếu ở **binding** (thuộc tính nào gắn với object nào), yếu ở đếm và phủ định.

Đây đúng là ví dụ slide 10: "cầm kem **VÀ** ở biển" — bất kỳ khoảnh khắc nào chỉ có biển *hoặc* chỉ có kem
đều **không hợp lệ**.

**Suy từ số 0.** Muốn diễn tả AND, bạn phải:
1. **phân rã** query thành các mệnh đề con,
2. chấm điểm mỗi mệnh đề *độc lập* lên biểu diễn đa mức độ (nối với lỗ hổng 1),
3. **hợp lại** bằng toán tử tôn trọng phép AND,
4. xử lý **binding**: "túi tím VÀ túi trắng" cần tím gắn vào một túi, trắng vào túi kia.

**Không gian thiết kế.**
- **Query decomposition bằng LLM:** "cầm kem VÀ ở biển" → `[ice cream in hand, seaside]` → chấm từng cái → hợp.
- **Toán tử hợp:** tổng (giống OR — chính là CLIP thường), **min** (AND nghiêm ngặt), soft-min, có trọng số.
  `min` phạt nặng khi thiếu bất kỳ điều kiện nào → hợp với KIS (vốn là tổ hợp chặt).
- **Grounding bằng Objects:** dùng kênh object để verify "có người", "đếm = 2", quan hệ không gian.
- **Phủ định** ("cảnh ăn ở nhà thì không liên quan"): CLIP không làm được NOT → cần lọc/ trừ điểm tường minh.
- **Binding:** khó thật sự; chấm điểm ở mức *vùng* (attribute so với vùng cụ thể, không so cả ảnh) sẽ đỡ hơn.

**Nguyên lý sâu.**
> Query là một **công thức logic**, không phải một chuỗi. Embedding search coi nó là *một điểm*; hệ thống tốt
coi nó là *một vị từ cần được đánh giá*. Khoảng cách giữa hai cách nhìn này chính là nơi **KIS được ăn/thua**,
vì ground truth đúng là frame thoả *toàn bộ* tổ hợp, còn distractor chỉ thoả *tập con*.

**Liên hệ lỗ hổng 4.** Chấm điểm phân rã thì đắt → chỉ chạy ở **stage verify** trên shortlist, không chạy toàn corpus.

**Tự kiểm.** Bạn có cơ chế cho truy vấn hợp nhiều điều kiện, hay chỉ so 1 embedding duy nhất?

---

### Lỗ hổng 3 — Temporal modeling (mô hình hoá thời gian)

**Gốc rễ.** Một keyframe là *một thời điểm*; embedding từng frame **bất biến theo thời gian** ngay từ thiết kế.
"Cởi mũ *rồi mới* vào phòng" và "vào phòng *rồi mới* cởi mũ" chứa cùng tập frame/object → mọi biểu diễn kiểu
"túi frame" đều chấm điểm chúng *giống hệt nhau*. Trật tự là thông tin nằm **giữa** các frame, không nằm **trong**
bất kỳ frame nào.

**Suy từ số 0.** Bạn cần:
- khái niệm **cục bộ thời gian** — gom frame thành shot/scene/window,
- khả năng khớp một **chuỗi** sự kiện con *theo thứ tự*,
- dung sai cho khoảng trống / tốc độ (hành động có thể 1s hoặc 10s).

**Không gian thiết kế.**
- **Shot/scene segmentation** (gom keyframe liền kề tương tự — slide 34 có nhắc).
- **Temporal window matching:** tìm frame-anchor cho sự kiện A, rồi tìm sự kiện B trong *cửa sổ phía sau*.
- **Sequence model** trên chuỗi embedding (nặng → chỉ ở stage verify).
- **VQA:** temporal reasoning là *bản chất bài toán* → phải localize rồi suy luận trên đoạn clip đã sắp thứ tự.
- **Trục thời gian cho không** nhờ metadata timestamp + thứ tự frame trong video.

**Lưu ý riêng cho dữ liệu.** Keyframe là I-frame → **thưa**. Hành động tinh vi ("đánh rơi chìa khoá trong 1–2s")
có thể **rơi trọn vào giữa hai I-frame** → giới hạn recall *cố hữu*. Cần chấp nhận: một số khoảnh khắc KIS
có thể *không có keyframe* bắt đúng đỉnh hành động; khi đó ta khớp bằng **ngữ cảnh xung quanh** thay vì chính hành động.

**Nguyên lý sâu.**
> Similarity thì **đối xứng và không thứ tự**; ý nghĩa thì thường *không*. Temporal logic là **ràng buộc để verify**,
không phải similarity để maximize. Nên về kiến trúc, nó thuộc về tầng **lọc/rerank trên các window ứng viên**,
sau khi coarse recall đã nổi đúng vùng video lên.

**Tự kiểm.** Bạn có tầng riêng cho thứ tự thời gian, hay coi mọi frame là độc lập?

---

### Lỗ hổng 4 — Coarse-recall vs fine-verify (+ thiết kế theo metric)

**Gốc rễ.** Hai ràng buộc cứng va nhau: (1) corpus khổng lồ (hàng nghìn giờ), (2) những model thật sự *hiểu*
tổ hợp/thời gian/binding (LVLM, cross-encoder) thì *đắt* — không thể chạy trên mọi frame. Vậy **bắt buộc** phải có
một bộ lọc rẻ đưa hàng triệu frame → vài trăm ứng viên với **recall cao**, rồi một stage đắt sắp xếp lại với
**precision cao**.

Đây là phân công lao động recall/precision kinh điển:
- **Coarse (thiên recall):** ANN search trên CLIP embedding (Faiss/Milvus). *Không được để lọt* ground truth. Rẻ, xấp xỉ, chạy toàn corpus.
- **Fine (thiên precision):** verify tổ hợp + thời gian + LVLM trên shortlist. Đắt, chính xác, tập rất nhỏ.

**Vì sao là "buộc phải", không phải "chọn cho vui".** Đây là cách *duy nhất* thoả đồng thời "hiểu sâu" VÀ "trả lời kịp giờ".
Nếu coarse recall làm rớt đáp án → *không reranker nào cứu* (recall là trần). Nếu bỏ fine verify → KIS Top-1 sụp.
Hai stage có **kiểu lỗi bù trừ nhau** và cả hai đều bắt buộc.

**Phần thiết kế theo metric — insight quan trọng: mỗi loại query cần đầu tư khác nhau.**
- **KIS:** metric ≈ trúng Top-1/Top-5, thường có đường cong thưởng rank sớm. → đầu tư vào **rerank/verify**; recall top-100 thường dễ, trận đánh nằm ở *sắp xếp top-5*.
- **AVS:** metric ≈ AP/recall trên nhiều item liên quan. → đầu tư vào **recall + calibrate điểm** để điểm *so sánh được xuyên corpus*, không chỉ tốt cục bộ.
- **VQA:** metric ≈ độ đúng câu trả lời. → đây là *retrieve-then-read*; đầu tư vào localize + model suy luận đọc lại bằng chứng.
- **KISC:** metric ≈ thành công trong N lượt. → đầu tư vào **state** (giữ tập ứng viên xuyên lượt) + **chọn câu hỏi** chia đôi tập ứng viên (information gain).

**Nguyên lý sâu.**
> **Metric nói cho bạn biết công sức biên tế nên đổ vào đâu.** Đội nào dồn hàng tuần vào một encoder hoa mỹ mà
quên rằng KIS là trò chơi Top-5 (phải rerank!) hoặc KISC là trò chơi information-gain (phải hỏi câu chia đôi!)
thì đang tối ưu sai chỗ. **Thiết kế ngược từ hàm chấm điểm.**

**Tự kiểm.** Kiến trúc có tách recall-rẻ-toàn-corpus và verify-đắt-tập-nhỏ không? Bạn có biết chính xác metric của từng loại query không?

---

### Lỗ hổng 5 — Iterative refinement / tương tác là thuật toán

**Gốc rễ.** Query đến từ **trí nhớ con người dễ sai** — thiếu, đôi khi lệch. Xét theo lý thuyết thông tin, một query
thường *không đủ bit* để cô lập một item giữa hàng nghìn. Vậy retrieval one-shot là **thiếu xác định (under-determined)
ngay từ bản chất** — không model nào bù được thông tin *vắng mặt*; nó chỉ có thể được *cung cấp dần theo thời gian*.

**Suy từ số 0.** Bạn cần một vòng lặp: (a) hiện ứng viên tốt nhất hiện tại, (b) cho người dùng (hoặc hệ thống trong KISC)
bơm thêm ràng buộc, (c) cập nhật tập ứng viên, (d) chọn *hỏi/hiện gì tiếp theo* để tối đa tiến độ. Đây là
**search-as-dialogue**, và cấu trúc giống nhau dù "tương tác" là người bấm filter hay LVLM hỏi lại.

**Không gian thiết kế.**
- **Thu hẹp theo độ chắc chắn** (slide 40): lọc theo trục *đáng tin nhất* trước — thời gian (metadata *chính xác*) > địa điểm > object có mặt (detector khá tin cậy) > mô tả cảnh mờ (kém tin nhất). Mỗi filter *precision cao và rẻ*, thu nhỏ không gian *trước khi* xài similarity CLIP mờ.
- **Query expansion:** LLM biến query thưa thành nhiều cách diễn đạt/kịch bản → tăng recall (phủ vấn đề lệch từ vựng).
- **Relevance feedback:** người dùng đánh dấu 1 kết quả gần đúng → search theo *độ tương đồng hình ảnh* với nó ("more like this", image-to-image).
- **Chọn câu hỏi trong KISC:** chọn câu hỏi làm rõ có **information gain kỳ vọng cao nhất** (chia tập ứng viên ~đôi) — đây là khung *active learning / lý thuyết quyết định*, không phải hỏi bừa.
- **Xem theo timeline/browse:** khai thác việc video/lifelog có *hàng xóm thời gian* — đáp án thường nằm *cạnh* một hit gần-đúng.

**Nguyên lý sâu.**
> **Tương tác biến một suy luận one-shot khó thành một chuỗi quyết định rẻ và dễ.** UI không phải trang trí quanh model;
nó là *nơi các bit còn thiếu đi vào hệ thống*. Với cuộc thi có human-in-the-loop, một model tầm trung + giao diện
thu hẹp/feedback xuất sắc **thường thắng** một model xịn + UI one-shot ngớ ngẩn. Đây là lý do lịch sử vì sao các hệ
thống thắng VBS/AIC "ăn nhau ở giao diện" ngang với ở embedding.

**Tự kiểm.** Hệ thống có cho phép lặp và tinh chỉnh (qua UI hoặc hội thoại), hay one-shot?

---

## Phần 2 — Tổng hợp xuyên suốt

- **Lỗ hổng 1–3** là về *sức mạnh biểu diễn & so khớp* → quyết định **trần** chất lượng.
- **Lỗ hổng 4** là về *cách tiêu compute để chạm tới cái trần đó* dưới ràng buộc quy mô/thời gian.
- **Lỗ hổng 5** là về *cách moi đủ thông tin từ người dùng* để bài toán trở nên *giải được*.

Một hệ thống hoàn chỉnh cần **cả năm**; một paper thường chỉ đẩy *một*.

**Bản đồ với dữ liệu được phát:**
| Lỗ hổng | Baseline ai cũng có | Yếu tố tạo khác biệt |
|---------|---------------------|----------------------|
| 1 Representation | CLIPFeatures, Objects | **OCR + ASR** (đặc biệt hợp bản tin tiếng Việt) |
| 2 Compositional | so cosine cả câu | **LLM phân rã query** + toán tử min + grounding bằng Objects |
| 3 Temporal | keyframe rời rạc | **shot/scene + window matching** |
| 4 Recall/Verify | Faiss trên CLIP | **pipeline 2 tầng thật sự** + rerank/LVLM verify |
| 5 Interaction | ô search một lần | **UI thu hẹp theo độ chắc chắn + feedback + KISC hỏi chia đôi** |

> Kim chỉ nam: **Thiết kế ngược từ metric của từng loại query, và luôn hỏi "thông tin phân biệt có sống sót vào index chưa".**
