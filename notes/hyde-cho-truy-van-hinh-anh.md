# HyDE cho truy vấn hình ảnh — tổng quát hoá từ RAG sang Visual Retrieval

> Ghi lại quan sát: ý tưởng Stable Diffusion sinh ảnh truy vấn (NII-UIT, VBS 2025 — xem
> `doi-chieu-thuc-te-cac-doi-vo-dich.md` §2) về bản chất là **HyDE (Hypothetical Document Embeddings)**
> áp dụng sang domain ảnh. Nối tiếp lỗ hổng 1 (representation) ở `brainstorm-retrieval-tu-ban-chat.md`.
> Ngày: 2026-07-07.

---

## 1. HyDE gốc là gì (đã xác minh)

**HyDE** (Gao, Ma, Lin, Callan — *"Precise Zero-Shot Dense Retrieval without Relevance Labels"*, ACL 2023):
thay vì embed trực tiếp câu query để tìm trong corpus, quy trình là:

1. Đưa query cho một LLM, yêu cầu **sinh ra một "tài liệu giả định"** trả lời câu hỏi đó — tài liệu này
   **có thể chứa chi tiết sai/bịa** (không sao, vì nó không được dùng làm đáp án, chỉ làm "mỏ neo" để truy vấn).
2. Encode tài liệu giả định đó bằng một **contrastive encoder không giám sát** (vd Contriever).
3. Dùng embedding đó để tìm **tài liệu thật** trong corpus theo similarity.

Hiệu quả vì: tài liệu giả định — dù nội dung có thể sai — **có hình thức/phân phối giống một tài liệu thật**
(độ dài, văn phong, mật độ thông tin), nên nó nằm **gần các tài liệu thật liên quan hơn** là câu query gốc
(vốn ngắn, trừu tượng) từng nằm. Đây chính là bài kiểm chứng thực nghiệm cho nguyên lý gốc rễ #1 đã đặt ra từ
đầu chuỗi note: *"con người mô tả bằng khái niệm trừu tượng, không đầy đủ, trong khi dữ liệu là [pixel/text] thô"*.

---

## 2. Tổng quát hoá sang ảnh — và một gốc rễ *thứ hai* mà mình chưa từng tách riêng

Ý tưởng NII-UIT (sinh ảnh bằng Stable Diffusion rồi search bằng ảnh đó) là **HyDE cho domain ảnh**: thay vì
so `text_embedding(query)` với `image_embedding(keyframe)` (cross-modal, CLIP text-vs-image), sinh ra một
**ảnh giả định** từ query, encode ảnh đó bằng đúng encoder đã dùng để index corpus, rồi so
`image_embedding(ảnh_giả_định)` với `image_embedding(keyframe)` — **cùng modal (image-image)**.

Tìm hiểu thêm phát hiện: retrieval cross-modal (text↔image) trong CLIP yếu hơn within-modal (image↔image)
không chỉ vì lý do "thông tin thiếu" (root cause #1 cũ) mà còn vì một hiện tượng **hình học đã được đo đạc
riêng** trong literature: **modality gap** — text embedding và image embedding trong không gian CLIP chung
thực ra nằm trên **hai "cụm/nón" tách biệt nhau**, lệch hẳn khỏi kỳ vọng lý tưởng "cặp khớp nghĩa thì trùng
điểm". Đây là **gốc rễ độc lập, mang tính hình học/thuật toán huấn luyện**, khác với gốc rễ #1 (mang tính
thông tin/ngôn ngữ). → Nên bổ sung vào danh sách gốc rễ semantic gap ban đầu:

> **Gốc rễ #4 (mới):** ngay cả khi thông tin đầy đủ, embedding text và embedding image trong không gian
> contrastive chung **không thực sự chồng lấp** — chúng lệch nhau một khoảng hệ thống (modality gap) do bản
> chất huấn luyện contrastive, không phải do thiếu thông tin ngữ nghĩa.

→ HyDE-cho-ảnh giải quyết *đồng thời* cả 2 gốc rễ: chuyển câu hỏi (ngắn, trừu tượng, gốc rễ #1) thành một ảnh
cụ thể (đủ chi tiết hơn), **và** đưa việc so khớp về đúng modal ảnh-ảnh (né gốc rễ #4).

---

## 3. Ba biến thể hiện thực — chi phí/độ chính xác khác nhau

| Biến thể | Cách làm | Chi phí | Ghi chú |
|---|---|---|---|
| **(A) Sinh pixel đầy đủ** (NII-UIT dùng) | text → Stable Diffusion → ảnh pixel thật → encode lại bằng CLIP/PE image encoder → search | Cao nhất (1 lượt sinh ảnh diffusion/query) | Dùng được với **bất kỳ image encoder có sẵn**, không cần train gì thêm |
| **(B) "Prior" trong không gian embedding** (kiểu DALL-E 2/unCLIP) | train một mạng nhỏ dự đoán **thẳng CLIP image-embedding** từ CLIP text-embedding, không sinh pixel | Thấp hơn nhiều (chỉ 1 forward pass mạng nhỏ, không cần diffusion decode) | Cần có/train sẵn mạng "prior" — tốn công chuẩn bị nhưng **rẻ tại thời điểm query** |
| **(C) Hiệu chỉnh trung bình (mean-shift)** | tính `mean_image_embedding` và `mean_text_embedding` trên toàn corpus, dịch `text_embedding(query)` theo `(mean_image − mean_text)` trước khi search | Rẻ nhất — không cần model sinh gì cả, chỉ 1 phép trừ vector | Không "làm giàu" thông tin như (A)/(B), chỉ **hiệu chỉnh độ lệch hình học hệ thống** — giải quyết gốc rễ #4 nhưng không giải quyết gốc rễ #1 |

**Gợi ý thực dụng cho việc ưu tiên:** thử **(C) trước** vì gần như miễn phí — chỉ cần tính 2 vector trung bình
từ chính CLIPFeatures đã có sẵn trong dữ liệu được phát, so sánh Recall@K (bộ metric đã dựng) trước/sau khi
áp mean-shift, để biết modality gap có phải vấn đề đáng kể trên chính dữ liệu HTV này hay không, **trước khi**
đầu tư vào (A) (chi phí cao hơn, cần gọi Stable Diffusion mỗi query — cộng vào latency, đúng ràng buộc
"hiệu năng" vừa đặt ra ở note trước).

---

## 4. Rủi ro — transfer thẳng từ bài học HyDE gốc, cần cảnh giác trước khi dùng

### Rủi ro 1 — Hallucination bias (giống hệt HyDE gốc)
Nếu ảnh/embedding giả định "tự tin" sinh ra một chi tiết **sai** (vd Stable Diffonic vẽ móc khoá gấu bông
màu **đỏ** trong khi thật ra là **màu hồng**), việc search sẽ bị neo vào chi tiết sai đó — **có thể tệ hơn**
so với dùng thẳng câu query gốc, vì ít nhất câu chữ gốc không "cam kết" sai một màu cụ thể như ảnh sinh ra.

### Rủi ro 2 — Hồi quy về nguyên mẫu (prototype), mâu thuẫn trực tiếp với yêu cầu của KIS
Model sinh ảnh được huấn luyện trên thống kê trung bình → có xu hướng vẽ ra phiên bản **"điển hình nhất"**
của một khái niệm (một "người lính chì" sinh ra sẽ giống hệt hình nộm Giáng sinh phổ biến nhất trên mạng),
**không phải** đúng vật thể cụ thể, khác thường xuất hiện trong đúng đoạn video cần tìm. Đây là **mâu thuẫn
trực tiếp** với bản chất bài toán KIS (lỗ hổng 4 — KIS cần Top-1 tuyệt đối cho **1 instance cụ thể**, không
phải "một ví dụ đại diện của loại này"). → HyDE-cho-ảnh giúp thu hẹp semantic gap nhưng có thể **vô tình kéo
xa khỏi instance thật** — cần cân nhắc kỹ khi áp cho KIS so với AVS (AVS chấp nhận "giống loại" hơn KIS).

### Mitigation — transfer thẳng từ literature HyDE, không cần nghĩ lại từ đầu
1. **Sinh nhiều mẫu, không chỉ 1** (HyDE gốc cũng làm vậy): sinh N=4-8 ảnh khác nhau (seed khác nhau) từ cùng
   query, encode tất cả, rồi **hợp pool ứng viên** từ cả N ảnh — giảm rủi ro 1 lần sinh bị lệch. Đây chính là
   pattern "recall-by-decomposition, hợp pool" đã thiết lập ở lỗ hổng 2.5 — **không cần cơ chế mới**, dùng lại
   nguyên hạ tầng đã có.
2. **Không thay thế, mà bổ sung song song:** chạy search bằng ảnh-giả-định như **một kênh recall thêm**, chạy
   *song song* với search bằng CLIP text-query gốc trực tiếp — giống hệt mô hình "multi-mode song song" đã
   chốt ở note trước (`lop-tuong-tac-thuc-dung-kiem-soat-duoc.md` Lớp 1) — không đặt cược toàn bộ vào 1 kênh.

---

## 5. Vị trí trong pipeline & chi phí — vì sao không vi phạm ràng buộc "hiệu năng"

Đây là **chi phí cố định mỗi query** (1 lần sinh ảnh + 1 lần encode), **không phụ thuộc kích thước corpus**
— vì phần index (embedding của toàn bộ keyframe) đã có sẵn từ trước, kênh mới chỉ thêm 1 điểm truy vấn nữa
vào ANN search đã tồn tại. Do đó nó **không vi phạm** ràng buộc "coarse recall phải rẻ trên toàn corpus" đã
nói ở lỗ hổng 4 — chi phí phát sinh chỉ nằm ở phía query, tách biệt khỏi corpus. Vẫn cần đo latency thật của
riêng bước sinh ảnh (biến thể A có thể mất 1-3 giây/lần với diffusion model, đáng kể trong bối cảnh thi có
đồng hồ) — đây là lý do biến thể (C)/(B) đáng thử trước biến thể (A) như đã nói ở mục 3.

---

## 6. Tổng kết — nối các note lại

- Xác nhận thêm 1 gốc rễ mới cho semantic gap (modality gap hình học), bổ sung cho `brainstorm-retrieval-tu-ban-chat.md`.
- Kỹ thuật cụ thể (3 biến thể + rủi ro + mitigation) bổ sung trực tiếp cho mục "Ý tưởng mới nổi bật" ở
  `doi-chieu-thuc-te-cac-doi-vo-dich.md` §2.
- Cách triển khai (kênh recall song song, không thay thế) khớp thẳng kiến trúc đã chốt ở
  `lop-tuong-tac-thuc-dung-kiem-soat-duoc.md` — không cần cơ chế mới, chỉ thêm 1 kênh vào hạ tầng multi-mode đã có.
- **Việc cần làm nếu muốn thử nghiệm:** bắt đầu bằng biến thể (C) — mean-shift, gần như miễn phí, đo bằng
  Recall@K trên validation set đã dựng (`bo-metric-va-validation-set.md`) trước khi cân nhắc đầu tư (A)/(B).

---

## Sources

- [Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE) — arXiv](https://arxiv.org/abs/2212.10496)
- [HyDE — ACL Anthology (bản chính thức ACL 2023)](https://aclanthology.org/2023.acl-long.99/)
- [GitHub texttron/hyde — cài đặt tham khảo](https://github.com/texttron/hyde)
- [The What and Why of Text-Image Modality Gap in CLIP Models — Jina AI](https://jina.ai/news/the-what-and-why-of-text-image-modality-gap-in-clip-models/)
- [Fill the Gap: Quantifying and Reducing the Modality Gap in Image-Text Representation Learning — arXiv](https://arxiv.org/html/2505.03703v1)
