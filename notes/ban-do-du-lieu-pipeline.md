# Bản đồ Dữ liệu → Tầng Pipeline

> Nối tiếp `brainstorm-retrieval-tu-ban-chat.md` (5 lỗ hổng) và `lo-hong-2-va-5-chi-tiet.md`.
> Mục tiêu: chốt xem **mỗi nguồn dữ liệu** (được phát + tự làm thêm) **rơi vào tầng nào** của pipeline,
> đóng **lỗ hổng nào**, và giúp **loại query nào** — để khi bắt tay code, biết ngay thứ tự ưu tiên.
> Ngày: 2026-07-06.

---

## 0. Khung pipeline tổng thể (nhắc lại, giờ vẽ đủ tầng)

```
                         ┌─────────────────────────────────────────┐
  OFFLINE / INDEXING     │  Video → Keyframe → [nhiều encoder song song] → lưu vào các index riêng  │
                         └─────────────────────────────────────────┘
                                          │
  ┌───────────────────────────────────────┴───────────────────────────────────────┐
  │ ONLINE / RETRIEVAL (mỗi query đi qua các tầng theo thứ tự)                     │
  │                                                                                 │
  │  Tầng 1 — FILTER (cứng, rẻ, tin cậy cao)                                       │
  │      lọc bằng: thời gian, kênh/video, địa điểm (nếu suy ra được)               │
  │                                                                                 │
  │  Tầng 2 — COARSE RECALL (rẻ, chạy trên phần còn lại của corpus)                │
  │      ANN search trên CLIP embedding (toàn cảnh) → top-K ứng viên (K ~ 100-1000)│
  │      (tuỳ chọn) hợp thêm pool từ recall-theo-từng-vị-từ (lỗ hổng 2.5)          │
  │                                                                                 │
  │  Tầng 3 — FINE VERIFY / COMPOSITIONAL SCORING (đắt, chỉ chạy trên top-K)       │
  │      parse query → chấm từng vị từ bằng evaluator riêng (Objects/OCR/ASR/LVLM) │
  │      combine (min/weighted) → re-rank                                          │
  │                                                                                 │
  │  Tầng 4 — TEMPORAL CONSOLIDATION                                               │
  │      gom theo shot/scene, khớp thứ tự sự kiện, mở rộng cửa sổ quanh anchor      │
  │                                                                                 │
  │  Tầng 5 — INTERACTION LOOP                                                     │
  │      hiển thị đa dạng (MMR) → nhận feedback/hỏi đáp (KISC) → quay lại Tầng 1/2  │
  │                                                                                 │
  │  Tầng 6 — TASK-SPECIFIC HEAD                                                   │
  │      KIS: chốt Top-1  |  AVS: xuất ranked list toàn cục  |  VQA: LVLM sinh câu  │
  │      trả lời có căn cứ trên frame đã localize | KISC: cập nhật belief + hỏi tiếp│
  └─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Bảng tổng — từng nguồn dữ liệu rơi vào đâu

| Nguồn dữ liệu | Có sẵn hay tự làm? | Tầng pipeline chính | Lỗ hổng đóng | Query type hưởng lợi nhiều nhất |
|---|---|---|---|---|
| **CLIPFeatures** (keyframe embedding) | **Có sẵn** | Tầng 2 (Coarse recall) | #1 (representation mức-cảnh), #4 (rẻ, ANN toàn corpus) | Tất cả (nền tảng) |
| **Objects** (detection, 600 categories) | **Có sẵn** | Tầng 3 (Fine verify) — evaluator cho `exists`, `count`, `attribute` (qua box); cũng dùng làm inverted index lọc thô ở Tầng 1/2 | #2 (compositional: exists/count/binding), #1 (mức object) | KIS (loại trừ distractor), AVS (lọc theo object) |
| **Metadata** (YouTube JSON) | **Có sẵn** | Tầng 1 (Filter cứng) | #5 (trục tin cậy cao nhất: thời gian, kênh, title/keywords) | KIS, KISC (thu hẹp nhanh, rẻ) |
| **Videos** (gốc) | **Có sẵn** | Nguồn cho mọi thứ tự làm thêm bên dưới (OCR/ASR/shot-seg cần decode video, không chỉ keyframe) | — (hạ tầng) | — |
| **Keyframes** (I-frame) | **Có sẵn** | Đơn vị index cơ bản của Tầng 2 & 3 | #3 (đơn vị thời gian thưa — giới hạn cố hữu đã ghi nhận) | Tất cả |
| **OCR** (chữ trong khung hình) | **Tự làm** (chạy trên Keyframes) | Tầng 3 (evaluator cho vị từ `text_in_frame`); có thể build index riêng để dùng luôn ở Tầng 2 (khớp chữ chính xác rất mạnh cho KIS) | #1 (kênh mới hoàn toàn), #2 | **KIS** đặc biệt mạnh với dữ liệu bản tin HTV (chữ chạy, banner số liệu) |
| **ASR** (speech-to-text) | **Tự làm — TẠM GÁC (2026-07-19)**, xem ghi chú cuối mục 2 | Tầng 1/2 — vì ASR có transcript theo mốc thời gian *liên tục*, dùng để **lọc theo nội dung lời nói** trước khi vào CLIP, hoặc làm text index song song | #1, #3 (ASR có timeline riêng, giúp neo temporal) | KIS/VQA khi câu hỏi liên quan lời thoại/tin tức |
| **Region-CLIP / object crop embedding** | **Tự làm** (crop theo box của Objects, encode lại bằng CLIP) | Tầng 3 — evaluator cho `attribute(O,A)` (chấm trên vùng, không phải cả ảnh) | #2 (binding — mấu chốt để tránh "túi tím" trôi sang túi khác) | KIS có mô tả thuộc tính chi tiết (màu, loại vật) |
| **Shot/Scene segmentation** (gom keyframe liền kề tương tự) | **Tự làm** (cluster trên CLIPFeatures hoặc pixel-diff) | Tầng 4 (Temporal) | #3 | AVS (gom kết quả trùng lặp trước khi xếp hạng), VQA (định biên đoạn cần đọc) |
| **LVLM/VQA model** (Gemini/GPT-4o hoặc mô hình mở) | **Tự làm/tích hợp** | Tầng 3 (verify quan hệ/phủ định phức tạp) + Tầng 6 (sinh câu trả lời VQA) + Tầng 5 (soạn câu hỏi làm rõ cho KISC) | #2 (relation/negation), #5 (chọn câu hỏi) | VQA, KISC, và verify cuối cho KIS |
| **Belief-state / candidate tracker** | **Tự làm** (cấu trúc dữ liệu + logic, không phải model) | Tầng 5 | #5 | KISC chủ yếu, nhưng cũng nên dùng cho UI người-lái thường |
| **Action recognition** | **Tự làm — OPTIONAL, chưa cam kết** (cần thử nghiệm trước khi build chính thức) | Tầng 3 (evaluator cho vị từ mô tả hành động) | #1 (mức hành động, chưa kênh nào che phủ) | KIS/VQA có câu hỏi dạng "người đang làm gì" |
| **Audio event detection** (tiếng khóc, mưa, còi... — KHÔNG phải lời nói) | **Tự làm — TẠM GÁC (2026-07-19)**, xem ghi chú cuối mục 2 | Tầng 2/3 (embedding kiểu CLIP nếu dùng CLAP → coarse recall; hoặc evaluator rời rạc nếu dùng classifier closed-set) | #1 (kênh âm thanh phi lời nói, chưa kênh nào che phủ — khác hẳn ASR chỉ bắt giọng nói) | KIS/VQA có câu hỏi nhắc tới âm thanh môi trường cụ thể |

---

## 2. Đi sâu từng kênh — vì sao nó nằm đúng tầng đó

### CLIPFeatures — xương sống của Tầng 2
Là embedding **toàn cảnh** của keyframe → khớp đúng với vai trò "coarse recall thiên OR/recall" đã phân tích ở lỗ hổng 2 & 4.
**Việc cần làm:** build ANN index (Faiss `IndexFlatIP` cho tập nhỏ, `IndexIVFPQ`/HNSW khi tập lớn) trên toàn bộ CLIPFeatures.
Đây là việc **rẻ nhất, nên làm đầu tiên** — nó là baseline chạy được ngay ngày 1.

### Objects — kênh duy nhất "vốn dĩ đếm được và định lượng được"
CLIP không đếm, không tách vùng; Objects (box + label + confidence) thì làm được cả hai. Vai trò kép:
- **Ở Tầng 1/2 (rẻ):** dùng làm **inverted index rời rạc** — "frame nào có label = person AND label = beach-ish scene..." — lọc thô cực nhanh trước khi động tới CLIP.
- **Ở Tầng 3 (đắt hơn):** evaluator cho `exists`, `count`; và làm **nguồn box** để crop ra Region-CLIP chấm `attribute`.

**Lưu ý dữ liệu thật:** 600 categories của Open Images khá tổng quát (Person, Man, Woman, Table, Chair, Food...) — sẽ **không có** nhãn "móc khoá gấu bông". Nghĩa là Objects tốt cho *vật thể phổ biến + đếm + vị trí*, nhưng vật thể hiếm/đặc thù vẫn phải dựa vào CLIP/region-CLIP hoặc LVLM. **Đừng kỳ vọng Objects giải hết bài toán vật thể nhỏ.**

### Metadata — trục filter rẻ nhất, tin cậy nhất, hay bị đánh giá thấp
`publish_date`, `channel`, `title`, `keywords`, `description` là **dữ liệu có cấu trúc, chính xác gần như tuyệt đối**
(khác hẳn với suy luận thị giác vốn xác suất). Đây chính là ví dụ điển hình của nguyên lý lỗ hổng 5:
*"ràng buộc cứng & tin cậy cao → dùng làm filter"*. Với dữ liệu **HTV/bản tin**, metadata cực kỳ mạnh vì:
- `title`/`description` thường **đã tóm tắt nội dung bản tin** ("Chính phủ đồng ý giảm 10% thuế nhập khẩu xăng") — gần như một caption có sẵn, miễn phí, không cần chạy model.
- `publish_date` cho lọc theo mốc thời gian nếu câu hỏi có yếu tố thời sự ("tin tức về xăng dầu giữa năm 2022").

**Việc cần làm:** build một index text đơn giản (BM25/keyword) trên metadata, chạy **trước** CLIP — vì rẻ hơn nhiều bậc và độ chính xác cao hơn nhiều so với suy luận thị giác cho các câu hỏi có yếu tố "chủ đề bản tin".

### OCR — kênh có khả năng là "vũ khí bí mật" riêng cho dữ liệu bản tin
Dữ liệu là tin tức HTV → khung hình **rất giàu chữ**: banner tên bản tin, dòng tin chạy, số liệu, tên người/địa danh
hiện dưới dạng chữ overlay (chú thích, tên khách mời...). Đây là thông tin **định danh cực mạnh** mà CLIP gần như bỏ qua
(CLIP đọc chữ trong ảnh rất tệ nếu chữ nhỏ hoặc là text overlay dày đặc).

**Việc cần làm:** chạy OCR (PaddleOCR hoặc EasyOCR, có hỗ trợ tiếng Việt) trên **Keyframes có sẵn** → build text index riêng.
**Đây là hạng mục ROI (return on investment) cao nhất trong danh sách "tự làm thêm"** — chi phí thấp (chỉ cần keyframe, không cần decode lại video), lợi ích lớn cho KIS/Textual KIS vì query nhiều khả năng nhắc tới nội dung chữ nhìn thấy trên màn hình.

### ASR — bổ sung kênh "nghe được" mà keyframe tĩnh không có
Video tin tức có lời dẫn liên tục — nhiều nội dung ngữ nghĩa **chỉ tồn tại trong lời nói**, không hiện ra hình.
**Việc cần làm:** Whisper (hoặc PhoWhisper cho tiếng Việt) chạy trên **audio track của Videos**, xuất transcript có timestamp.
Khác OCR (gắn với 1 keyframe), ASR có **track thời gian liên tục** → hữu ích để neo Tầng 4 (Temporal): biết "câu nói X xảy ra ở giây thứ mấy" giúp định vị window chính xác hơn xử lý bằng keyframe thưa.

**Đánh đổi:** ASR tốn compute hơn OCR (decode + chạy audio model trên toàn bộ video, không chỉ ảnh tĩnh) → ưu tiên **sau** OCR nếu thời gian hạn chế.

**Rủi ro hallucination khi âm thanh rè/ít lời (2026-07-15):** ASR (đặc biệt Whisper) không chỉ "im lặng" khi gặp
tiếng ồn/nhạc nền/đoạn câm — nó **tự tin sinh ra chữ nghe hợp lý nhưng sai** (hallucination), ví dụ lặp các cụm
kiểu "Thank you for watching" khi không có giọng nói thật, vì model được train trên dữ liệu có phụ đề và "đoán"
theo prior tập huấn khi tín hiệu giọng nói không rõ. Đây **tệ hơn** không có dữ liệu: text index tưởng tin cậy
(giống Metadata) nhưng lẫn rác, có thể lọc nhầm đáp án đúng mà không biết là do rác.

**Cách chống (2 lớp):**
1. **VAD (Voice Activity Detection) tiền lọc** — Silero VAD/WebRTC VAD, phát hiện đoạn thật sự có giọng nói
   trước khi đưa vào ASR; không có giọng nói → bỏ qua, không chạy ASR.
2. **Tín hiệu tin cậy có sẵn trong Whisper** — `no_speech_prob` và `avg_logprob`/`compression_ratio` mỗi
   segment; lọc bỏ segment có `no_speech_prob` cao hoặc `compression_ratio` bất thường (dấu hiệu lặp từ do
   hallucination) trước khi đưa vào index.

**Bằng chứng domain-dependent:** MSR-VTT/MSVD (clip ngắn tổng quát, nhiều clip không có ai nói) xác nhận đúng
nguyên tắc — giá trị ASR phụ thuộc hoàn toàn vào domain có narration thật hay không, cần kiểm tra trên dữ liệu
AIC 2026 thật trước khi cam kết mức đầu tư.

### Region-CLIP / object crop — mảnh ghép còn thiếu cho binding
Slide 39 (túi tím/túi trắng) chính là ví dụ cần kênh này: Objects cho box của từng cái túi, nhưng **màu** phải chấm
bằng CLIP trên **vùng crop đó**, không phải trên cả ảnh (nếu chấm cả ảnh, "tím" và "trắng" đều "có mặt" trong ảnh
mà không biết cái nào thuộc túi nào → binding fail, đúng như phân tích lỗ hổng 2).

**Việc cần làm:** với mỗi box từ Objects, crop ảnh, chạy CLIP image-encoder lên riêng crop đó, so với text "màu X + tên object".
Chi phí: rẻ (crop từ ảnh có sẵn, không cần model mới) nhưng số lượng crop có thể lớn (100 object/frame) → nên **chỉ crop theo yêu cầu** (lazy, chỉ khi query cần chấm attribute) thay vì tiền tính toàn bộ.

### Shot/Scene segmentation — dọn dẹp cho AVS và định biên cho VQA
Nhiều keyframe liên tiếp thường thuộc cùng một cảnh quay → nếu không gom, AVS có thể trả về **10 kết quả gần như trùng nhau**
(cùng 1 shot) thay vì 10 khoảnh khắc *khác nhau* — vừa lãng phí thứ hạng vừa vi phạm tinh thần "liệt kê tất cả phân cảnh".
**Việc cần làm:** cluster keyframe liền kề có CLIP similarity cao (đã có CLIPFeatures, không cần model mới) thành shot.
Với VQA, shot/scene cũng giúp xác định **đoạn video nào cần đưa cho LVLM đọc** thay vì đưa cả video dài.

### LVLM — không phải "làm hết mọi thứ", mà là verify + generate + hỏi
Ba vai trò tách bạch, **không nên gộp lại thành "hỏi LVLM mọi câu"** vì đắt và chậm:
1. **Verify quan hệ/phủ định phức tạp** ở Tầng 3, chỉ trên **shortlist rất nhỏ** (top 5-20) — vì gọi LVLM cho mọi frame trong corpus là bất khả thi (đúng lỗ hổng 4).
2. **Sinh câu trả lời VQA** ở Tầng 6, đọc lại đoạn clip đã được Tầng 4 định vị (không đọc cả video 5 tiếng).
3. **Soạn câu hỏi làm rõ cho KISC** ở Tầng 5 — cần prompt LVLM/LLM theo đúng nguyên lý information-gain (lỗ hổng 5.3): "chọn câu hỏi chia đôi tập ứng viên hiện tại", không phải hỏi ngẫu nhiên.

### Belief-state / candidate tracker — hạ tầng nhỏ nhưng bắt buộc cho KISC (và nên dùng cả cho UI thường)
Đây **không phải một model**, mà là cấu trúc dữ liệu: tập ứng viên hiện tại + điểm/trọng số + lịch sử ràng buộc đã áp.
Không có nó, mỗi lượt hỏi-đáp trong KISC sẽ **chạy lại retrieval từ đầu** thay vì *thu hẹp* tập đã có — sai hoàn toàn tinh thần lỗ hổng 5. **Việc cần làm:** một object/class đơn giản giữ `candidate_set`, `applied_filters`, `score_history`; mỗi lượt gọi `update(new_constraint)` để lọc/reweight.

### Action recognition — kênh mới, CHỈ thêm nếu tổng quát hoá được (2026-07-15)
Khác Objects/OCR (chạy tốt trên 1 keyframe tĩnh), action cần **ngữ cảnh thời gian** (chuỗi frame liên tiếp) — nếu
chỉ lưu keyframe thưa thì không đủ tín hiệu chuyển động; cần quyết định có lưu thêm cửa sổ vài frame quanh mỗi
keyframe hay không.

Domain thật của DB AIC 2026 **chưa biết** (giống vấn đề đã ghi ở lọc thô) — nên **không chọn model action-recognition
chuyên dụng** (VideoMAEv2/SlowFast/InternVideo2, pretrained Kinetics-400/700): vocabulary cố định của các model này
gắn chặt với domain huấn luyện, cùng vấn đề như Objects' 600 categories không có "móc khoá gấu bông" — chọn sai lúc
chưa biết domain rất dễ vô dụng khi domain thật lộ ra khác hẳn.

**Hướng chọn (tổng quát, không giả định domain):** VLM prompt-based — tái dùng đúng LLM đang gọi để caption
(`agent/llm.py`), mở rộng prompt để mô tả hành động thay vì chỉ mô tả cảnh, gửi vài frame liên tiếp thay vì 1 ảnh
nếu cần tín hiệu chuyển động. Đây là bản mở rộng tự nhiên của nguyên tắc đã áp cho Object detection (VERGE 2026:
VLM tổng quát trước, chuyên biệt hoá sau khi có bằng chứng từ validation set thật) — **chưa cam kết build**, chỉ
ghi nhận là hướng khả thi khi có DB thật để thử nghiệm.

### Audio event detection — kênh mới, phi lời nói, CHỈ thêm nếu tổng quát hoá được (2026-07-15)
Khác ASR (chỉ bắt được **lời nói**), rất nhiều tín hiệu âm thanh có nghĩa lại là **phi ngôn ngữ** — tiếng khóc,
tiếng mưa, tiếng còi xe, tiếng vỗ tay... ASR bỏ qua hoàn toàn các tín hiệu này vì nó không phải giọng nói. Đây là
1 kênh dữ liệu riêng biệt, không phải phần mở rộng của ASR.

**2 hướng model, cùng vấn đề vocabulary như Action recognition:**
- **Closed-vocabulary** (PANNs, YAMNet, AST — pretrained AudioSet 527 nhãn): cho nhãn rời rạc, dùng làm inverted-index được ngay, nhưng gắn chặt với 527 nhãn AudioSet — domain thật có thể cần nhãn khác.
- **Open-vocabulary, zero-shot** (**CLAP** — Contrastive Language-Audio Pretraining, kiến trúc y hệt CLIP nhưng
  cho audio: encode text + encode audio, cosine similarity): không giới hạn taxonomy cố định, hỏi được bất kỳ
  mô tả âm thanh nào. Về kiến trúc có thể tái dùng nguyên pattern `channels/clip.py` (embedding + ANN +
  `score_all(text)`) cho 1 channel `audio_clap` mới — rẻ để thử vì hạ tầng đã có sẵn mẫu.

**Hướng chọn (tổng quát, không giả định domain):** CLAP, cùng lý do đã áp cho Action recognition — domain AIC
2026 chưa biết, tránh cam kết vào 1 taxonomy cố định (PANNs/YAMNet) trước khi biết domain thật cần gì.

**Cảnh báo domain-dependent (giống ASR):** nếu domain là tin tức phòng thu (giả định ban đầu dựa trên dữ liệu
cũ), âm thanh chủ yếu là nhạc hiệu/background music, ít sự kiện âm thanh môi trường thật → giá trị thấp, dễ bị
nhiễu bởi nhạc nền. MSR-VTT/MSVD cũng không khả thi (clip ngắn tổng quát, hiếm sự kiện âm thanh nổi bật). Nên
chạy **VAD trước** (xem mục ASR ở trên) để tách vùng non-speech, chỉ chạy audio-event model trên đúng vùng đó —
tránh nhầm lẫn với giọng nói chồng lấp. **Chưa cam kết build**, chỉ ghi nhận hướng khả thi khi có DB thật.

### TẠM GÁC cả ASR lẫn Audio event detection (2026-07-19)

Domain AIC 2026 đã xác nhận là **CCTV/giám sát công cộng + lifelogging (thiết bị đeo)** (xem xác nhận BTC ở
trên) — hoá ra đây là **đúng 2 loại nội dung khó nhất cho mọi kênh dựa trên audio**, không phải may rủi:

- **CCTV/giám sát công cộng:** mic (nếu có) rẻ, đặt xa nguồn âm → tín hiệu yếu, lẫn tiếng xe cộ/gió/đám đông
  thành nền ồn liên tục. **Cần xác minh trước khi thiết kế thêm:** nhiều hệ CCTV **tắt hẳn audio** vì lý do
  pháp lý/quyền riêng tư ở nơi công cộng — có khả năng thật là phần lớn video giám sát trong bộ dữ liệu
  **không có track audio nào** để phân tích.
- **Lifelogging (thiết bị đeo):** mic gắn trên người → tiếng bước chân/hơi thở/va quẹt vải áo **to hơn hẳn**
  âm thanh môi trường muốn phát hiện (hiệu ứng "close-mic"); gió gần như luôn hiện diện (không có màng chắn
  gió, hoạt động ngoài trời là domain chính).

**Failure mode nguy hiểm hơn "điểm thấp":** cả ASR (hallucination, đã ghi ở trên) lẫn audio-event zero-shot
đều có xu hướng **tự tin gán nhầm nhiễu thành nhãn nghe hợp lý** khi SNR thấp (tiếng ồn băng rộng dễ bị gán
nhầm "mưa rơi"/"vỗ tay") — không phải model "không biết", mà là **tự tin sai**, nguy hiểm hơn hẳn im lặng.

**Quyết định:** tạm gác khai thác cả 2 kênh — không đầu tư thêm (VAD-gating, margin-threshold, denoising...)
cho tới khi có mẫu audio thật từ BTC để **nghe thử** mức độ nhiễu thực tế, thay vì thiết kế mù. Nếu mẫu thật
cho thấy tín hiệu đủ sạch ở 1 phần dữ liệu (vd đoạn lifelog trong nhà, ít gió), có thể mở lại có chọn lọc thay
vì áp dụng toàn corpus.

---

## 3. Đường đi riêng theo từng loại query (cùng hạ tầng, khác cách ghép)

| Query type | Đường đi qua các tầng | Tầng nào quyết định thắng-thua |
|---|---|---|
| **KIS (Video/Textual)** | Filter (Metadata) → Coarse recall (CLIP) → **Fine verify nặng** (Objects+OCR+Region-CLIP+LVLM) → chốt Top-1 | **Tầng 3** — vì metric là Top-1/Top-5, sai một ly cũng thua |
| **AVS** | Filter nhẹ → Coarse recall rộng (K lớn) → Fine verify **vừa phải** (đủ để rank đúng, không cần verify sâu từng cái) → **Temporal/shot dedup** → xuất ranked list | **Tầng 2 (recall) + Tầng 4 (dedup)** — vì metric quan tâm cả tập, không chỉ 1 điểm |
| **VQA** | Localize đoạn liên quan (Coarse recall + Temporal) → cắt clip → đưa **LVLM sinh câu trả lời** | **Tầng 6** — retrieval chỉ là bước chuẩn bị ngữ cảnh, chất lượng câu trả lời phụ thuộc LVLM |
| **KISC** | Vòng lặp: Filter/Coarse recall ban đầu (mơ hồ, tập lớn) → **Tầng 5**: chọn câu hỏi info-gain cao → nhận trả lời → cập nhật belief-state → lặp lại đến khi tập đủ nhỏ → Fine verify chốt | **Tầng 5** — thắng thua nằm ở chất lượng câu hỏi chọn, không phải encoder |

---

## 4. Thứ tự ưu tiên xây dựng (theo ROI, không phải theo độ "cool")

Nối với nguyên lý lỗ hổng 4 ("thiết kế ngược từ metric") — áp cho *thứ tự build* trong giai đoạn chuẩn bị:

| Ưu tiên | Việc | Vì sao |
|---|---|---|
| **1 — Phải có ngày đầu** | Faiss/ANN index trên CLIPFeatures có sẵn + UI hiện Top-K | Baseline chạy được; không có cái này thì không có gì để cải tiến |
| **2 — Rẻ, lợi ngay** | Filter theo Metadata (BM25 trên title/description/date) trước khi vào CLIP | Gần như miễn phí (dữ liệu có sẵn, chỉ cần index text), độ chính xác filter rất cao |
| **3 — ROI cao nhất trong "tự làm"** | OCR trên Keyframes có sẵn → text index riêng | Dữ liệu bản tin cực giàu chữ; chi phí thấp (không cần decode video); giúp KIS trực tiếp |
| **4 — Nền cho compositional scoring** | Dùng Objects làm evaluator cho exists/count + inverted-index lọc thô | Đã có sẵn dữ liệu detection, chỉ cần build logic combine (min/weighted) |
| **5 — Dedup cho AVS** | Shot/scene clustering trên CLIPFeatures đã có | Rẻ (không cần model mới), tránh mất điểm AVS vì trả trùng lặp |
| **6 — Nâng cấp binding** | Region-CLIP crop theo box (lazy, chỉ khi cần) | Cần thêm code nhưng không cần model mới; giải quyết đúng lỗi binding đã phân tích |
| **7 — Đắt, làm sau** | Tích hợp LVLM cho verify + VQA + soạn câu hỏi KISC | Cần gọi API/model lớn, latency cao — chỉ nên chạy trên shortlist rất nhỏ |
| **8 — Hạ tầng tương tác** | Belief-state tracker + UI feedback + hiển thị đa dạng (MMR) | Quan trọng về lâu dài (điểm 5) nhưng có thể bắt đầu đơn giản (chỉ giữ candidate set) rồi nâng cấp dần |
| **TẠM GÁC** | ASR trên toàn bộ Videos, Audio event detection (CLAP) | Domain CCTV+lifelogging xác nhận là loại audio khó nhất (SNR thấp, có thể không có track audio) — chờ nghe mẫu thật trước khi đầu tư tiếp, xem mục 2 |

> **Ghi chú:** thứ tự này giả định thời gian chuẩn bị có hạn. Nếu có thể chạy song song (nhiều người trong đội),
> việc 2-3-4-5 độc lập nhau và có thể làm cùng lúc; việc 6-7-8 phụ thuộc vào việc 1 (cần pipeline coarse recall chạy trước để có shortlist mà verify).

---

## 5. Một câu tóm bản đồ

> **Dữ liệu có sẵn (CLIPFeatures, Objects, Metadata) dựng được Tầng 1-2 (filter + coarse recall) gần như miễn phí.
> Chính "tự làm thêm" (OCR, Region-CLIP, shot-seg) mới lấp được Tầng 3-4 (compositional + temporal) —
> và đó là nơi phân biệt đội khá với đội giỏi, vì baseline CLIP-only ai cũng có. ASR và Audio event detection
> tạm gác (2026-07-19) — domain CCTV+lifelogging xác nhận là loại audio khó nhất, chờ nghe mẫu thật trước khi
> đầu tư tiếp.**
