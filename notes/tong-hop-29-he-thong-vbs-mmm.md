# Tổng hợp 29 hệ thống VBS/MMM (2025-2026) — Đối chiếu với khung đã dựng

> Đọc toàn bộ 29 paper trong `papers/` (VBS 2025/2026, MMM 2025/2026) qua 5 agent song song.
> Nối tiếp `doi-chieu-thuc-te-cac-doi-vo-dich.md`, `prak-v4-vo-dich-vbs-2026.md`, và toàn bộ chuỗi note trước.
> Mục tiêu: lọc ra ý tưởng **thật sự mới** chưa có trong khung, và **xác nhận thêm** những gì đã đúng.
> Ngày: 2026-07-09.

---

## 0. Danh sách nguồn

DiveXplore · Exquisitor 2025/2026 · Fusionista/2.0 · H-EAGLE · HORUS · IMSearch 2.0 · Interactive Video
Search with mLLM · MediaMix 2025/2026 · NII-UIT 2025/2026 · PraK V3/V4 · SnapMind · SnapSeek 2.0 · TapesVRy ·
U-Cker · VEAGLE · VERGE 2025/2026 · VideoEase · ViewsInsight2.0 · ViFi · VIREO · vitrivr-Engine 2025/2026 ·
vitrivr-VR. Tất cả đều là hệ thống dự thi VBS (Video Browser Showdown) hoặc anh em họ hàng (CBMI IVR4B),
cùng dạng bài với AIC (KIS/AVS/VQA + biến thể temporal).

---

## 1. Bảng tra nhanh — 29 hệ thống, ý tưởng nổi bật nhất mỗi hệ

| Hệ thống | Năm | Encoder chính | Ý tưởng nổi bật nhất |
|---|---|---|---|
| DiveXplore | VBS25 | OpenCLIP ViT-H/14 | Phân tích định lượng "query của expert vs máy" → rút quy tắc viết query tốt |
| Exquisitor 25 | VBS25 | CLIP | Rank aggregation 2 chiều giữa conversational search ↔ relevance feedback |
| Exquisitor 26 | VBS26 | CLIP + eCP index | **Sequence-chain temporal**: nối chuỗi theo video, RRF tie-break |
| Fusionista | VBS25 | Multi-CLIP variants | Temporal search bằng **dynamic programming** (max-sum similarity, O(n·f)) |
| Fusionista 2.0 | VBS26 | CLIP ensemble (α=0.7) | **Interactive confirmation rerank**: LLM tự sinh câu hỏi yes/no, VLM trả lời để rerank |
| H-EAGLE | VBS26 | SigLIP+NVemb 3-tầng | **Index phân cấp K=1(frame)/2(shot)/3(narrative action)** — chọn mức trừu tượng để query |
| HORUS | VBS25 | CLIP+IRRA+LaBSE+VideoLLaMA | IRRA sửa lỗi misbind attribute-object của CLIP; weighted fusion có slider |
| IMSearch 2.0 | VBS25 | **ALIGN** (thắng CLIP/BLIP/JINA) | Test trên **"Ho Chi Minh AI City Challenge dataset"** — xem mục 5 |
| Interactive+mLLM | MMM25 | IITV (concept-bank) | Concept-distribution histogram/wordcloud từ top-1000 để refine query không cần xem video |
| MediaMix 25/26 | VBS25/26 | CLIP+DINO(v2) | Browse trong VR (Vision Pro): "globe" 3D, nhiều globe song song, sắp theo t-SNE |
| NII-UIT 25 | VBS25 — **vô địch** | BEiT-3+OpenCLIP | Dynamic Temporal Search + Stable Diffusion sinh ảnh truy vấn (đã biết) |
| NII-UIT 26 | VBS26 | **SigLIP** | 3 module VQA mới: Answer Span Prediction, Candidate Answer Suggestion, In-Video Retrieval |
| PraK V3/V4 | VBS25/26 — **vô địch V4** | CLIP fine-tune | Spatial conjunction, hybrid layout FLAS, Bayesian display-only (đã biết đầy đủ) |
| SnapMind | VBS26 | OpenCLIP+Elasticsearch | **LLM Planner + 3 mức tự chủ (Guide/Assist/Auto)** + audit trail — xem mục 2.5 |
| SnapSeek 2.0 | VBS25 | OpenCLIP+BLIP-2 | Sketch/drag-drop object+color+pose lên lưới 20×20; "Magic Brush" tự hoàn thiện contour |
| TapesVRy | VBS26 | CLIP+LLM label | Browse 360° VR dạng "video universes" theo cụm chủ đề do LLM đặt tên |
| U-Cker | VBS26 | OpenCLIP CoCa | **Exact search** (không ANN) trên GPU memory; Qwen3-8B 1-prompt làm 3 việc (sửa lỗi+dịch+paraphrase) |
| VEAGLE | VBS25 | Milvus semantic | **Eye-tracking**: phát hiện ảnh đúng bị "bỏ sót" qua gaze duration, tự re-suggest |
| VERGE 25/26 | VBS25/26 | GoogleNet→VLM rerank | 2026: thay ensemble detector chuyên biệt bằng **1 VLM (Qwen2.5-VL) tổng quát** cho mọi concept |
| VideoEase | VBS25 | CLIP+BLIP2+OpenCLIP | So sánh trọng số fusion vs RRF; OpenCLIP thắng riêng lẻ |
| ViewsInsight2.0 | VBS25 | CLIP (DFN-5B) | Auto Query Generator: Llama3.1(temporal)+Stanza(NER)+T5(paraphrase) từ 1 đoạn mô tả dài |
| ViFi | VBS25 | **SigLIP** | Temporal "now-and-then" (max similarity trong k-frame kế tiếp) |
| VIREO | VBS26 | Milvus dense+sparse | **Object-sketch + "recommendation shading"**: precompute rank list theo (ô lưới × object class) |
| vitrivr-Engine 25 | MMM25 | CLIP | Feature-driven segmentation (mỗi feature tự có ranh giới shot riêng); SVM/hyperplane relevance feedback |
| vitrivr-Engine 26 | MMM26 | OpenCLIP+DINOv2 | **Modality cảm xúc mới**: facial+text+speech emotion embedding, nearest-neighbor theo cảm xúc |
| vitrivr-VR | VBS25 | OpenCLIP | VR hợp nhất input+browse trên Vision Pro, bàn phím vật lý + speech-to-text on-device |

---

## 2. Ý tưởng THẬT SỰ MỚI — theo từng lỗ hổng đã dựng

### 2.1. Lỗ hổng 1 (Representation) — encoder & keyframe extraction

**Encoder:** SigLIP đang nổi lên như bản nâng cấp CLIP (NII-UIT 2026, ViFi) — được nêu lý do cụ thể: **giảm
"semantic drift"**, alignment tốt hơn với query có thuộc tính tinh vi. CLIP huấn luyện trên **DFN-5B**
(ViewsInsight2.0 — lọc tự động từ 43 tỷ xuống 5 tỷ cặp) là một lựa chọn khác đáng thử. Xu hướng chung: **không
ai chỉ dùng 1 CLIP gốc** — hầu hết chạy 2-3 encoder song song rồi fusion (VideoEase: CLIP+BLIP2+OpenCLIP;
Fusionista: nhiều biến thể CLIP).

**Keyframe extraction — đa dạng hơn hẳn dự đoán ban đầu, không có "chuẩn" chung:**
- **Self-Similarity Matrix + Kernel Temporal Segmentation** (VIREO 2026): dựng ma trận tương đồng giữa các
  frame bằng BLIP embedding, tính "novelty signal" qua checkerboard kernel, chọn điểm cắt candidate, refine
  bằng KTS — phức tạp hơn TransNetV2 nhưng giải quyết đúng vấn đề "cắt shot đều không hợp với video ngắn/dài"
  (chính là hạn chế VIREO tự nhận ở bản trước).
- **Feature-driven segmentation** (vitrivr-Engine 2025): mỗi loại feature (CLIP, màu...) có ranh giới shot
  **riêng** — vì "similarity notion" của mỗi feature khác nhau, ranh giới cố định chung dễ lệch.
- **Hierarchical shot→subshot→VSS** (VERGE 2026): TransNetV2 (shot) → DCT-based subshot → Very Similar
  Segments → chọn frame sắc nét nhất (variance of Laplacian) → dedup bằng khoảng cách Euclidean ResNet152.
- **Semi-automatic + human review** (IMSearch 2.0): ALIGN cosine similarity ngưỡng 0.97 để nhóm frame trùng,
  nhưng vẫn có bước admin duyệt tay — thừa nhận tự động hoàn toàn chưa đủ tin cậy.

> **Bài học:** không nên coi keyframe extraction là bài toán đã giải xong bằng 1 thuật toán cố định — đáng
> thử nghiệm 2-3 cách trên chính dữ liệu AIC và đo bằng Recall@K (đúng tinh thần `bo-metric-va-validation-set.md`).

**H-EAGLE's 3-tầng trừu tượng — bổ sung quan trọng cho tư duy "đa mức hạt" đã có:**
Thay vì chỉ liệt kê các *kênh* riêng biệt (scene/object/OCR/ASR như đã làm), H-EAGLE tổ chức chúng thành
**thứ bậc tường minh K=1 (frame) → K=2 (shot, encode bằng video encoder) → K=3 (narrative action trải dài
nhiều shot, VLM sinh mô tả rồi align)**, và **để người dùng chọn mức trừu tượng để query**. Search 2 giai
đoạn: Segment Retrieval (thu hẹp không gian theo cấp số nhân ở K=2/3) → Detailed Selection (tìm cục bộ trong
nhóm top-rank ở K=1). Đây là một cách **tổ chức lại** ý tưởng multi-granularity đã có, chặt chẽ hơn — đáng
cân nhắc khi thiết kế lại kiến trúc index.

**Vintern-1B-v3.5** (nhắc lại từ Fusionista2.0): OCR fine-tune tiếng Việt, xử lý được ký tự mờ/che khuất — cụ
thể hoá đúng khuyến nghị "OCR ROI cao nhất" đã ghi ở `ban-do-du-lieu-pipeline.md`.

### 2.2. Lỗ hổng 2 (Compositional/Binding) — ý tưởng rẻ hơn hẳn cách đã phác thảo

**VIREO's "object-sketch" + "recommendation shading" — đáng chú ý nhất trong toàn bộ 29 paper cho lỗ hổng 2:**
mỗi cặp (ô lưới × object class COCO) là một "atomic query" được **precompute rank list offline** (dùng
confidence detector + IoU, shot signature bằng max-pooling). Khi người dùng tương tác với canvas lưới, hệ
thống hiển thị **"recommendation shading"** — độ đậm mỗi ô ước lượng **số kết quả sẽ trả về** *trước khi*
người dùng submit — và cập nhật rank tức thời (late-fusion) khi chọn object+cell.

→ Đây là bản hiện thực **chính xác** nguyên tắc đã đặt ra ở `lop-tuong-tac-thuc-dung-kiem-soat-duoc.md`
("tính trước lúc indexing, runtime chỉ tra cứu") — nhưng cụ thể hoá thêm một bước: không chỉ tính trước facet
mà còn **cho xem trước ước lượng kết quả** khi rê chuột, trước khi cam kết click. Đáng bổ sung vào thiết kế
UI: shading/preview số lượng kết quả trên từng ô lưới trước khi submit.

**VERGE 2026 thay ensemble detector chuyên biệt bằng 1 VLM tổng quát (Qwen2.5-VL 7B):** thay vì huấn luyện/
maintain nhiều detector riêng cho từng loại concept (>2000 khái niệm ở bản 2025), dùng **prompt tự nhiên** cho
VLM để sinh confidence score cho ~50 concept mục tiêu. Đánh đổi: mất độ chính xác chuyên biệt, nhưng giảm hẳn
chi phí kỹ thuật (không cần train/maintain nhiều model). Đáng cân nhắc cho MVP: bắt đầu bằng VLM prompt-based
detection, chỉ chuyên biệt hoá khi đã có bằng chứng (từ validation set) rằng nó không đủ chính xác.

### 2.3. Lỗ hổng 3 (Temporal) — 5 thuật toán khác nhau, hội tụ về 1 nguyên lý chung

Toàn bộ 5 cách tiếp cận nhìn khác nhau bề ngoài nhưng đều theo đúng khung "tìm anchor trước, ràng buộc pool
theo sau, rồi hợp điểm" đã đặt ra ở lỗ hổng 2.5 — khác nhau ở *công thức hợp điểm cụ thể*:

| Hệ thống | Công thức/cơ chế |
|---|---|
| Exquisitor 2026 | Top-r (r=1000) query đầu → pool segment sau đó cùng video → chạy tiếp theo batch → xếp theo **độ dài chuỗi**, RRF (k=60) tie-break, lọc trùng bằng IoU |
| NII-UIT (25→26) | "Dynamic": xét *các shot xung quanh* kết quả trước (không chỉ chặt trước/sau), re-rank theo độ liên quan mới |
| ViewsInsight2.0 | 3 input (main+before+after) → score "now" = main + max(score(before), score(after)) cùng video; có **event-level filtering** để dedup |
| ViFi | "now-and-then": present similarity + max similarity trong k-frame *kế tiếp*, trung bình có trọng số |
| Fusionista | Dynamic programming tối đa hoá tổng similarity qua thứ tự nhiều query, O(n·f) |

→ **Bài học tổng quát:** không có 1 công thức "đúng nhất" — tất cả đều là biến thể của "ràng buộc thời gian +
hợp điểm nhiều truy vấn con". Công thức của MERVIN (`S = 10·S_pair + 5·(S̄1+S̄2)`, đã ghi ở
`doi-chieu-thuc-te-cac-doi-vo-dich.md`) là bản đơn giản nhất trong nhóm này — vẫn dùng được làm khởi điểm,
nâng cấp dần theo hướng NII-UIT (xét shot xung quanh, không chỉ trước/sau chặt) nếu có thời gian.

### 2.4. Lỗ hổng 4 (Recall/Verify & thiết kế theo metric)

**U-Cker chọn exact search (inner-product chính xác) thay vì ANN**, giữ toàn bộ embedding trong GPU memory —
đánh đổi bộ nhớ lấy độ tin cậy tuyệt đối, khả thi vì quy mô V3C (~4 triệu keyframe × 768 chiều) vẫn vừa GPU
hiện đại. Đáng cân nhắc **nếu** quy mô dữ liệu AIC đủ nhỏ để làm tương tự — tránh hẳn rủi ro "ANN làm rớt đáp
án đúng" (đúng nỗi lo đã nêu về "cái trần" của coarse recall ở `bo-metric-va-validation-set.md`).

**Exquisitor 2026 chuyển từ Product Quantization sang eCP index (disk-based)** để giữ độ chính xác cho
text-query/feedback mà vẫn tiết kiệm bộ nhớ hơn giữ toàn bộ trong RAM — phương án trung gian giữa ANN xấp xỉ
và exact search toàn RAM.

### 2.5. Lỗ hổng 5 (Interaction) — phát hiện quan trọng nhất của toàn bộ đợt đọc paper này

**SnapMind's "3 mức độ tự chủ" (Guide / Assist / Auto) — xác nhận trực tiếp cuộc tranh luận vừa bàn hôm nay
về ranh giới agent nên tự quyết tới đâu:**
- **Guide**: user tự thực thi từng bước (thấp nhất, kiểm soát tối đa).
- **Assist**: user chọn bước nào chạy (trung bình).
- **Auto**: approve toàn bộ plan cùng lúc (tự động cao nhất).

Kiến trúc nền: LLM Planner sinh **plan** (chuỗi lệnh gọi từ **Component Registry cố định**: text/image/OCR/
color/object/ADL) — Planner **chỉ được chọn trong registry đóng**, không tự sáng tác thao tác mới — **chính
xác** nguyên tắc "tập lệnh đóng" đã thiết kế ở `kien-truc-2-tang-agent-va-ui.md`. Thêm 2 cơ chế minh bạch quan
trọng chưa có trong thiết kế của mình:
- **`source_contrib` vector**: giải thích mỗi item được xếp hạng cao *nhờ đóng góp của modality nào* (text bao
  nhiêu %, object bao nhiêu %...) — một dạng "giải thích được" (explainability) cụ thể hơn hẳn.
- **Audit trail đầy đủ**: log toàn bộ plan, tham số, ranked list từng bước, và mọi chỉnh sửa của user — phục
  vụ reproducibility (quan trọng khi cần xem lại tại sao hệ thống trả về kết quả này).
- **Early stopping** khi ranking đã ổn định (đo bằng Jaccard similarity + ΔNDCG@K giữa 2 bước liên tiếp) —
  cách tự động biết "khi nào nên dừng tinh chỉnh thêm", một cơ chế còn thiếu trong thiết kế `BeliefState` hiện tại.

→ **Nên bổ sung vào `kien-truc-2-tang-agent-va-ui.md`:** (1) thêm khái niệm mức độ tự chủ tường minh (đặt tên
theo đúng SnapMind luôn, dễ tham chiếu), (2) thêm `source_contrib` vào mỗi Action trả về, (3) thêm điều kiện
dừng dựa trên độ ổn định ranking (không chỉ dựa vào ngưỡng entropy/top1-top2 margin đã có).

**VEAGLE's eye-tracking implicit feedback** — một kênh phản hồi hoàn toàn mới, chưa có trong khung: đo thời
gian nhìn (gaze duration) trên từng ảnh, phát hiện ảnh **điểm cao nhưng bị user "lướt qua"** (nhìn quá ngắn),
tự động re-surface. Không khả thi nếu không có thiết bị eye-tracking, nhưng ý tưởng tổng quát hoá được: **bất
kỳ tín hiệu hành vi ngầm nào (thời gian dừng chuột, thời gian trước khi cuộn qua) đều có thể dùng làm feedback
yếu**, không cần đợi click tường minh — đáng ghi nhận như một hướng mở rộng tương lai cho `phac-thao-belief-state-kisc.md`.

**MediaMix (VR, 3D globe + t-SNE layout)** và **TapesVRy (360° VR "universes")**: cả 2 đều là biến thể 3D/VR
của đúng nguyên lý "similarity-preserving layout để quét nhanh" (PraK V4's FLAS) — xác nhận thêm lần nữa rằng
đây là một nguyên lý được nhiều nhóm độc lập hội tụ về, không phải ý tưởng lẻ tẻ của 1 hệ thống.

---

## 3. Xác nhận thêm (không mới, nhưng củng cố các quyết định đã có)

- **OCR + ASR gần như phổ biến ở mọi hệ thống 2025-2026** (PaddleOCR/Vintern cho OCR; Whisper/faster-whisper/
  wav2vec2 cho ASR) — củng cố mạnh quyết định ưu tiên cao ở `ban-do-du-lieu-pipeline.md`.
- **Multi-encoder ensemble + fusion là chuẩn mực, không phải ngoại lệ** — không hệ thống nào chỉ dùng 1 CLIP.
- **Query expansion bằng LLM phổ biến** (GPT-4o, Qwen, Llama) — xác nhận hướng đã ghi ở lỗ hổng 5.5.
- **Rocchio/relevance feedback dạng nào đó có mặt ở hầu hết hệ thống** (Bayesian, SVM, Ide Regular, rank
  aggregation) — xác nhận lựa chọn Rocchio-style đã đưa vào `phac-thao-belief-state-kisc.md`.
- **Novice/Advanced mode tách biệt** (VERGE, NII-UIT, VideoEase) — nhiều hệ thống độc lập đi đến cùng giải
  pháp "ẩn bớt tính năng nâng cao cho người mới" — củng cố hướng "AI gợi ý là tuỳ chọn, không mặc định".

---

## 4. Phát hiện đặc biệt — IMSearch 2.0 test trên dữ liệu giống AIC

Paper IMSearch 2.0 nêu bảng Recall@1/5/10/20 trên **"test 35-query (Ho Chi Minh AI City Challenge dataset)"**:

| Encoder | R@1 | R@5 | R@10 | R@20 |
|---|---|---|---|---|
| **ALIGN** | **0.29** | **0.49** | **0.63** | **0.69** |
| CLIP | 0.31 | 0.43 | 0.46 | 0.49 |
| BLIP | 0.26 | 0.31 | 0.37 | 0.43 |
| JINA | 0.26 | 0.31 | 0.40 | 0.54 |

ALIGN thắng rõ ở R@5/10/20 dù thua nhẹ CLIP ở R@1. Đây là **dữ liệu thực nghiệm gần nhất với chính bài toán
của bạn** trong toàn bộ 29 paper — đáng thử ALIGN như một ứng viên encoder thay thế/bổ sung CLIP khi build
pipeline thật, và tự đo lại trên chính validation set của mình để xác nhận (không nên tin số liệu chéo dataset
mù quáng — nhưng đây là gợi ý khởi điểm mạnh).

---

## 5. Cập nhật bảng ưu tiên xây dựng (bổ sung cho `ban-do-du-lieu-pipeline.md`)

| Việc | Điều chỉnh sau khi đọc 29 paper |
|---|---|
| Encoder | Thử thêm **ALIGN** và **SigLIP** cạnh CLIP — không dừng ở 1 lựa chọn, đo trên validation set riêng |
| Keyframe extraction | Không tin baseline duy nhất — thử tối thiểu 2 cách (uniform interval vs SSM/TransNetV2) và đo Recall@K |
| Compositional/binding | Ưu tiên **object-sketch kiểu VIREO** (precompute rank list theo ô lưới×object, có shading ước lượng kết quả) trước khi xây bộ máy parse-evaluate-combine đầy đủ — rẻ hơn, đã có bằng chứng dùng được |
| Agent layer | Thêm tường minh **3 mức tự chủ (Guide/Assist/Auto)** kiểu SnapMind vào `kien-truc-2-tang-agent-va-ui.md`, thêm `source_contrib` + audit trail + early-stopping theo NDCG |
| Temporal | Khởi điểm bằng công thức MERVIN (đơn giản nhất), nâng cấp dần theo hướng NII-UIT (xét ngữ cảnh xung quanh) nếu có thời gian |
| Concept/object detection | Cân nhắc VLM prompt-based (Qwen2.5-VL style) thay vì train nhiều detector chuyên biệt — làm MVP nhanh hơn |

---

## 6. Giới hạn của việc tổng hợp này

- Đọc qua 5 agent song song, mỗi agent đọc PDF độc lập — có thể sai sót nhỏ trong trích xuất số liệu, nên coi
  các con số ở đây là "định hướng", không phải trích dẫn nguyên văn chính xác tuyệt đối; kiểm tra lại paper gốc
  trong `papers/` nếu cần trích dẫn chính xác.
- 29 hệ thống đều là hệ thống **dự thi/thiết kế**, phần lớn paper viết **trước khi thi đấu** — số liệu recall/
  rank thực tế tại VBS2025/2026 (ai thắng, ai thua) không nằm trong các paper này (trừ vài chỗ tự nhận đã từng
  đoạt giải ở kỳ trước).
- Không phải mọi ý tưởng đều áp dụng được cho dữ liệu AIC (một số dành riêng cho MVK/LapGynLHE/VR) — đã lọc bớt
  các phần quá đặc thù domain đó, giữ lại phần tổng quát hoá được.
