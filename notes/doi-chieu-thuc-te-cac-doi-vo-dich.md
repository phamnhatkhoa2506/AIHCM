# Đối chiếu thực tế — Các đội vô địch/mạnh làm gì

> Kiểm chứng khung 5 lỗ hổng (và các phác thảo cụ thể) với hệ thống thật của các đội thi thật.
> Nối tiếp toàn bộ chuỗi note trước. Ngày: 2026-07-07.

---

## 0. Nguồn đã khai thác — và mức độ tin cậy

| Đội/Hệ thống | Cuộc thi | Vị trí | Mức độ chi tiết lấy được |
|---|---|---|---|
| **NII-UIT** — *"Multimodal Video Retrieval with LLM Integration and Dynamic Temporal Search"* | **VBS 2025** (quốc tế, Nara, Nhật, 17 đội) | **Vô địch (top scorer)** | Chỉ có abstract — bài full-text bị chặn đăng nhập (Springer/ACM) |
| **Dionysus** (ĐH Công nghệ Thông tin — ĐHQG TP.HCM) | **AIC HCMC 2023** | **Vô địch (giải nhất)** | Chỉ có tin báo chí, không có báo cáo kỹ thuật |
| **MERVIN** (team "chmod", ĐH Khoa học Tự nhiên — ĐHQG TP.HCM) | **AIC HCMC 2025** | Top các đội đại học (79/88 điểm vòng loại, vào chung kết) — *không rõ có phải giải nhất tuyệt đối không* | **Full-text đầy đủ**, đây là nguồn chi tiết nhất |

**Phát hiện thú vị:** hai thành viên đầu trong danh sách tác giả NII-UIT ("Bao Tran Gia", "Tuong Bui Cong Khanh") khớp tên rất sát với hai thành viên Dionysus ("Trần Gia Bảo", "Bùi Công Khánh Tường") — nhiều khả năng là cùng một nhóm nòng cốt, đi từ vô địch AIC trong nước (2023) lên vô địch VBS quốc tế (2025). Điều này nói lên: **kinh nghiệm từ chính sân chơi AIC là nền tảng trực tiếp dẫn tới thành công quốc tế** — không cần lý thuyết xa vời, giải AIC tốt là con đường thật.

**Trùng hợp thêm:** MERVIN đến từ ĐH Khoa học Tự nhiên — **chính là đơn vị SELab tổ chức buổi tập huấn** mà toàn bộ chuỗi note này bắt nguồn.

---

## 1. Đối chiếu trực tiếp với khung 5 lỗ hổng

### Lỗ hổng 1 (Representation granularity) — **xác nhận + bổ sung 1 mức hạt mới**

MERVIN xây **3 index song song**: keyframe (visual), transcript (theo đoạn 5-segment), và **video summary** (tóm tắt sự kiện toàn video do Gemini 1.5 Flash sinh ra). Đây khớp đúng nguyên lý "biểu diễn đa mức độ" — nhưng có một mức hạt mình **chưa từng liệt kê**: **mức video/sự kiện tổng thể**, thô hơn cả `scene`. Query kiểu "tìm clip về hoạt động thu gom pin ở Úc sau khi 3 nhà khoa học đoạt Nobel..." (ví dụ thật trong paper) mang tính *tường thuật sự kiện* hơn là một khung hình cụ thể — mức `scene`/`object` không đủ, cần một tầng tóm tắt ngữ nghĩa cấp video.

**Bổ sung quan trọng khác — encoder không nhất thiết phải là CLIP gốc:**
- MERVIN dùng **PE-Core-bigG-14-448** (Meta Perception Encoder) cho visual — họ báo cáo vượt cả CLIP ViT-H/14 lẫn OpenCLIP ViT-bigG/14 ở zero-shot, và có pretraining trên dữ liệu video tổng hợp giúp biểu diễn "motion-consistent" hơn — tức đã có sẵn một ít "cảm nhận thời gian" ngay ở tầng encoder, đỡ việc cho lỗ hổng 3.
- **Quan trọng nhất cho dữ liệu tiếng Việt:** MERVIN **không dùng text tower của CLIP** (vốn huấn luyện chủ yếu tiếng Anh) mà dùng **model embedding tiếng Việt chuyên biệt** (`dangvantuan/vietnamese-embedding`), chọn dựa trên benchmark STS tiếng Việt — họ so sánh hẳn 5 model khác nhau trước khi chọn (bảng trong paper). Đây là chi tiết **mình đã bỏ sót hoàn toàn** trong các note trước: dữ liệu HTV/YouTube tiếng Việt cần một **text encoder tiếng Việt riêng cho kênh transcript/summary**, tách biệt khỏi CLIP text tower (vốn chỉ nên dùng cho query gốc nếu multilingual CLIP đủ tốt, hoặc cần kiểm tra riêng).

### Lỗ hổng 2 (Compositional scoring) — **xác nhận một phần, nhưng bị "vượt mặt" bởi một cách tiếp cận rẻ hơn**

Có xác nhận: tên hệ thống "Fustar: **divide and conquer** query in video retrieval system" (SOICT 2024) — chính là ý tưởng phân rã query thành phần nhỏ hơn để xử lý, đúng tinh thần lỗ hổng 2. NII-UIT cũng có "**object filtering**" như một bước lọc riêng — khớp với việc dùng Objects làm evaluator riêng cho `exists`/`count`.

**Nhưng phát hiện quan trọng nhất, gây bất ngờ:** MERVIN — hệ thống có báo cáo chi tiết nhất mình đọc được — **không xây bộ máy compositional scoring tự động** (parse → evaluate từng predicate → combine bằng min/weighted) như đã phác thảo ở `phac-thao-compositional-scoring.md`. Thay vào đó:
- Trả về **top-k = 1000** kết quả thô từ mỗi index (frame/transcript/summary).
- **Con người trực tiếp duyệt mắt** qua danh sách đó ("each keyframe is manually inspected"), có phát lại video để xác minh ngữ cảnh.
- Việc "thoả tất cả điều kiện AND" — chính bộ não người dùng đang làm, không phải một combiner tự động.

→ **Bài học cần điều chỉnh khung ưu tiên:** với thời gian có hạn của một đội thi, **đầu tư vào (a) nhiều index bổ sung nhau (frame/transcript/summary) + (b) UI duyệt nhanh top-1000 hiệu quả** có thể có ROI cao hơn **xây một compositional scoring engine tự động hoàn chỉnh** — vì con người vốn *rất giỏi* nhận ra "cả 2 điều kiện có xảy ra cùng lúc không" chỉ bằng mắt nhìn 1 khung hình, việc máy làm thay chỉ đáng giá khi *số lượng ứng viên còn quá lớn để người xem xuể* hoặc thời gian thi quá gấp không kịp duyệt tay.

**Điều chỉnh cụ thể vào phác thảo cũ:** compositional scoring engine tự động (`phac-thao-compositional-scoring.md`) vẫn đáng xây, nhưng nên coi là **tầng lọc thô bổ sung để giảm số lượng xuống mức người duyệt xuể** (ví dụ từ top-1000 xuống top-50), **không phải** để thay thế hoàn toàn việc con người xác nhận — verify cuối cùng vẫn nên là mắt người, vì rẻ hơn và đáng tin hơn một combiner tự động chưa chắc calibrate đúng.

### Lỗ hổng 3 (Temporal) — **xác nhận, và có công thức cụ thể thay cho phần "sequence model" trừu tượng**

MERVIN có "Temporal Search" xử lý đúng bài toán slide 15/AIC nêu: tìm chuỗi 2 sự kiện E1→E2. Cách làm **đơn giản hơn nhiều** so với "sequence model" mình từng đề xuất ở `lo-hong-2-va-5-chi-tiet.md`:

1. Chạy **2 truy vấn độc lập** cho E1 và E2 (không phải 1 model chuỗi phức tạp).
2. **Lọc cứng theo ràng buộc thời gian:** loại cặp có `T2 < T1` (sai thứ tự) hoặc `T2 - T1 > 5 phút` (quá xa nhau).
3. **Xếp hạng bằng công thức trọng số đơn giản:**
   ```
   S_video = 10.0 × S_pair + 5.0 × (S̄1 + S̄2)
   ```
   trong đó `S_pair` là điểm của cặp khớp tốt nhất, `S̄1, S̄2` là điểm trung bình top-10 của từng sự kiện riêng lẻ — ưu tiên mạnh cặp khớp tốt nhất, nhưng vẫn cộng thêm để không bỏ sót video có cả 2 sự kiện đều khớp tốt dù cặp cụ thể chưa phải tốt nhất.

→ Đây chính là bản hiện thực hoá rất cụ thể của nguyên lý "coarse recall độc lập từng vị từ rồi hợp lại" đã nói ở lỗ hổng 2.5 — **nhưng đơn giản hơn nhiều so với việc mình từng nghĩ cần "sequence model"**. Bài học: đừng vội nghĩ cần model phức tạp cho temporal — **filter cứng + công thức trọng số tuyến tính** đã đủ để một đội mạnh đạt điểm cao.

NII-UIT (vô địch VBS) có "Dynamic Temporal Search" được mô tả là "đánh giá toàn diện độ liên quan của frame, vượt trội hơn phương pháp truyền thống" — khả năng là một phiên bản tinh vi hơn công thức tuyến tính trên, nhưng **không lấy được chi tiết thuật toán** (bài báo bị chặn đăng nhập). Nếu muốn đào sâu, đây là hướng cần tìm bản PDF đầy đủ (MMM 2025 proceedings).

### Lỗ hổng 4 (Coarse/fine split & thiết kế theo metric) — **xác nhận mạnh nhất, gần như trùng khớp tuyệt đối**

Đây là phần khớp thực tế rõ nhất. Metric chính thức BTC dùng cho AIC HCMC (ghi rõ trong paper MERVIN, mục 4.2):

```
Final Score = (1/5) × Σ_k  max_{1≤i≤k} R-Score(r_i),  với k ∈ {1, 5, 20, 50, 100}
```

R-Score là nhị phân đúng/sai (KIS: khớp đúng video+frame; VQA: khớp cả câu trả lời; Temporal: đoạn overlap ground truth).

→ Đây **chính là Recall@K trung bình trên nhiều giá trị K** — đúng y hệt cấu trúc đã đề xuất ở `bo-metric-va-validation-set.md` mục 1 (KIS). Đây không phải trùng hợp: BTC *thật sự* chấm theo đúng logic "top-K nào cũng phải tốt, không chỉ top-1" — xác nhận trực tiếp rằng khung Recall@K (K=1,5,10,100...) đã dựng là **đúng theo sát chuẩn thi thật**, không phải suy đoán lý thuyết suông.

Cũng xác nhận nguyên lý coarse/fine: MERVIN dùng `k=1000` cho coarse recall keyframe (rất rộng, ưu tiên không bỏ sót), rồi thu hẹp dần qua duyệt tay + lọc từ khoá — đúng tinh thần "coarse rẻ trên toàn corpus, fine đắt trên tập nhỏ dần".

### Lỗ hổng 5 (Iterative refinement) — **xác nhận, và xác nhận luôn hướng KISC là "chưa ai làm"**

MERVIN có vòng lặp tương tác rõ ràng qua giao diện React: tìm bằng frame/transcript/summary, xem kết quả, tinh chỉnh câu query, lặp lại ("Iterative Refinement and evaluation" — đúng tên gọi). Giao diện tối ưu cho tốc độ (stream trực tiếp từ YouTube thay vì tải về máy, giảm độ trễ).

**Điểm xác nhận quan trọng nhất cho nhánh brainstorm gần đây:** ở mục "Future Work", MERVIN viết thẳng:
> *"a significant opportunity lies in exploring the use of **LLM agents to automate parts of the retrieval workflow**... such agents could autonomously manage tasks currently requiring manual intervention, like **query refinement and result verification**"*

→ Đây **xác nhận trực tiếp** rằng hướng brainstorm KISC/agentic (lấy cảm hứng từ Claude Code harness ở `cam-hung-tu-claude-code-harness.md`) **là một khoảng trống thật sự**, chưa được các đội mạnh giải quyết — không phải mình tưởng tượng ra một hướng viển vông. Một đội tốp đầu tự nhận đây là hướng *tương lai* của họ, chưa làm.

---

## 2. Ý tưởng mới nổi bật — nối lại với chính slide tập huấn

**Sinh ảnh truy vấn bằng Stable Diffusion** (NII-UIT): thay vì chỉ tìm bằng text-to-image (CLIP text vs image), họ **sinh một ảnh minh hoạ** từ mô tả text bằng Stable Diffusion, rồi **dùng ảnh sinh đó làm truy vấn hình ảnh** (image-to-image search). Lý do kỹ thuật: similarity ảnh-ảnh trong không gian embedding thường **phân biệt tốt hơn** similarity text-ảnh (CLIP cross-modal vốn yếu hơn within-modal, đúng gốc rễ semantic gap #1 đã nêu từ đầu chuỗi note).

**Đây chính xác là điều slide 37 của tài liệu tập huấn đã gợi ý** (Case Study 1 — "lính chì"): *"Tìm một ví dụ hình ảnh minh họa cho lính chì?"* — một trong ba chiến thuật được liệt kê ngay từ buổi tập huấn. NII-UIT (đội vô địch VBS 2025) đã hiện thực hoá đúng gợi ý đó bằng công cụ sinh ảnh hiện đại. **Đáng để thử nghiệm sớm** — chi phí thấp (một API Stable Diffusion/tương đương), giá trị cao đặc biệt cho các query mô tả object/khái niệm cụ thể mà từ ngữ khó diễn đạt trực tiếp qua CLIP text.

---

## 3. Tổng hợp — sửa lại "thứ tự ưu tiên xây dựng" ở `ban-do-du-lieu-pipeline.md` dựa trên bằng chứng thực tế

| Mục cũ | Điều chỉnh sau khi đối chiếu thực tế |
|---|---|
| #3 OCR (ROI cao nhất) | **Giữ nguyên** — không đội nào trong 3 nguồn nhắc OCR cụ thể, nhưng "Vo et al. 2025" (SOICT) có dùng OCR cho tin tức tiếng Việt — vẫn đáng làm sớm |
| #6 Region-CLIP / binding (làm sau) | **Hạ ưu tiên thêm** — MERVIN không cần tới nó mà vẫn đạt 79/88; con người duyệt mắt giải quyết binding tự nhiên hơn máy |
| #7 LVLM verify (đắt, làm sau) | **Giữ nguyên**, nhưng bổ sung: cân nhắc thay một phần bằng **Stable Diffusion sinh ảnh truy vấn** — rẻ hơn LVLM, giải quyết đúng vấn đề semantic gap cho vật thể cụ thể |
| #8 Belief-state/UI tương tác | **Nâng ưu tiên** — đây chính là thứ MERVIN đầu tư kỹ nhất (kiến trúc phân tách embedding/UI, stream trực tiếp YouTube) và là hướng còn bỏ ngỏ theo lời chính đội mạnh |
| *(mới)* Text encoder tiếng Việt chuyên biệt cho transcript/summary | **Thêm mới, ưu tiên cao** — chi tiết bị bỏ sót hoàn toàn trước đó, ảnh hưởng trực tiếp chất lượng kênh ASR/Metadata |
| *(mới)* Video-level summary index (LLM tóm tắt cả video) | **Thêm mới, ưu tiên trung bình** — dễ làm (1 lần gọi LLM/video), hữu ích cho query mang tính tường thuật sự kiện |
| Compositional scoring engine tự động (note phác thảo #2) | **Hạ vai trò** — nên coi là bộ lọc thô phụ trợ (giảm số ứng viên xuống mức người duyệt xuể), không phải cơ chế quyết định cuối cùng thay con người |

---

## 4. Giới hạn của việc đối chiếu này

- Chỉ MERVIN có full-text — architecture cụ thể nhất trong 3 nguồn, nhưng **không chắc là quán quân tuyệt đối** năm đó, chỉ "top các đội đại học".
- NII-UIT (vô địch VBS 2025 thật) chỉ có abstract — thiếu chi tiết thuật toán "Dynamic Temporal Search" thật sự hoạt động ra sao. Nếu muốn khai thác sâu hơn, cần tìm bản PDF đầy đủ từ MMM 2025 proceedings (hiện bị chặn sau đăng nhập Springer/ACM).
- Dionysus (vô địch AIC 2023) chỉ có tin báo chí, không có báo cáo kỹ thuật công khai.
- Đây là 1 lát cắt nhỏ (2-3 hệ thống) trong hàng chục đội mỗi năm — không đại diện cho "cách làm chuẩn", chỉ là bằng chứng cụ thể để **kiểm chứng chéo** khung lý thuyết đã tự dựng, không phải để copy y nguyên.

---

## Sources

- [Results of the 2025 Video Browser Showdown (arXiv)](https://arxiv.org/abs/2509.12000)
- [Video Browser Showdown — Teams & Papers (All Years)](https://videobrowsershowdown.org/teams/)
- [NII-UIT at VBS2025: Multimodal Video Retrieval with LLM Integration and Dynamic Temporal Search (Springer)](https://link.springer.com/chapter/10.1007/978-981-96-2074-6_38)
- [Giải nhất AI Challenge 2023 — đội Dionysus (UIT tuyển sinh)](https://tuyensinh.uit.edu.vn/giai-nhat-tai-ai-challenge-2023-thuoc-ve-doi-dionysus-den-tu-truong-dai-hoc-cong-nghe-thong-tin-dhqg-hcm)
- [MERVIN: A Unified Framework for Multimodal Event Retrieval in Vietnamese News Videos (arXiv 2605.16120)](https://arxiv.org/pdf/2605.16120)
- [Event Retrieval from Large Video Collection in Ho Chi Minh City AI Challenge 2024 (Springer)](https://link.springer.com/chapter/10.1007/978-981-96-4291-5_1)
- [diveXplore at the Video Browser Showdown 2025 (Springer)](https://link.springer.com/chapter/10.1007/978-981-96-2074-6_30)
