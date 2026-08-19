# Lớp tương tác thực dụng cho Lỗ hổng 5 — kiểm soát được, tiện lợi, hiệu năng

> Điều chỉnh `phac-thao-belief-state-kisc.md` và `cam-hung-tu-claude-code-harness.md` sau khi đối chiếu thực tế
> (`doi-chieu-thuc-te-cac-doi-vo-dich.md`). Quyết định: đẩy mạnh lỗ hổng 5, nhưng ràng buộc bởi 3 tiêu chí —
> **kiểm soát được, tiện lợi, hiệu năng** — thay vì tối đa hoá mức độ "agentic/tự động". Ngày: 2026-07-07.

---

## 0. Vì sao cần thuần hoá lại thiết kế cũ

`phac-thao-belief-state-kisc.md` thiết kế một vòng lặp khá "agentic": hệ thống tự chọn câu hỏi bằng LLM/info-gain
tại runtime, tự cập nhật belief-state, tự quyết định khi nào dừng. Đây là bản đầy cảm hứng nhưng có 3 rủi ro
đúng như bạn chỉ ra:

1. **Khó kiểm soát:** quyết định "hỏi gì tiếp" nằm trong một hộp đen (LLM), người dùng không dễ biết vì sao hệ
   thống chọn câu đó, và không dễ can thiệp giữa chừng.
2. **Kém tiện lợi:** buộc người dùng đi theo *một luồng hội thoại tuần tự* do hệ thống dẫn dắt, thay vì tự do
   thao tác theo cách họ muốn.
3. **Kém hiệu năng:** mỗi lượt cần 1 lệnh gọi LLM để chọn câu hỏi/parse câu trả lời — tốn thời gian thật,
   trong khi KIS bị chấm theo đồng hồ.

MERVIN (note đối chiếu thực tế) đã cho thấy: đội mạnh thắng bằng UI **nhanh, nhiều lựa chọn song song, con
người tự lái** — không phải bằng một agent tự quyết định. Bài học: **giữ phần "thông minh" nhưng đẩy nó ra
khỏi đường găng thời gian thực (critical path), để tốc độ và quyền kiểm soát luôn thuộc về người dùng.**

---

## 1. Ba ràng buộc thiết kế — cụ thể hoá thành quy tắc kỹ thuật

| Ràng buộc | Quy tắc kỹ thuật kéo theo |
|---|---|
| **Kiểm soát được** | Mọi ràng buộc đã áp (do hệ thống gợi ý hay người dùng tự chọn) phải **hiển thị tường minh** và **gỡ được từng cái một**. Không có bước nào hệ thống tự commit một quyết định mà người dùng không thấy. |
| **Tiện lợi** | Ưu tiên **click** hơn **gõ**; ưu tiên **nhiều lựa chọn hiển thị song song** hơn **hội thoại tuần tự** một-câu-một-lúc. Người dùng luôn được tự quyết đường đi, hệ thống chỉ gợi ý bên cạnh chứ không chặn lối. |
| **Hiệu năng** | Bất cứ phép tính nào **có thể làm trước (offline, lúc indexing)** thì phải làm trước — không để việc "chọn câu hỏi/facet tối ưu" trở thành một lệnh gọi LLM tốn thời gian *tại* lúc người dùng đang chờ. |

---

## 2. Kiến trúc điều chỉnh — "Bounded Interaction Layer"

Ý tưởng cốt lõi: **tách phần hạ tầng thông minh (belief-state, information gain) ra khỏi đường găng thời gian
thực.** Phần tính toán tốn kém chuyển hết sang **offline/precompute lúc indexing**; phần **runtime chỉ còn tra
cứu và hiển thị** — nhanh gần như tra bảng, không phải suy luận.

```
OFFLINE (lúc indexing, rẻ vì không có ai đang chờ):
  - Cluster toàn bộ keyframe theo CLIP embedding (K-means) → cụm ngữ nghĩa thô (không cần gán nhãn tay)
  - Từ Objects: tính sẵn nhãn facet phổ biến mỗi frame (có người/không, số người, loại cảnh...)
  - Từ Metadata: nhóm sẵn theo khung giờ/kênh/chủ đề

ONLINE (lúc người dùng thao tác, phải nhanh):
  Lớp 1 — Multi-mode browsing song song (không tuần tự, không chờ)
  Lớp 2 — Facet filter dạng click nhanh (tra bảng đã precompute, không gọi LLM)
  Lớp 3 — Predicate transparency + undo (mọi filter hiện dưới dạng chip, gỡ được)
  Lớp 4 — Gợi ý AI (LLM/agent) — CHỈ chạy khi được yêu cầu rõ ràng, không nằm trên đường chính
```

### Lớp 1 — Multi-mode browsing song song (học từ MERVIN)
Thay vì một luồng hội thoại "hệ thống hỏi → chờ trả lời → hỏi tiếp", hiển thị **đồng thời** nhiều chế độ tìm
kiếm độc lập (tìm theo frame/CLIP, tìm theo transcript/ASR, tìm theo summary cấp video, tìm theo temporal 2 sự
kiện) — đúng mô hình 4 module của MERVIN. Người dùng tự chọn dùng cái nào trước, không bị dẫn dắt theo kịch
bản cố định. **Đây chính là phần "tiện lợi" + "hiệu năng":** không độ trễ chờ hệ thống quyết định bước kế tiếp.

### Lớp 2 — Facet filter nhanh (bản "thuần hoá" của info-gain question)
Thay cho việc LLM *runtime* chọn câu hỏi tối ưu information gain (như thiết kế cũ), giờ:
- Các trục facet (indoor/outdoor, số người, khung giờ...) đã **precompute sẵn** lúc indexing.
- Khi có tập ứng viên hiện tại, hệ thống chỉ cần **đếm/group-by** (rẻ, tức thời) để biết trục nào đang chia
  tập ứng viên đều nhất — công thức entropy ở note cũ (`expected_info_gain`) **vẫn dùng được nguyên vẹn**,
  chỉ khác là chạy trên dữ liệu đã cluster sẵn, không cần LLM đánh giá real-time.
- Hiển thị như **chip/nút bấm** ("Trong nhà" / "Ngoài trời" / "Không rõ"), không phải câu hỏi hội thoại dạng
  chữ — người dùng bấm 1 cái là xong, không cần gõ.

```python
# Runtime: chỉ tra cứu, không gọi model
def suggest_facets(belief: BeliefState, precomputed_clusters) -> list[FacetSuggestion]:
    scored = []
    for facet in precomputed_clusters.available_facets(belief.candidates):
        gain = expected_info_gain(belief, facet)   # rẻ: chỉ đếm trên cluster đã có sẵn
        scored.append((facet, gain))
    return sorted(scored, key=lambda x: -x[1])[:3]   # gợi ý top-3 facet, không ép chọn
```

### Lớp 3 — Predicate transparency + undo (đây là "kiểm soát được")
Mọi ràng buộc đang áp — dù đến từ facet-click hay từ filter tay — hiển thị thành **chip có nút X**:
`[Ngoài trời ✕]  [Có 2 người ✕]  [Tuần trước ✕]`. Người dùng luôn thấy chính xác đang lọc theo gì, gỡ bất kỳ
cái nào bất kỳ lúc nào. Đây là bản UI hoá trực tiếp của `applied_predicates` — biến từ log nội bộ (dev nhìn
thấy) thành giao diện người dùng nhìn thấy và điều khiển được.

### Lớp 4 — LLM/agentic là tuỳ chọn, không phải mặc định
Toàn bộ phần "hội thoại thông minh" (LLM parse câu trả lời tự do, chọn câu hỏi mở, Stable-Diffusion sinh ảnh
truy vấn...) **vẫn giữ lại từ các note trước** — nhưng đặt sau một nút bấm rõ ràng ("Gợi ý cho tôi" / "Sinh ảnh
minh hoạ") thay vì tự động chạy mỗi lượt. Người dùng chủ động mời AI tham gia khi cần, AI không đứng chắn giữa
người dùng và kết quả. Điều này giữ nguyên "cảm hứng KISC" nhưng gỡ bỏ đúng rủi ro đã nêu ở mục 0.

---

## 3. Vì sao cách này vẫn "đẩy mạnh lỗ hổng 5" chứ không phải rút lui

Dễ hiểu lầm rằng bounded lại = làm ít đi. Thực ra không — phần **hạ tầng thu hẹp thông tin** (belief-state,
info-gain, predicate history) **được giữ nguyên toàn bộ**, chỉ thay đổi **nơi nó chạy và ai bấm nút**:

| | Thiết kế cũ (agentic) | Thiết kế mới (bounded) |
|---|---|---|
| Ai chọn câu hỏi/facet kế tiếp | Hệ thống (LLM, runtime) | Hệ thống **gợi ý** (precomputed, tức thời), người dùng **quyết** |
| Câu trả lời | Gõ tự do → LLM parse | Click chip có sẵn (nhanh, không mơ hồ) |
| Info-gain có dùng không | Có | **Vẫn có, y nguyên công thức** — chỉ tính trên dữ liệu rẻ hơn |
| Khi nào AI "thông minh" tham gia | Mỗi lượt, mặc định | Khi người dùng chủ động bấm gọi |
| Tốc độ | Phụ thuộc độ trễ LLM mỗi lượt | Gần tức thời (tra cứu) |
| Ai luôn thấy/gỡ được ràng buộc | Không rõ ràng | Luôn thấy, luôn gỡ được (chip) |

→ Đây vẫn là hệ thống thu hẹp candidate set qua nhiều lượt tương tác (đúng lõi lỗ hổng 5), chỉ khác **cơ chế
điều khiển vòng lặp chuyển từ "AI dẫn dắt" sang "người dùng dẫn dắt, AI hỗ trợ ở bên cạnh"** — đúng tinh thần
"tiện lợi và hiệu suất mới là thứ đáng lưu tâm hiện tại" bạn vừa nêu.

---

## 4. Việc cần làm, xếp theo thứ tự (bổ sung cho bảng ưu tiên ở `ban-do-du-lieu-pipeline.md`)

1. **Cluster CLIP embedding thành facet thô** (K-means lúc indexing) — rẻ, làm được ngay, không cần model mới.
2. **UI multi-mode song song** (giống 4 module MERVIN: frame/transcript/summary/temporal) — ưu tiên trước cả
   phần facet-click, vì đây là thứ MERVIN chứng minh có ROI cao nhất thực tế.
3. **Chip filter có thể gỡ** — UI đơn giản, giá trị kiểm soát cao, chi phí thấp.
4. **Facet suggestion (Lớp 2)** dùng công thức info-gain đã có sẵn từ note cũ — chỉ cần đổi nguồn dữ liệu đầu
   vào từ "LLM runtime" sang "cluster precomputed".
5. **LLM/agent nâng cao (Lớp 4)** — làm sau cùng, coi là tính năng cộng thêm, không phải hạ tầng lõi.

---

## 5. Nối với các note trước

- Tái sử dụng nguyên vẹn: schema `Predicate`, hàm `expected_info_gain`, `update_belief` từ
  `phac-thao-belief-state-kisc.md` — không xây lại, chỉ đổi **nơi gọi** và **tần suất gọi**.
- Áp dụng trực tiếp bài học ROI từ `doi-chieu-thuc-te-cac-doi-vo-dich.md` §1 (lỗ hổng 2 & 5): con người duyệt
  nhanh qua nhiều tín hiệu song song thắng một cơ chế tự động phức tạp, dưới ràng buộc thời gian thi thật.
- Pattern (1) và (5) trong `cam-hung-tu-claude-code-harness.md` (MCQ rời rạc, không hỏi lại cái đã biết) vẫn
  giữ nguyên giá trị — chỉ thực thi bằng **chip UI tiền tính sẵn** thay vì **LLM quyết định runtime**.
