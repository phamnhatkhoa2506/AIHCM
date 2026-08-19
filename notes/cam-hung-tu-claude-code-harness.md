# Cảm hứng từ Claude Code Harness — áp dụng cho KISC & vòng lặp tương tác

> Note ghi lại một quan sát nảy ra ngoài lề trong lúc brainstorm: harness của Claude Code (công cụ đang dùng
> để làm các note này) có cấu trúc tương tác rất giống bài toán KISC. Nối tiếp `phac-thao-belief-state-kisc.md`
> và `phac-thao-compositional-scoring.md`. Ngày: 2026-07-06.

---

## 1. Vì sao quan sát này trúng, không phải chỉ là liên tưởng vui

Về bản chất, harness của Claude Code là **một agent hoạt động dưới sự bất định**: nhận chỉ dẫn có thể mơ hồ,
dùng tool để kiểm chứng trạng thái thật thay vì đoán, và **chỉ hỏi lại người dùng khi quyết định thật sự thuộc
về họ** (không tự suy ra được từ context/code). Đây chính xác là hình dạng của bài toán **KISC** — chỉ khác domain
(kỹ thuật phần mềm vs video retrieval).

| Claude Code | Hệ thống retrieval của bạn (KISC / vòng lặp tương tác) |
|---|---|
| User đưa chỉ dẫn mơ hồ ("sửa cái bug đó") | User đưa mô tả trí nhớ mơ hồ ("gặp bạn cũ tuần trước") |
| Agent dùng đúng tool để kiểm chứng (Read/Grep/Bash) thay vì đoán | Hệ thống route predicate tới đúng evaluator (Objects/OCR/ASR/LVLM) thay vì ép hết qua CLIP |
| Hỏi lại khi quyết định thuộc về user — ở dạng trắc nghiệm 2-4 lựa chọn, không hỏi mở | Hỏi lại để thu hẹp candidate set — nên cũng là lựa chọn rời rạc, không hỏi mở |
| Giữ context/memory xuyên suốt phiên, không hỏi lại điều đã biết | `applied_predicates` trong belief-state — không hỏi lại trục đã hỏi |
| Plan trước khi hành động tốn kém/khó đảo ngược (EnterPlanMode) | Chỉ chạy Compositional Scoring Engine đầy đủ khi belief đã đủ tự tin |
| Dispatch nhiều sub-agent độc lập, song song, cho các phần việc tách rời | Recall song song theo từng vị từ con đã phân rã, rồi hợp pool ứng viên |

**Bằng chứng đây không phải suy diễn:** slide 25 và 28 của chính tài liệu tập huấn AIC 2026 đã chỉ thẳng hướng này.
Timeline slide 25: "2022 Objects → 2023 Personalized Concepts → 2024 Natural description → 2025 **AI Agent** →
2026 **Next steps???**". Slide 28 nói nguyên văn: *"Smart Interaction: Trợ lý ảo có khả năng đặt câu hỏi ngược lại
cho người dùng để làm rõ ý định (Interactive Search)"*. Ban tổ chức đang ngầm gợi ý đúng hướng mà việc quan sát
Claude Code vừa gợi ra một cách tự nhiên.

---

## 2. Các pattern cụ thể có thể mượn (xếp theo giá trị chuyển giao)

### (1) Câu hỏi trắc nghiệm thay vì câu hỏi mở — **giá trị cao nhất**
`AskUserQuestion` của Claude Code không bao giờ hỏi mở kiểu "kể thêm đi" — luôn ép về **2-4 lựa chọn cụ thể,
loại trừ lẫn nhau**. Lý do sâu: câu trả lời mở khó quy về một phép chia sạch của belief space (cần parse thêm,
dễ mơ hồ khi convert thành predicate — rủi ro đúng như "parse sai lan truyền" đã ghi ở note compositional scoring).
Câu trắc nghiệm **chính là** một phép chia belief space tường minh → info gain tính được ngay (đúng công thức entropy
đã viết ở note KISC mục 4.3) → convert thành predicate không mơ hồ.

**Ý tưởng nâng cấp riêng cho domain video (hơn cả bản gốc text-only):** thay vì trắc nghiệm bằng chữ, **hiển thị
2-4 cụm thumbnail đại diện cho các nhánh trả lời khả dĩ**, để người dùng chọn "giống cảnh nào hơn" bằng mắt.
Tự nhiên hơn hẳn cho domain thị giác — người dùng nhận diện hình ảnh nhanh hơn diễn đạt bằng lời rất nhiều
(đây cũng là lý do "trí nhớ hình ảnh" thường chính xác hơn "trí nhớ diễn đạt bằng câu chữ" — đúng gốc rễ semantic
gap #1 đã nêu từ đầu chuỗi note). **→ Đây là hạng mục đáng thử nghiệm sớm nhất trong toàn bộ ý tưởng này.**

### (2) Kỷ luật "không đoán, dùng đúng tool cho đúng việc"
Claude Code tách bạch Read/Grep/Glob/Bash, chọn công cụ đúng việc thay vì làm mọi thứ qua một cơ chế chung.
Đây khớp thẳng với thiết kế evaluator registry đã tự đề ra ở lỗ hổng 2 (route mỗi predicate tới evaluator bản địa
— scene→CLIP, count→detector, chữ→OCR...). Quan sát này có giá trị như một **xác nhận chéo**: pattern tự nghĩ ra
độc lập lại khớp với pattern đã được kiểm chứng tốt trong một hệ agent hoàn toàn khác domain — dấu hiệu tốt cho
thấy đây là nguyên lý chung của "agent dưới bất định", không phải trùng hợp ngẫu nhiên.

### (3) "Trust but verify" — đừng để một nguồn không calibrate đứng ngang hàng nguồn đã calibrate
System prompt của Claude Code nói rõ: đừng tin tuyên bố của agent con là sự thật, phải kiểm chứng lại trạng thái
thật. Áp vào hệ retrieval: **đừng để phán đoán của LVLM đứng một mình như sự thật tuyệt đối** khi verify quan hệ
phức tạp (holding, dropping...) — nên cross-check bằng tín hiệu rẻ hơn đã có sẵn (box overlap giữa 2 object,
region-CLIP score) trước khi tin, hoặc tối thiểu phải **calibrate riêng độ tin của kênh LVLM** thay vì trộn thẳng
vào combiner như các evaluator đã calibrate khác (đúng nguyên lý calibration đã nhấn mạnh ở note compositional scoring).

### (4) Plan trước khi hành động tốn kém/khó đảo ngược
`EnterPlanMode`/`ExitPlanMode` của Claude Code: gom đủ context, lập kế hoạch, xin duyệt, rồi mới thực thi việc
tốn kém. Khớp thẳng với điều kiện dừng ở belief-state (`should_terminate`): chỉ chạy full Compositional Scoring
Engine (đắt) khi belief đã đủ tự tin, không chốt nửa vời rồi chạy nhầm.

### (5) Không hỏi lại / không lật lại điều đã biết
Nguyên tắc "don't re-derive facts already established, don't re-litigate a decision the user has already made"
trong system prompt Claude Code khớp thẳng với `applied_predicates` history đã thiết kế trong belief-state —
tránh hỏi lặp một trục đã hỏi, và tránh để một ràng buộc cứng đã áp bị "quên" giữa các lượt.

### (6) Dispatch song song cho các phần việc độc lập, rồi hợp lại
`Agent` tool của Claude Code cho phép chạy nhiều sub-agent song song khi các việc **độc lập, không phụ thuộc
lẫn nhau** — mỗi agent làm việc trong context cô lập, rồi kết quả được tổng hợp lại. Khớp thẳng với ý tưởng
"recall-by-decomposition" đã ghi ở note lỗ hổng 2 (mục 2.5): với query có nhiều vị từ con (`A ∧ B ∧ C`),
chạy coarse recall **độc lập, song song** cho từng vị từ con, rồi **hợp các pool ứng viên** trước khi áp AND
ở tầng verify — tránh trường hợp coarse recall gộp cả câu bỏ sót ứng viên mà chỉ 1 điều kiện nổi bật.

---

## 3. Giới hạn cần nói thẳng — tránh ngộ nhận

Đây là mượn **pattern tương tác & triết lý kiến trúc của một agent dưới bất định**, **không phải** "bọc Claude Code
quanh video search". Phần khó thật sự của cuộc thi (CLIP embedding, region-matching, video, temporal reasoning)
là domain hoàn toàn khác — harness của Claude Code không đụng vào và không giúp gì ở đó. Cái transfer được chỉ là
phần **domain-agnostic**: cách một agent thu thập thông tin dưới bất định trước khi cam kết hành động. Đó cũng
chính xác là lý do nó transfer tốt — các nguyên lý này (hỏi trắc nghiệm để tối đa info gain, verify trước khi tin,
không hỏi lại cái đã biết, plan trước khi hành động đắt) không đặc thù cho code hay cho video, mà đặc thù cho
**bài toán ra quyết định dưới thông tin thiếu**.

## 4. Ý tưởng đầu cơ, chưa chắc áp dụng — ghi lại để sau này cân nhắc

Claude Code có hệ **memory xuyên phiên** (user/feedback/project/reference) — học dần về người dùng qua thời gian.
Nếu về sau cuộc thi có yếu tố cá nhân hoá thật sự (một người dùng cụ thể, nhiều phiên tìm kiếm lifelog của chính họ),
một "hồ sơ dài hạn" kiểu tương tự (vd hệ thống dần học "bạn của X thường mặc áo xanh") có thể hữu ích. Dữ liệu
AIC 2026 hiện tại là video tin tức HTV/YouTube (không phải lifelog cá nhân liên tục của 1 người), nên ý tưởng này
**đầu cơ, để đó** — không phải ưu tiên hiện tại, chỉ ghi lại phòng khi bài toán mở rộng.

---

## 5. Nối với các note trước

- Pattern (1) và (3) áp trực tiếp vào `choose_next_question` và evaluator registry trong `phac-thao-belief-state-kisc.md`.
- Pattern (2) xác nhận chéo thiết kế evaluator registry trong `phac-thao-compositional-scoring.md`.
- Pattern (6) cụ thể hoá thêm mục "gửi sub-query nào vào coarse recall" đã nêu ở `lo-hong-2-va-5-chi-tiet.md` §2.5.
- Nếu thử nghiệm pattern (1) phiên bản thumbnail-cluster, nên đo bằng chính bộ metric ở `bo-metric-va-validation-set.md`
  (Turn efficiency, Info-gain/lượt) để biết nó có thật sự tốt hơn trắc nghiệm chữ hay không — đừng chỉ tin cảm giác "trực quan hơn".
