# Kiến trúc 2 tầng — Agent (dưới) & UI (trên), và hợp đồng giao tiếp giữa chúng

> Nối tiếp `lop-tuong-tac-thuc-dung-kiem-soat-duoc.md`, `phac-thao-belief-state-kisc.md`,
> `phac-thao-compositional-scoring.md`, `hyde-cho-truy-van-hinh-anh.md`. Quyết định kiến trúc: tách bạch
> **tầng Agent** (logic, không biết gì về giao diện) khỏi **tầng UI** (bộ widget cố định, chỉ render).
> Phần mới cần thiết kế: **hợp đồng giao tiếp (contract)** giữa hai tầng. Ngày: 2026-07-08.

---

## 0. Vì sao tách 2 tầng — và nó giải quyết gì

Ở các vòng brainstorm trước, hai câu hỏi bị trộn lẫn vào nhau: *"giao diện nên trông thế nào"* và *"khi nào nên
hỏi gì/hiện gì"*. Trộn lẫn khiến ta lo cả hai cùng lúc, dẫn tới rủi ro đã nêu (chat tự do vẽ UI mỗi lượt, agent
tự quyết hoàn toàn không có rule). Tách bạch giải quyết gọn:

- **Tầng Agent (dưới):** toàn bộ logic đã thiết kế qua các note trước (recall, compositional scoring,
  belief-state, HyDE) — không biết, không quan tâm nó đang chạy trong chat hay dashboard.
- **Tầng UI (trên):** một bộ **widget cố định, hữu hạn** — chỉ nhận lệnh và render, không tự quyết định gì.
- **Hợp đồng (contract):** tập lệnh đóng mà Agent được phép phát ra, và tập sự kiện đóng mà UI được phép báo
  ngược lại. Đây là phần **mới**, note này tập trung vào đây.

**Lợi ích kép:** (1) đúng nguyên lý "kiểm soát được" — Agent chỉ được chọn trong menu cố định, không tự sáng
tác giao diện; (2) **tái sử dụng được** — cùng 1 tầng Agent có thể phục vụ cả bề mặt chat (cho KISC, nơi hội
thoại là bản chất bài toán) lẫn 1 nút "hỏi AI" nhỏ nhúng trong dashboard (cho KIS/AVS/VQA) — không cần xây
2 lần logic, chỉ khác nơi hợp đồng này được gọi.

---

## 1. Tầng Agent (dưới) — tổng hợp lại, không thiết kế mới

| Thành phần | Nguồn | Vai trò |
|---|---|---|
| Coarse recall | `ban-do-du-lieu-pipeline.md` | CLIP + Metadata + Objects → shortlist |
| Compositional scoring | `phac-thao-compositional-scoring.md` | parse → evaluate (registry) → calibrate → combine (min/weighted) |
| Belief-state | `phac-thao-belief-state-kisc.md` | candidate set, applied predicates, info-gain chọn facet/câu hỏi kế tiếp |
| HyDE-ảnh | `hyde-cho-truy-van-hinh-anh.md` | kênh recall bổ sung khi cần |

Tầng này expose ra ngoài đúng **một hàm duy nhất** (về mặt khái niệm):

```
next_action(belief_state, last_event) -> Action
```

Nhận trạng thái hiện tại + sự kiện người dùng vừa gửi, trả về **đúng 1 Action** thuộc tập đóng ở mục 2. Toàn bộ
độ "thông minh" (info-gain, calibration, HyDE...) nằm gọn trong hàm này — UI không cần biết bên trong nó làm gì.

---

## 2. Tập lệnh Agent → UI (Action) — đóng, hữu hạn

Mỗi Action có: `type` (loại widget cố định) + `payload` (dữ liệu điền vào) + `prompt_text` (1 câu ngắn đi kèm,
theo đúng quyết định "kèm câu nói cho dễ hiểu" — nhưng **ép ngắn**, xem mục 5).

```jsonc
// 1. Trắc nghiệm làm rõ (khi belief-state mơ hồ, có trục info-gain cao)
{
  "type": "show_options",
  "prompt_text": "Bạn nhớ không gian đó trong nhà hay ngoài trời?",
  "payload": {
    "choices": [
      { "id": "indoor", "label": "Trong nhà" },
      { "id": "outdoor", "label": "Ngoài trời" },
      { "id": "unsure", "label": "Không chắc" }
    ],
    "allow_free_text": true   // LUÔN true — không bao giờ ép chỉ chọn trong option
  }
}

// 2. Bản đồ cụm ngữ nghĩa (KHÔNG phải GPS — cụm theo nội dung thị giác, vd "biển", "núi", "trong nhà")
{
  "type": "show_concept_map",
  "prompt_text": "Đây là các nhóm cảnh đang có trong tập ứng viên — chọn nhóm gần trí nhớ của bạn hơn.",
  "payload": {
    "clusters": [
      { "id": "c1", "label": "Biển", "thumb_url": "...", "candidate_count": 42 },
      { "id": "c2", "label": "Núi", "thumb_url": "...", "candidate_count": 8 },
      { "id": "c3", "label": "Trong nhà / studio", "thumb_url": "...", "candidate_count": 120 }
    ],
    "layout": "spatial | grid"  // spatial = bố trí 2D theo độ giống nhau (kiểu FLAS); grid = đơn giản hơn
  }
  // nguồn cụm: precomputed K-means trên CLIP embedding của candidate set hiện tại —
  // dùng lại đúng "Lớp 1 — Precomputed facet clustering" ở lop-tuong-tac-thuc-dung-kiem-soat-duoc.md,
  // KHÔNG cần dữ liệu địa lý/GPS nào cả -> câu hỏi mở cũ ở mục 8 (dữ liệu AIC có GPS không) không còn áp dụng.
}

// 3. Lưới ảnh để chọn/loại (khi candidate set đã đủ nhỏ để duyệt mắt)
{
  "type": "show_image_grid",
  "prompt_text": "Vài khoảnh khắc gần khớp nhất — cái nào giống trí nhớ của bạn hơn?",
  "payload": {
    "images": [ { "id": "kf_00092", "thumb_url": "...", "score": 0.33 }, ... ],
    "selection_mode": "single | multi | like_dislike"
  }
}

// 4. Chọn vùng không gian (kiểu PraK V4 — AND theo vùng, cho object nhỏ/khó detect)
{
  "type": "show_region_picker",
  "prompt_text": "Mô tả chi tiết ở từng vùng nếu có — để trống nếu không áp dụng.",
  "payload": { "keyframe_id": "kf_00092", "grid": "3x3" }
}

// 5. Xem video / tua theo cảnh (khi đã xác định đúng video, cần đúng khoảnh khắc)
{
  "type": "show_video",
  "prompt_text": "Đúng video rồi — tua tới đoạn bạn nhớ.",
  "payload": { "video_id": "ljY7UNp2fCk", "start_ts": 92, "scene_marks": [0, 34, 92, 145, 210] }
}

// 6. Chỉ text, không widget (khi không cần hỏi gì thêm)
{
  "type": "reply_text",
  "prompt_text": "Đã thu hẹp còn 3 ứng viên, độ tin cậy cao — mình chọn kết quả #1."
}
```

---

## 3. Tập sự kiện UI → Agent (Event) — tương ứng 1-1 với Action

| Action đã hiện | Event UI trả về | Predicate/update tương ứng ở belief-state |
|---|---|---|
| `show_options` | `user_picked_option(id)` hoặc `user_typed_free_text(text)` | free-text → chạy qua parser (`phac-thao-compositional-scoring.md` §3) trước khi thành predicate |
| `show_concept_map` | `user_selected_cluster(id)` | `hard_filter` hoặc `soft_reweight` theo cụm đã chọn (tuỳ độ tin cậy của cluster đó) |
| `show_image_grid` | `user_picked_image(id)` hoặc `user_marked(id, liked/disliked)` | `like` → Rocchio-style embedding nudge (`phac-thao-belief-state-kisc.md` §3); `disliked` → trừ |
| `show_region_picker` | `user_tagged_region(region_id, text)` (nhiều region cùng lúc) | nhóm binding, chấm theo `phac-thao-compositional-scoring.md` §5 |
| `show_video` | `user_scrubbed_to(ts)` hoặc `user_confirmed(frame_id)` | nếu confirm → kết thúc phiên, chốt đáp án |
| `reply_text` | *(không cần event — hoặc `user_continue`)* | — |

**Nguyên tắc bắt buộc:** mọi Event đều được **agent diễn giải thành 1 predicate** rồi đưa qua đúng
`update_belief(belief, predicate, mode)` đã có sẵn — Event không tự ý sửa belief-state trực tiếp trong UI.
Điều này giữ agent làm **nguồn sự thật duy nhất**, UI không có logic nghiệp vụ riêng.

---

## 4. Rule chọn Action — tường minh, không để agent tự quyết mù

Đúng giới hạn kỹ thuật #6 đã nêu (nhất quán hành vi): `next_action()` nên tra theo **bảng quyết định rõ ràng**
trước, chỉ rơi vào "để LLM tự chọn" khi không rule nào khớp:

| Điều kiện trên belief-state | Action ưu tiên |
|---|---|
| Có trục facet info-gain cao (vd chia ~đôi) VÀ chưa hỏi trục đó | `show_options` |
| Belief-state có ≥2 cụm ngữ nghĩa rõ rệt (vd biển/núi/trong nhà) chưa được chọn | `show_concept_map` |
| `len(candidates) <= NGƯỠNG_NHỎ` (vd 20) | `show_image_grid` |
| Query có ≥2 mô tả thuộc tính trên cùng loại object (binding) | `show_region_picker` |
| Top-1 đã vượt ngưỡng tin cậy + cách biệt rõ top-2 | `show_video` (xác nhận) rồi `reply_text` chốt |
| Không rule nào khớp | fallback: `reply_text` báo tiến triển, không ép hỏi thêm |

→ Bảng này **thay thế** việc để agent "tự do quyết định mỗi lần" — hành vi dự đoán được, dễ debug, khớp đúng
nguyên tắc đã đặt ra khi so sánh với PraK V4/MERVIN (đội mạnh thắng nhờ hành vi UI *nhất quán*, không phải
*sáng tạo tuỳ hứng*).

---

## 5. Ràng buộc `prompt_text` — ngắn, vì đây là nơi dễ "rò" thời gian nhất

Đã quyết định giữ câu giải thích đi kèm mỗi widget (cho dễ hiểu), nhưng dưới áp lực đồng hồ thi, **mỗi
`prompt_text` giới hạn 1 câu ngắn** (gợi ý: ≤ 15 từ), không phải đoạn văn. Vai trò của nó chỉ là **gắn ngữ
cảnh cho widget**, không phải giải thích lý do đầy đủ — nếu cần giải thích thêm, để trong 1 dòng phụ nhỏ, thu
gọn được (collapsed by default), không bắt buộc đọc mới thao tác được.

---

## 6. Ví dụ luồng đầy đủ — dùng lại kịch bản slide 15

```
User: "Tìm giúp tôi đoạn video tôi gặp một người bạn cũ vào tuần trước."
Agent: hard_filter theo Metadata (tuần trước) + coarse recall -> belief-state ~300 ứng viên, entropy cao
Agent -> Action: show_options
  prompt_text: "Không gian đó trong nhà hay ngoài trời?"
  choices: [Trong nhà, Ngoài trời, Không chắc]  (allow_free_text: true)

User -> Event: user_picked_option("outdoor")
Agent: soft_reweight theo predicate scene(outdoor) -> belief-state co lại còn ~80, vẫn cần thêm trục
Agent -> Action: show_options
  prompt_text: "Người bạn đó là nam hay nữ?"

User -> Event: user_typed_free_text("một người bạn nam, mặc áo sơ mi xanh, ở quán cà phê")
Agent: parse free-text -> 3 predicate (male, blue_shirt, outdoor_cafe) -> update belief -> còn 4 ứng viên
Agent -> Action: show_image_grid
  prompt_text: "4 khoảnh khắc gần khớp nhất — cái nào đúng?"

User -> Event: user_picked_image("kf_00251")
Agent -> Action: show_video
  prompt_text: "Đây rồi — xác nhận đúng đoạn này chứ?"

User -> Event: user_confirmed("kf_00251")
Agent -> Action: reply_text "Đã chốt kết quả." (kết thúc phiên)
```

Toàn bộ luồng này chỉ dùng **3 loại Action** trong tập 6 loại đã đóng ở mục 2 — không có bước nào cần agent
"tự vẽ" giao diện mới.

---

## 7. Vị trí trong kiến trúc tổng & tái sử dụng

```
                 ┌───────────────────────────┐
                 │   Tầng Agent (headless)   │   <- next_action(belief_state, event) -> Action
                 │  recall / scoring / HyDE  │
                 │  belief-state / info-gain │
                 └─────────────┬─────────────┘
                     Action ↓     ↑ Event
        ┌────────────────────┴─────────────────────┐
        │                                            │
 ┌──────┴───────┐                          ┌─────────┴────────┐
 │  Bề mặt Chat  │                          │ Nút "hỏi AI" nhỏ  │
 │  (KISC)       │                          │ nhúng trong       │
 │               │                          │ Dashboard (KIS/   │
 │               │                          │ AVS/VQA)          │
 └───────────────┘                          └───────────────────┘
```

Vì Agent không biết gì về nơi nó được gọi, **1 lần xây đúng hợp đồng này phục vụ được cả hai bề mặt** — khớp
đúng quyết định thu hẹp phạm vi chat đã chốt trước đó (`lop-tuong-tac-thuc-dung-kiem-soat-duoc.md`), mà không
làm mất đi phần "cảm hứng KISC" đã theo đuổi từ đầu.

---

## 8. Câu hỏi còn mở — chưa chốt, cần bàn tiếp khi đào sâu

- Ngưỡng cụ thể trong bảng rule ở mục 4 (`NGƯỠNG_NHỎ`, ngưỡng tin cậy top-1/top-2...) cần tinh chỉnh bằng
  chính bộ validation set đã dựng (`bo-metric-va-validation-set.md`), không đoán suông.
- ~~`show_map` phụ thuộc dữ liệu địa lý~~ — **đã giải quyết**: đổi thành `show_concept_map` (cụm ngữ nghĩa theo
  nội dung thị giác, không cần GPS), dùng lại hạ tầng precomputed clustering đã có sẵn.
- Giới hạn cấu trúc #1 và #2 (tốc độ tuần tự, AVS cần xem nhiều kết quả song song) **chưa được note này giải
  quyết** — note này chỉ xử lý việc *bên trong* KISC, không đổi quyết định phạm vi đã chốt.

---

## 9. Nối với các note trước

Tái sử dụng nguyên vẹn không đổi: schema `Predicate`, `evaluate_predicate`, `calibrate`, `combine_scores`
(`phac-thao-compositional-scoring.md`); `BeliefState`, `update_belief`, `expected_info_gain`
(`phac-thao-belief-state-kisc.md`); nguyên tắc precomputed facet, chip transparency + undo
(`lop-tuong-tac-thuc-dung-kiem-soat-duoc.md`). Note này chỉ thêm lớp "hợp đồng giao tiếp" nối chúng với UI.
