# Dataset Profile — cơ chế chống LLM bịa đặt, tuỳ biến theo từng bộ dữ liệu

> Vá lỗ hổng của `agent-planning-nang-cap.md`: Query Analyzer + Planner đang giả định ngầm về dữ liệu
> (có OCR, có địa điểm, có object...). Giả định sai → LLM hỏi/gợi ý những thứ **không tồn tại trong dữ liệu**
> = bịa đặt. Note này thiết kế cơ chế **tách tri thức về dữ liệu ra khỏi logic agent**.
> Ngày: 2026-07-09.

---

## 0. Vấn đề cụ thể — LLM bịa như thế nào

| Tình huống | LLM bịa gì | Vì sao |
|---|---|---|
| Dữ liệu MSR-VTT (video web đa chủ đề, tiếng Anh) | Hỏi "cảnh đó trong nhà hay ngoài trời?" | Trục này **không phân biệt được gì** — MSR-VTT có đủ mọi loại, chia gần 50/50 nhưng chẳng thu hẹp về đúng video |
| Dữ liệu lifelog (ego-centric, wearable) | Gọi `ocr_search` tìm chữ trên banner | Lifelog **hầu như không có chữ overlay** — tool trả rác |
| Dữ liệu lifelog | Hỏi "người đó là nam hay nữ?" | Camera ego-centric — người đeo **không xuất hiện trong khung hình** của chính mình |
| Dữ liệu tin tức HTV | Hỏi "bạn đang ở đâu lúc đó?" | Đây là video **của người khác** (bản tin), không phải trải nghiệm của người dùng |
| Bất kỳ bộ nào | Gợi ý option "quán cà phê ngoài trời" khi corpus **không có cụm nào như vậy** | LLM sinh option từ trí nhớ huấn luyện, không từ dữ liệu thật |

> **Nguyên nhân gốc:** Query Analyzer và Planner đang lấy tri thức từ **prior của LLM**, không phải từ
> **thống kê thật của corpus**. Bất cứ khi nào agent nói về dữ liệu mà không *nhìn* dữ liệu, nó đang đoán.

---

## 1. Nguyên tắc chống bịa — 3 tầng phòng vệ

```
Tầng 1 — DATASET PROFILE (tĩnh, viết tay 1 lần/bộ dữ liệu)
         "Bộ này CÓ gì / KHÔNG CÓ gì" -> gate tool & gate câu hỏi

Tầng 2 — CORPUS STATISTICS (động, tính tự động lúc indexing)
         "Trục nào thật sự phân biệt được?" -> chỉ hỏi trục có info-gain thật

Tầng 3 — GROUNDED OPTIONS (bắt buộc, runtime)
         Mọi option/gợi ý phải TRÍCH TỪ dữ liệu thật, không được LLM tự nghĩ ra
```

Ba tầng này bổ trợ nhau: tầng 1 chặn tool sai, tầng 2 chặn câu hỏi vô dụng, tầng 3 chặn nội dung bịa.

---

## 2. Tầng 1 — Dataset Profile (file cấu hình, viết tay, 1 lần mỗi bộ)

Một file khai báo **năng lực và giới hạn** của từng bộ dữ liệu. Agent **không được** dùng tool/hỏi trục nào
không được khai báo ở đây.

```jsonc
// profiles/msrvtt.json
{
  "dataset_id": "msrvtt",
  "description": "Video web đa chủ đề, tiếng Anh, clip ngắn 10-32s, third-person",
  "perspective": "third_person",           // ai đang quay? -> quyết định câu hỏi nào hợp lệ
  "available_channels": ["clip"],          // CHỈ có CLIP - không có OCR/ASR/object detection
  "unavailable_channels": ["ocr", "asr", "object", "gps", "timestamp"],
  "queryable_axes": [],                    // KHÔNG có trục facet nào đáng hỏi (xem tầng 2)
  "forbidden_questions": [
    "bạn đang ở đâu", "bạn nhớ lúc đó là mấy giờ",   // video của NGƯỜI KHÁC, không phải của user
    "người trong video là bạn phải không"
  ],
  "task_framing": "third_person_search",   // "tìm video mô tả X" (KHÔNG phải "tìm lúc BẠN làm X")
  "notes": "Mỗi video có ~20 caption tiếng Anh. Query nên bằng tiếng Anh."
}
```

```jsonc
// profiles/lifelog.json
{
  "dataset_id": "lifelog",
  "description": "Ego-centric, wearable camera, quay liên tục nhiều giờ",
  "perspective": "first_person_ego",
  "available_channels": ["clip", "timestamp", "object"],
  "unavailable_channels": ["ocr"],         // hầu như không có chữ overlay
  "weak_channels": ["asr"],                // có tiếng nhưng ồn môi trường, ASR kém tin cậy
  "queryable_axes": ["time_of_day", "indoor_outdoor", "activity_type", "location_cluster"],
  "forbidden_questions": [
    "người đeo camera mặc gì",             // KHÔNG thấy được chính mình
    "khuôn mặt bạn trông thế nào",
    "camera đang quay từ góc nào"          // luôn là góc nhìn thứ nhất
  ],
  "strong_signals": ["timestamp"],         // lifelog có timeline liên tục -> lọc thời gian CỰC MẠNH
  "task_framing": "first_person_recall",   // "tìm lúc BẠN làm X"
  "notes": "Góc quay rung lắc, ánh sáng đổi liên tục -> CLIP kém hơn bình thường. Shot boundary mờ."
}
```

```jsonc
// profiles/aic_news.json  (giả định nếu AIC vẫn dùng tin tức HTV)
{
  "dataset_id": "aic_news",
  "perspective": "broadcast",
  "available_channels": ["clip", "ocr", "asr", "object", "metadata"],
  "unavailable_channels": ["gps"],
  "strong_signals": ["ocr", "metadata"],   // bản tin GIÀU chữ + metadata YouTube chính xác
  "queryable_axes": ["topic", "publish_date", "channel", "indoor_outdoor"],
  "forbidden_questions": ["bạn đang ở đâu", "lúc đó bạn đang làm gì"],
  "task_framing": "third_person_search"
}
```

### Cách profile này được dùng (thay đổi trực tiếp `agent-planning-nang-cap.md`)

```python
def build_tool_registry(dataset_profile) -> list[Tool]:
    registry = []
    for tool in ALL_TOOLS:
        if tool.required_channel in dataset_profile.unavailable_channels:
            continue                       # LOẠI HẲN - Planner không bao giờ thấy tool này
        if tool.required_channel in dataset_profile.weak_channels:
            tool.cost = "high"             # vẫn có, nhưng bị hạ ưu tiên
            tool.reliability = "low"
        registry.append(tool)
    return registry
```

→ **LLM Planner chỉ nhìn thấy tool thật sự dùng được.** Không thể gọi `ocr_search` trên lifelog vì tool đó
**không tồn tại trong registry được đưa cho nó**. Đây là chống bịa ở mức *kiến trúc*, không phải mức prompt
("nhớ đừng dùng OCR nhé" — kiểu này LLM vẫn vi phạm).

---

## 3. Tầng 2 — Corpus Statistics (tự động, tính lúc indexing)

Dataset Profile khai báo trục *có thể* hỏi. Nhưng trục đó có **thật sự phân biệt được** trong corpus cụ thể
này không, thì phải **đo**, không phải đoán.

```python
# Chạy 1 lần lúc indexing, lưu ra file
def compute_axis_statistics(corpus_embeddings, profile) -> dict:
    stats = {}
    for axis in profile.queryable_axes:
        # Với mỗi giá trị khả dĩ của trục, đếm bao nhiêu frame rơi vào
        distribution = classify_corpus_by_axis(corpus_embeddings, axis)
        stats[axis] = {
            "distribution": distribution,           # vd {"indoor": 0.48, "outdoor": 0.52}
            "entropy": entropy(distribution),       # cao = chia đều = hỏi có giá trị
            "max_bucket_ratio": max(distribution.values()),
            "usable": entropy(distribution) > MIN_ENTROPY   # <- CỔNG QUYẾT ĐỊNH
        }
    return stats
```

**Quy tắc:** agent **chỉ được hỏi** trục có `usable == True`.

| Ví dụ thống kê thật | Kết luận |
|---|---|
| `indoor_outdoor: {indoor: 0.48, outdoor: 0.52}`, entropy cao | ✓ Hỏi được — chia gần đôi, thu hẹp ~50% |
| `indoor_outdoor: {indoor: 0.97, outdoor: 0.03}` (dữ liệu phẫu thuật) | ✗ **Không hỏi** — 97% trong nhà, hỏi vô nghĩa |
| `has_text: {yes: 0.02, no: 0.98}` (lifelog) | ✗ **Không hỏi về chữ**, và cũng **không gọi ocr_search** |

→ Đây là **information gain đo từ dữ liệu thật**, đúng nguyên lý đã đặt ra ở lỗ hổng 5.3 — nhưng giờ được
tính **trước, tự động**, thay vì để LLM đoán trục nào đáng hỏi.

**Điểm mạnh phụ:** cơ chế này **tự thích nghi** khi đổi dataset — không cần sửa code, chỉ chạy lại thống kê.

---

## 4. Tầng 3 — Grounded Options (bắt buộc: option phải TRÍCH từ dữ liệu)

Đây là tầng chống bịa **quan trọng nhất**, và là chỗ dễ sai nhất.

### ✗ Cách SAI (LLM tự nghĩ option)
```
Prompt: "Query mơ hồ. Hãy đưa ra 3 lựa chọn để làm rõ."
LLM  -> ["quán cà phê ngoài trời", "công viên", "bãi biển"]
```
Vấn đề: **corpus có thể không có cái nào trong 3 cái đó**. Người dùng chọn "bãi biển" → 0 kết quả. LLM đã bịa
ra một thế giới không tồn tại trong dữ liệu.

### ✓ Cách ĐÚNG (option trích từ cụm thật của candidate set hiện tại)
```python
def generate_grounded_options(belief_state, k=3) -> list[Option]:
    # 1. Cluster CHÍNH tập ứng viên hiện tại (không phải toàn corpus, không phải trí nhớ LLM)
    clusters = kmeans(belief_state.candidate_embeddings, n_clusters=k)

    options = []
    for c in clusters:
        # 2. Lấy ảnh đại diện THẬT (gần tâm cụm nhất)
        representative = nearest_to_centroid(c)
        # 3. LLM chỉ được ĐẶT TÊN cho cụm đã có, KHÔNG được nghĩ ra cụm mới
        label = vlm_caption(representative, prompt="Mô tả cảnh này bằng 3-5 từ")
        options.append(Option(
            id=c.id,
            label=label,                      # tên do VLM đặt cho ảnh THẬT
            thumbnail=representative,          # <- người dùng THẤY được ảnh thật
            candidate_count=len(c.members)     # <- biết chọn xong còn bao nhiêu
        ))
    return options
```

**Ba khác biệt quyết định:**
1. **Cụm có sẵn trong dữ liệu** → chọn xong chắc chắn còn ≥1 ứng viên, không bao giờ về 0.
2. **Hiển thị thumbnail thật** → người dùng nhìn ảnh mà chọn, không phải đọc chữ do LLM bịa (và đúng insight
   đã bàn: người nhận diện hình ảnh nhanh và chính xác hơn diễn đạt bằng lời).
3. **LLM chỉ làm 1 việc duy nhất: đặt tên cho thứ đã tồn tại.** Đây là tác vụ LLM làm rất tốt và gần như không
   bịa được — khác hẳn "nghĩ ra option từ hư không".

> **Nguyên tắc tổng quát, áp cho MỌI widget:**
> *LLM không bao giờ được sinh ra **nội dung** về dữ liệu. Nó chỉ được **diễn giải** thứ đã trích từ dữ liệu.*
> Ánh xạ vào các Action đã định nghĩa:
> - `show_options` → option = cụm thật, label = VLM đặt tên cho ảnh đại diện thật ✓
> - `show_concept_map` → cụm thật từ K-means trên candidate set ✓
> - `show_image_grid` → ảnh thật từ kết quả retrieval ✓ (vốn đã đúng)
> - `reply_text` → chỉ được nói về **số liệu thật** ("còn 12 ứng viên"), không được mô tả nội dung nó chưa thấy

---

## 5. Bảng tổng: mỗi tầng chặn kiểu bịa nào

| Kiểu bịa | Tầng chặn | Cơ chế |
|---|---|---|
| Gọi tool cho kênh không tồn tại (`ocr_search` trên lifelog) | Tầng 1 | Tool bị **xoá khỏi registry**, LLM không thấy |
| Hỏi câu vô nghĩa với dữ liệu ("bạn mặc gì" với ego-centric) | Tầng 1 | `forbidden_questions` |
| Hỏi trục không phân biệt được (indoor/outdoor trên data 97% indoor) | Tầng 2 | `usable == False` → không đưa vào |
| Sinh option không tồn tại trong corpus ("bãi biển" khi không có) | Tầng 3 | Option **trích từ cụm thật**, kèm thumbnail + số lượng |
| Mô tả nội dung video mà chưa thực sự xem | Tầng 3 | `reply_text` chỉ được nói số liệu, mọi mô tả phải qua VLM nhìn ảnh thật |
| Dùng sai khung tác vụ ("tìm lúc BẠN..." cho video người khác) | Tầng 1 | `task_framing` + `perspective` |

---

## 6. Thay đổi cần áp vào các note trước

**`agent-planning-nang-cap.md`:**
- Tool Registry: thêm bước `build_tool_registry(dataset_profile)` — lọc tool theo `available_channels` **trước
  khi** đưa cho Planner.
- Query Analyzer: các trường trong Query Profile (`has_text_in_scene`, `has_speech_cue`...) phải được **giao
  với** `available_channels` — query có nhắc chữ nhưng dataset không có OCR → `has_text_in_scene = false`.

**`kien-truc-2-tang-agent-va-ui.md`:**
- `show_options.payload.choices` → bắt buộc thêm 2 trường: `thumbnail` (ảnh thật) và `candidate_count`.
- Thêm ràng buộc tường minh: *option phải sinh bằng `generate_grounded_options()`, cấm LLM tự nghĩ.*

**`phac-thao-belief-state-kisc.md`:**
- `CANDIDATE_AXES` không còn là danh sách viết cứng, mà **đọc từ `dataset_profile.queryable_axes` đã lọc qua
  `corpus_statistics.usable`**.

---

## 7. Lợi ích phụ — đổi dataset không cần sửa code

Cơ chế này biến "tri thức về dữ liệu" thành **cấu hình**, không phải **code**:

```
prototype/
  profiles/
    msrvtt.json        <- test pipeline, đo Recall@K
    lifelog.json       <- nếu AIC dùng lifelog
    aic_news.json      <- nếu AIC vẫn dùng tin tức
  stats/
    msrvtt_axes.json   <- tự sinh lúc indexing
    ...
```

Đổi từ MSR-VTT sang dữ liệu AIC thật = **đổi 1 file profile + chạy lại thống kê**, không sửa logic agent.
Đây cũng chính là cách trả lời cho tình huống hiện tại: **chưa biết AIC 2026 dùng lifelog hay tin tức** —
viết sẵn cả 2 profile, khi biết chắc thì chỉ cần trỏ đúng file.

---

## 8. Câu hỏi còn mở

- **Ai viết Dataset Profile?** Hiện là viết tay. Có thể bán tự động: chạy VLM trên mẫu ngẫu nhiên 100 frame,
  hỏi "dữ liệu này có chữ không? góc nhìn thứ mấy? có mặt người không?" → sinh draft profile, người duyệt lại.
  Đáng làm nếu phải xử lý nhiều bộ dữ liệu.
- **Ngưỡng `MIN_ENTROPY`** để coi 1 trục là `usable` — phải tinh chỉnh bằng validation set thật.
- **Trục facet nào cần precompute?** Với lifelog, `time_of_day` lấy từ timestamp (rẻ, chính xác). Với
  `indoor_outdoor`/`activity_type` thì cần chạy CLIP zero-shot classify toàn corpus lúc indexing (tốn 1 lần,
  sau đó tra bảng miễn phí — đúng nguyên lý "precompute lúc indexing" đã chốt).
