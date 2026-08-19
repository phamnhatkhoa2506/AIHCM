# Kiến trúc triển khai tăng dần — dễ sửa, dễ bảo trì, dễ mở rộng

> Chốt cách hiện thực hoá toàn bộ chuỗi note thiết kế thành code, theo nguyên tắc: **mỗi lần thêm chức năng
> mới = thêm 1 file, không sửa file cũ**. Nối tiếp `agent-planning-nang-cap.md`,
> `dataset-profile-chong-bia-dat.md`, `kien-truc-2-tang-agent-va-ui.md`.
> Ngày: 2026-07-09.

---

## 0. Nguyên tắc chủ đạo

> **Open/Closed:** hệ thống **mở để mở rộng** (thêm tool/kênh/dataset mới dễ dàng), **đóng để sửa đổi**
> (không phải động vào code lõi đã chạy ổn).

Cụ thể hoá thành 3 quy tắc kỹ thuật:

1. **Mọi năng lực là một Plugin tự đăng ký.** Thêm kênh OCR mới → tạo `channels/ocr.py`, khai báo decorator,
   xong. Không sửa `pipeline.py`, không sửa `agent.py`.
2. **Tri thức về dữ liệu nằm trong config, không nằm trong code.** Đổi dataset = đổi file JSON
   (đã thiết kế ở `dataset-profile-chong-bia-dat.md`).
3. **Giao tiếp giữa các tầng qua interface cố định.** Tầng trên không biết tầng dưới cài đặt thế nào — chỉ
   biết nó trả về đúng kiểu dữ liệu đã hẹn.

---

## 1. Cấu trúc thư mục — mỗi thư mục là 1 "điểm mở rộng"

```
aic/
├── core/                      # LÕI - viết 1 lần, gần như không sửa nữa
│   ├── types.py               #   Predicate, BeliefState, Action, Event, SearchResult, Plan, Step
│   ├── registry.py            #   cơ chế đăng ký plugin (decorator @register_channel, @register_tool...)
│   └── protocols.py           #   interface (Protocol) mà mọi plugin phải tuân theo
│
├── channels/                  # ĐIỂM MỞ RỘNG 1: thêm nguồn tín hiệu mới
│   ├── clip.py                #   @register_channel("clip")
│   ├── ocr.py                 #   @register_channel("ocr")      <- thêm file = xong
│   ├── asr.py                 #   @register_channel("asr")
│   ├── object.py              #   @register_channel("object")
│   └── metadata.py            #   @register_channel("metadata")
│
├── tools/                     # ĐIỂM MỞ RỘNG 2: thêm cách tìm kiếm mới
│   ├── clip_search.py         #   @register_tool(needs="clip", cost="low")
│   ├── ocr_search.py          #   @register_tool(needs="ocr", cost="low")
│   ├── region_search.py       #   @register_tool(needs=["clip","object"], cost="medium")
│   ├── temporal_search.py     #   @register_tool(needs="clip", cost="medium")
│   ├── hyde_image.py          #   @register_tool(needs="clip", cost="high")
│   └── vlm_verify.py          #   @register_tool(needs="clip", cost="high")
│
├── agent/                     # TẦNG AGENT (headless - không biết gì về UI)
│   ├── analyzer.py            #   Query -> QueryProfile
│   ├── planner.py             #   QueryProfile + Registry -> [Plan]
│   ├── critic.py              #   [Plan] -> [Plan] đã lọc/sửa
│   ├── executor.py            #   Plan -> chạy từng Step, fusion tích luỹ
│   ├── monitor.py             #   phát hiện kẹt -> tín hiệu replan
│   └── belief.py              #   BeliefState + update_belief
│
├── fusion/                    # ĐIỂM MỞ RỘNG 3: thêm cách hợp điểm mới
│   ├── rrf.py                 #   @register_fusion("rrf")
│   ├── weighted_min.py        #   @register_fusion("weighted_min")   <- combine AND đã thiết kế
│   └── combsum.py             #   @register_fusion("combsum")
│
├── profiles/                  # ĐIỂM MỞ RỘNG 4: thêm dataset mới (chỉ là JSON!)
│   ├── msrvtt.json
│   ├── lifelog.json
│   └── aic_news.json
│
├── ui/                        # TẦNG UI (chỉ render Action, không có logic)
│   ├── server.py              #   FastAPI: nhận Action (JSON) từ Session, trả cho browser
│   └── static/index.html      #   ĐIỂM MỞ RỘNG 5: render Action -> widget, thuần JS
│                                   (KHÔNG dùng registry Python @register_widget như dự tính
│                                    ban đầu - widget là HTML/CSS/JS hiển thị, không có lý do
│                                    bọc qua 1 lớp Python; thêm widget mới = thêm 1 nhánh
│                                    trong hàm renderAction() của index.html)
│
├── eval/                      # ĐO LƯỜNG - dựng sớm, dùng suốt
│   ├── metrics.py             #   Recall@K, MRR, per-stage metrics
│   ├── harness.py             #   chạy validation set -> bảng kết quả
│   └── validation/            #   bộ query + ground truth
│
└── index/                     # BUILD INDEX (offline)
    ├── build.py
    └── stats.py               #   corpus statistics (tầng 2 chống bịa)
```

---

## 2. Cơ chế Registry — trái tim của khả năng mở rộng

### 2.1. Định nghĩa interface (viết 1 lần, trong `core/protocols.py`)

```python
from typing import Protocol

class Channel(Protocol):
    """Một nguồn tín hiệu từ dữ liệu (CLIP embedding, OCR text, object boxes...)."""
    name: str
    def build(self, keyframes: list[Keyframe]) -> None: ...     # chạy lúc indexing
    def load(self) -> None: ...                                  # nạp index đã build
    def is_available(self) -> bool: ...                          # có dữ liệu chưa?

class Tool(Protocol):
    """Một cách tìm kiếm/lọc/rerank, dùng 1 hoặc nhiều Channel."""
    name: str
    needs: list[str]              # tên các channel bắt buộc phải có
    cost: str                     # "low" | "medium" | "high"
    gated_by: str | None          # điều kiện trên QueryProfile (vd "has_text_in_scene")

    def run(self, query: Predicate, candidates: CandidateSet) -> SearchResult: ...
    def estimate_selectivity(self, query: Predicate) -> float: ...   # cho CRITIC ước lượng

class FusionOp(Protocol):
    name: str
    def fuse(self, results: list[SearchResult]) -> SearchResult: ...
```

### 2.2. Registry + decorator (viết 1 lần, trong `core/registry.py`)

```python
CHANNELS: dict[str, Channel] = {}
TOOLS:    dict[str, Tool]    = {}
FUSIONS:  dict[str, FusionOp] = {}
WIDGETS:  dict[str, Widget]  = {}

def register_channel(name: str):
    def deco(cls):
        CHANNELS[name] = cls()
        return cls
    return deco

def register_tool(name: str, needs: list[str], cost: str, gated_by: str | None = None):
    def deco(cls):
        inst = cls()
        inst.name, inst.needs, inst.cost, inst.gated_by = name, needs, cost, gated_by
        TOOLS[name] = inst
        return cls
    return deco
# ... tương tự cho fusion, widget
```

### 2.3. Thêm chức năng mới = thêm 1 file (không sửa gì cả)

**Ví dụ: 3 tháng nữa muốn thêm kênh "emotion" (như vitrivr-Engine 2026):**

```python
# channels/emotion.py   <- FILE MỚI, KHÔNG SỬA FILE NÀO KHÁC
from core.registry import register_channel

@register_channel("emotion")
class EmotionChannel:
    name = "emotion"
    def build(self, keyframes):
        # chạy model nhận diện cảm xúc, lưu vector
        ...
    def is_available(self):
        return (INDEX_DIR / "emotion.npy").exists()
```

```python
# tools/emotion_search.py   <- FILE MỚI
from core.registry import register_tool

@register_tool(name="emotion_search", needs=["emotion"], cost="low",
               gated_by="has_emotion_cue")
class EmotionSearch:
    def run(self, query, candidates): ...
    def estimate_selectivity(self, query): ...
```

**Xong.** Planner tự động thấy tool mới trong registry. Critic tự động ước lượng được. Không sửa `planner.py`,
`executor.py`, hay bất kỳ file lõi nào. Chỉ cần thêm `"emotion"` vào `available_channels` của dataset profile
nào có kênh đó.

---

## 3. Cách Agent dùng Registry (tự động lọc, đúng cơ chế chống bịa)

```python
# agent/planner.py  -- viết 1 lần, không sửa khi thêm tool mới
def get_usable_tools(profile: DatasetProfile, qprofile: QueryProfile) -> list[Tool]:
    usable = []
    for tool in TOOLS.values():
        # Tầng 1 chống bịa: dataset có đủ channel không?
        if not all(ch in profile.available_channels for ch in tool.needs):
            continue
        # Tầng 1: channel đã build index chưa?
        if not all(CHANNELS[ch].is_available() for ch in tool.needs):
            continue
        # gated_by: query có đặc tính phù hợp không?
        if tool.gated_by and not getattr(qprofile, tool.gated_by, False):
            continue
        usable.append(tool)
    return usable
```

→ Hàm này **không bao giờ phải sửa** dù thêm bao nhiêu tool. Nó lọc theo *thuộc tính* của tool, không theo
danh sách cứng.

---

## 4. Lộ trình triển khai — 5 lát cắt dọc, mỗi lát CHẠY ĐƯỢC

Nguyên tắc: **không xây theo tầng ngang** (xong hết channel rồi mới tới tool rồi mới tới agent — kiểu này
mất rất lâu mới thấy hệ thống chạy). Xây theo **lát cắt dọc**: mỗi giai đoạn là một hệ thống **hoàn chỉnh,
chạy được, đo được**, chỉ khác độ phong phú.

### Giai đoạn 1 — "Xương sống" (mục tiêu: đo được Recall@K)
```
core/types.py + registry.py + protocols.py
channels/clip.py
tools/clip_search.py
index/build.py
eval/metrics.py + harness.py
profiles/msrvtt.json
```
**Chạy được:** `search("a man playing guitar")` → top-K + **Recall@K trên MSR-VTT**.
**Chưa có:** agent, planner, UI. Chỉ là pipeline retrieval thuần + đo lường.
→ Đây là **nền móng bắt buộc**: có số liệu rồi thì mọi cải tiến sau mới biết là tốt hay xấu.

### Giai đoạn 2 — "Nhiều kênh + fusion"
```
+ channels/ocr.py, asr.py, object.py, metadata.py
+ tools/ocr_search.py, object_filter.py
+ fusion/rrf.py, weighted_min.py
+ index/stats.py                      (corpus statistics)
```
**Chạy được:** tìm bằng nhiều kênh, hợp điểm, đo xem kênh nào cải thiện Recall@K bao nhiêu.
→ Lần đầu tiên trả lời được câu hỏi thực nghiệm: **OCR có thật sự giúp không?** (thay vì tin lý thuyết)

### Giai đoạn 3 — "Agent biết lập kế hoạch"
```
+ agent/analyzer.py    (Query Profile)
+ agent/planner.py     (sinh plan)
+ agent/critic.py      (kiểm chứng plan)
+ agent/executor.py    (chạy plan, fusion tích luỹ)
+ agent/monitor.py     (phát hiện kẹt)
```
**Chạy được:** đưa query → agent tự chọn tool, tự lập plan, tự phát hiện kẹt và đổi hướng.
**Đo được:** so sánh Recall@K của "agent tự chọn" vs "chỉ dùng clip_search" → biết agent có giá trị thật không.

### Giai đoạn 4 — "Tương tác" (đã xong — xem `agent/session.py`, `agent/simple_agent.py`)
```
+ agent/belief.py (dùng BeliefState có sẵn), agent/session.py (2 phase + undo)
+ ui/static/index.html  (5 widget: show_options, show_concept_map, show_image_grid,
                          show_region_picker, show_video - render bằng JS, không phải Python plugin)
+ ui/server.py
```
**Chạy được:** vòng lặp người-máy, thu hẹp dần qua nhiều lượt. Bổ sung thêm nguyên tắc 2 Phase
(chưa có trong bản gốc note này): Phase REFINE tua lại được (undo qua Turn snapshot), Phase
EXECUTE khoá 1 chiều (chỉ thoát bằng huỷ toàn bộ, không undo từng bước) - xem docstring đầu file
`agent/session.py` để biết chi tiết, vì quyết định này chưa được ghi thành note riêng.
**Đo được:** Turn efficiency, Info-gain/lượt.

### Giai đoạn 5 — "Nâng cao" (chỉ làm nếu còn thời gian)
```
+ tools/hyde_image.py, vlm_verify.py, region_search.py, temporal_search.py
+ in-session learning (trọng số tool)
```
Mỗi tool ở đây là **1 file độc lập** — thêm/bỏ tự do, không ảnh hưởng gì tới 4 giai đoạn trước.

---

## 5. Vì sao lộ trình này an toàn

| Rủi ro thường gặp | Cách kiến trúc này chặn |
|---|---|
| Xây 2 tháng mới thấy hệ thống chạy | Giai đoạn 1 đã chạy được + đo được (vài ngày) |
| Thêm tính năng làm hỏng tính năng cũ | Plugin độc lập; `eval/harness.py` chạy lại toàn bộ validation set mỗi lần đổi → phát hiện regression ngay |
| Không biết cải tiến nào thật sự có ích | Mỗi giai đoạn đều **đo bằng Recall@K** trên cùng validation set |
| Đổi dataset phải viết lại | Chỉ đổi file `profiles/*.json` |
| AIC dùng lifelog thay vì tin tức | Tắt `ocr` khỏi `available_channels`, hệ thống tự thích nghi |
| Agent phức tạp nhưng vô dụng | Giai đoạn 3 **so sánh có/không có agent** — nếu không cải thiện, biết ngay để bỏ |

---

## 6. Quy tắc bảo trì (viết ra để tự tuân thủ)

1. **Không bao giờ `if tool_name == "ocr"` trong code lõi.** Nếu cần hành vi riêng cho 1 tool → thêm *thuộc
   tính* vào Tool protocol, đừng hard-code tên.
2. **Mọi tool mới phải cài `estimate_selectivity()`** — nếu không, Critic không ước lượng được và tool đó
   thành "hộp đen" phá vỡ khả năng lập kế hoạch.
3. **Mọi thay đổi phải chạy `eval/harness.py`** trước khi giữ lại — nguyên tắc đã đặt ra từ
   `bo-metric-va-validation-set.md`: trực giác "cải tiến này chắc tốt hơn" *thường sai*.
4. **Không để logic nghiệp vụ trong `ui/`.** UI chỉ render Action và gửi Event. Nếu thấy mình viết `if` về
   dữ liệu trong UI → nó thuộc về `agent/`.
5. **Dataset knowledge chỉ nằm trong `profiles/`.** Nếu thấy mình viết "lifelog thì không có OCR" trong code
   → sai chỗ, đưa vào JSON.

---

## 7. Bước tiếp theo cụ thể

Giai đoạn 1 là thứ nên làm ngay khi có MSR-VTT:
- Refactor `prototype/` hiện có (đang là script rời) thành cấu trúc `aic/` ở trên.
- Giữ nguyên logic đã chạy được (CLIP index + search), chỉ **bọc lại** thành `channels/clip.py` +
  `tools/clip_search.py`.
- Thêm `eval/harness.py` + `profiles/msrvtt.json` → có số liệu Recall@K đầu tiên.

Sau đó mọi thứ khác chỉ là **thêm file**.
