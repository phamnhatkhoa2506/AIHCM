"use strict";

// 2026-08-21 (theo yêu cầu người dùng: "xem ta đã handle exception cho toàn bộ app tốt chưa") -
// rà lại thấy 4 chỗ gọi fetch KHÔNG hề try/catch (runAutofill/removeItem/moveItem/sp-clear) -
// lỗi mạng/server crash giữa chừng khiến nút kẹt mãi ở trạng thái "⏳ Đang thêm..." vô thời hạn,
// không có gì báo cho người dùng biết. Cũng phát hiện backend trước đây trả "Internal Server
// Error" dạng text/plain (không phải JSON) cho lỗi không lường trước -> `res.json()` tự ném
// SyntaxError -> catch bắt được NHƯNG hiện nhầm "Lỗi kết nối" dù đây là SERVER CRASH (đã sửa
// backend trả JSON {detail} qua exception_handler chung, xem backend/main.py). Helper CHUNG này
// đọc JSON AN TOÀN (không throw nếu body rỗng/không phải JSON) rồi ném lỗi RÕ RÀNG kèm status +
// đúng message server trả về (nếu có), thay vì để lỗi tự nổ ở nhiều chỗ khác nhau.
async function apiFetch(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Lỗi server (HTTP ${res.status})`);
  }
  return res;
}

// ---- Theme (nhớ lựa chọn qua localStorage; mặc định theo hệ điều hành, xem theme.css) ----
const themeToggleBtn = document.getElementById("theme-toggle");
const themeStored = localStorage.getItem("aic-theme");
if (themeStored) document.documentElement.setAttribute("data-theme", themeStored);

function currentTheme() {
  return document.documentElement.getAttribute("data-theme")
    || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
}
// Icon hiện đúng theme ĐANG áp dụng - 🌙 khi đang sáng (bấm để chuyển sang tối), ☀️ khi đang tối.
function updateThemeIcon() {
  themeToggleBtn.textContent = currentTheme() === "dark" ? "☀️" : "🌙";
}
updateThemeIcon();

themeToggleBtn.addEventListener("click", () => {
  const next = currentTheme() === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("aic-theme", next);
  updateThemeIcon();
});

// ---- Sidebar resize/collapse (tương đương sidebar Streamlit kéo giãn + đóng/mở được) ----
const appShell = document.querySelector(".app-shell");
const sidebarEl = document.getElementById("sidebar");
const sidebarResize = document.getElementById("sidebar-resize");
const sidebarToggle = document.getElementById("sidebar-toggle");
const SIDEBAR_MIN = 220;
const SIDEBAR_MAX = 520;

let sidebarWidth = parseInt(localStorage.getItem("aic-sidebar-w"), 10) || 320;
let sidebarCollapsed = localStorage.getItem("aic-sidebar-collapsed") === "1";

function applySidebarState() {
  document.documentElement.style.setProperty("--sidebar-w", sidebarWidth + "px");
  appShell.classList.toggle("sidebar-collapsed", sidebarCollapsed);
  // Tab bán nguyệt gắn liền mép sidebar (cạnh phẳng áp sát, không lệch/nổi) - xem layout.css.
  sidebarToggle.style.left = (sidebarCollapsed ? 0 : sidebarWidth) + "px";
  sidebarToggle.textContent = sidebarCollapsed ? "›" : "‹";
}
applySidebarState();

sidebarToggle.addEventListener("click", () => {
  sidebarCollapsed = !sidebarCollapsed;
  localStorage.setItem("aic-sidebar-collapsed", sidebarCollapsed ? "1" : "0");
  applySidebarState();
});

let resizing = false;
sidebarResize.addEventListener("mousedown", (e) => {
  e.preventDefault();
  resizing = true;
  appShell.classList.add("sidebar-resizing");
});
window.addEventListener("mousemove", (e) => {
  if (!resizing) return;
  sidebarWidth = Math.min(SIDEBAR_MAX, Math.max(SIDEBAR_MIN, e.clientX));
  applySidebarState();
});
window.addEventListener("mouseup", () => {
  if (!resizing) return;
  resizing = false;
  appShell.classList.remove("sidebar-resizing");
  localStorage.setItem("aic-sidebar-w", String(sidebarWidth));
});

// ---- Session id (cho submission store phía backend) ----
let sessionId = localStorage.getItem("aic-session-id");
if (!sessionId) {
  sessionId = crypto.randomUUID();
  localStorage.setItem("aic-session-id", sessionId);
}

// ---- Mode control ----
let mode = "kis";
const modeControl = document.getElementById("mode-control");
const anchorsBox = document.getElementById("anchors-box");
const anchorsRow = document.getElementById("anchors-row");
const queryInput = document.getElementById("query-input");
const qaBox = document.getElementById("qa-box");
const canvasSection = document.getElementById("canvas-section");
const spatialOpRow = document.getElementById("spatial-op-row");
const maxGapBox = document.getElementById("max-gap-box");
const maxGapInput = document.getElementById("max-gap");
const maxGapVal = document.getElementById("max-gap-val");
maxGapInput.addEventListener("input", () => { maxGapVal.textContent = Number(maxGapInput.value).toFixed(2); });

// 2026-08-21 (bug thật: "Ctrl Shift R thì canvas không load ra" - áp dụng đúng NGAY LÚC TẢI
// TRANG, không chỉ lúc bấm đổi Chế độ) - tách riêng thành hàm gọi được ở CẢ 2 nơi: listener bấm
// tab VÀ 1 lần lúc khởi động cho mode mặc định "kis" (trước đây chỉ toggle trong listener click,
// nên mode mặc định KHÔNG BAO GIỜ tự hiện canvas nếu người dùng chưa từng bấm qua tab khác).
function applyModeVisibility() {
  // TRAKE và Temporal dùng CHUNG luồng tìm kiếm nhiều mốc (anchors) - giống hệt v3
  // (is_temporal = mode in ("Temporal","TRAKE")), chỉ khác lúc NỘP BÀI (xem addToSubmission).
  const isTemporalLike = mode === "trake" || mode === "temporal";
  const hasCanvas = mode === "kis" || mode === "qa";
  anchorsBox.classList.toggle("hidden", !isTemporalLike);
  maxGapBox.classList.toggle("hidden", !isTemporalLike);
  qaBox.classList.toggle("hidden", mode !== "qa");
  queryInput.classList.toggle("hidden", mode !== "kis");
  // Canvas OCR/Object — toàn cục (KIS/Q&A) hoặc riêng-từng-mốc (TRAKE/Temporal, xem
  // renderAnchorCards). AND/OR (spatial-op-row) dùng CHUNG cho cả 2 kiểu, giống v3.
  canvasSection.classList.toggle("hidden", !hasCanvas);
  // AND/OR toàn cục CHỈ dùng cho canvas KIS/Q&A - TRAKE/Temporal giờ có AND/OR RIÊNG/mốc
  // (xem renderAnchorCards), không cần hàng chung này nữa. Với KIS/Q&A, hàng này được DỜI vào
  // trong panel canvas (nó chỉ có nghĩa khi đang vẽ khung) - xem #canvas-side-op.
  spatialOpRow.classList.toggle("hidden", !hasCanvas);
  if (hasCanvas) document.getElementById("canvas-side-op").appendChild(spatialOpRow);
  if (isTemporalLike && !anchorState.length) initAnchors();
}

modeControl.addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-mode]");
  if (!btn) return;
  mode = btn.dataset.mode;
  [...modeControl.children].forEach((b) => b.classList.toggle("active", b === btn));
  applyModeVisibility();
});
applyModeVisibility(); // áp dụng NGAY cho mode mặc định "kis" lúc tải trang (xem ghi chú trên)

// ---- Canvas toàn cục (KIS/Q&A) — 1 instance duy nhất, xem static/js/canvas.js ----
// 2026-08-21 (theo yêu cầu người dùng: "KIS và Q&A bị dư 1 khoảng trống làm xấu hệ thống") -
// canvas gấp gọn mặc định (tính năng tuỳ chọn, ít dùng); thanh gấp gọn hiện tóm tắt "N khung" +
// liệt kê nội dung để biết mình đang lọc gì mà KHÔNG cần bung ra xem.
const canvasToggle = document.getElementById("canvas-toggle");
const canvasPanel = document.getElementById("canvas-panel");
const canvasBadge = document.getElementById("canvas-badge");
const canvasBoxList = document.getElementById("canvas-box-list");

function describeBox(b) {
  if (b.kind === "object") return `🟡 ${b.label || "(chưa chọn nhãn)"}${b.minCount > 1 ? ` ×${b.minCount}` : ""}`;
  return `🟢 ${b.text ? `"${b.text}"` : "(chưa gõ chữ)"}`;
}

function renderCanvasSummary(boxes) {
  canvasBadge.classList.toggle("hidden", !boxes.length);
  canvasBadge.textContent = boxes.length ? `${boxes.length} khung` : "";
  canvasBoxList.innerHTML = boxes.length
    ? `<span class="cbl-label">Khung đã vẽ</span>` + boxes.map((b, i) =>
        `<div class="cbl-item"><span class="cbl-idx">${i + 1}</span>${describeBox(b)}</div>`).join("")
    : `<span class="cbl-empty">Chưa vẽ khung nào — chọn chế độ 🟢 OCR hoặc 🟡 Object rồi kéo chuột trên lưới bên trái.</span>`;
}

canvasToggle.addEventListener("click", () => {
  const open = canvasPanel.classList.toggle("hidden") === false;
  canvasToggle.setAttribute("aria-expanded", String(open));
  canvasToggle.querySelector(".ct-caret").textContent = open ? "▾" : "▸";
});

const globalCanvas = window.createCanvasWidget(
  document.getElementById("canvas-global-mount"), { onChange: renderCanvasSummary },
);

// ---- Mốc TRAKE/Temporal — mỗi mốc có canvas OCR/Object RIÊNG + quan hệ AND/OR RIÊNG (theo
// yêu cầu người dùng: "quan hệ riêng cho từng box" — KHÁC v3, nơi AND/OR dùng chung toàn cục) ----
let anchorState = []; // [{text, spatialOp, widget}]

function renderAnchorCards() {
  // 2026-08-21 (bug thật: "khi thêm mốc thì canvas OCR/Object tự động mất những cái đã vẽ") -
  // hàm này render LẠI TOÀN BỘ hàng thẻ Mốc (innerHTML = "") nên mọi canvas widget cũ bị huỷ và
  // tạo mới rỗng -> khung đã vẽ biến mất sạch. Chốt khung đang vẽ về `a.boxes` TRƯỚC khi wipe,
  // rồi dựng lại canvas mới từ đúng dữ liệu đó (xem canvas.js::opts.initialBoxes).
  anchorState.forEach((a) => {
    if (a.widget) { a.boxes = a.widget.getBoxes(); a.widget.destroy(); }
  });
  anchorsRow.innerHTML = "";
  anchorState.forEach((a, i) => {
    if (!a.spatialOp) a.spatialOp = "and";
    const card = document.createElement("div");
    card.className = "anchor-card";
    card.innerHTML = `
      <div class="anchor-card-head">
        <h3>Mốc ${i + 1}</h3>
        <button type="button" class="btn subtle anchor-card-close" ${anchorState.length <= 2 ? "disabled" : ""} title="Xoá mốc này">✕</button>
      </div>
      <input type="text" class="anchor-text" placeholder="Mô tả mốc ${i + 1} (vd: giậm nhảy)" />
      <div class="anchor-relation">
        <span class="anchor-relation-label">Quan hệ giữa các khung</span>
        <label class="canvas-radio"><input type="radio" name="anchor-op-${i}" value="and" ${a.spatialOp === "and" ? "checked" : ""} /> AND</label>
        <label class="canvas-radio"><input type="radio" name="anchor-op-${i}" value="or" ${a.spatialOp === "or" ? "checked" : ""} /> OR</label>
      </div>
      <div class="anchor-canvas-mount"></div>`;
    const input = card.querySelector(".anchor-text");
    input.value = a.text || "";
    input.addEventListener("input", () => { a.text = input.value; });
    card.querySelectorAll(`input[name="anchor-op-${i}"]`).forEach((r) => {
      r.addEventListener("change", () => { if (r.checked) a.spatialOp = r.value; });
    });
    card.querySelector(".anchor-card-close").addEventListener("click", () => {
      if (a.widget) a.widget.destroy();
      anchorState.splice(i, 1);
      renderAnchorCards();
    });
    anchorsRow.appendChild(card);
    a.widget = window.createCanvasWidget(
      card.querySelector(".anchor-canvas-mount"), { initialBoxes: a.boxes || [] },
    );
  });
}

function initAnchors() {
  anchorState.forEach((a) => { if (a.widget) a.widget.destroy(); });
  anchorState = [{ text: "", spatialOp: "and" }, { text: "", spatialOp: "and" }];
  renderAnchorCards();
}

document.getElementById("add-anchor").addEventListener("click", () => {
  anchorState.push({ text: "", spatialOp: "and" });
  renderAnchorCards();
});

// "Xoá tất cả các mốc" — reset về 2 mốc trống mặc định (khác ✕ từng mốc: xoá SẠCH toàn bộ).
document.getElementById("reset-anchors").addEventListener("click", initAnchors);

// ---- Q&A: "Dùng LVLM tự động trả lời" + số ứng viên gọi VQA thật (giống hệt
// _render_qa_query_inline bên v3 — mặc định TẮT, tốn phí/lần gọi API NIM) ----
const qaUseLvlm = document.getElementById("qa-use-lvlm");
const qaVqaTopN = document.getElementById("qa-vqa-top-n");
const qaVqaMinus = document.getElementById("qa-vqa-minus");
const qaVqaPlus = document.getElementById("qa-vqa-plus");

const qaQuestion = document.getElementById("qa-question");
qaUseLvlm.addEventListener("change", () => {
  const on = qaUseLvlm.checked;
  qaVqaTopN.disabled = !on;
  qaVqaMinus.disabled = !on;
  qaVqaPlus.disabled = !on;
  qaQuestion.disabled = !on; // "Câu hỏi" chỉ có tác dụng khi thực sự gọi VQA (use_lvlm=True)
});
function stepQaVqaTopN(delta) {
  const next = Math.min(20, Math.max(1, Number(qaVqaTopN.value) + delta));
  qaVqaTopN.value = next;
}
qaVqaMinus.addEventListener("click", () => stepQaVqaTopN(-1));
qaVqaPlus.addEventListener("click", () => stepQaVqaTopN(1));

// ---- Top-K slider ----
const topKInput = document.getElementById("top-k");
const topKVal = document.getElementById("topk-val");
topKInput.addEventListener("input", () => (topKVal.textContent = topKInput.value));

// ---- Build filters payload ----
function buildFilters() {
  const f = {};
  const kw = document.getElementById("f-keywords").value.trim();
  if (kw) f.keywords_any = kw.split(",").map((s) => s.trim()).filter(Boolean);
  const df = document.getElementById("f-date-from").value;
  if (df) f.date_from = df;
  const dt = document.getElementById("f-date-to").value;
  if (dt) f.date_to = dt;
  const vids = document.getElementById("f-video-ids").value.trim();
  if (vids) f.video_ids = vids.split(",").map((s) => s.trim()).filter(Boolean);
  const ocr = document.getElementById("f-ocr").value.trim();
  if (ocr) f.ocr_text = ocr;
  const asr = document.getElementById("f-asr").value.trim();
  if (asr) f.asr_text = asr;
  return f;
}

// ---- Search ----
const resultsEl = document.getElementById("results");
const steplogEl = document.getElementById("steplog");
const resultsBar = document.getElementById("results-bar");
const resultsCountEl = document.getElementById("results-count");
const autofillBtn = document.getElementById("autofill-btn");
let lastRows = [];
let lastDenseModel = "";
let lastAnchorTexts = []; // mô tả từng mốc của lần search TRAKE/Temporal gần nhất (hiện trong card)
let lastResultMode = null; // mode CỦA lần search vừa render kết quả (dùng lại khi lọc theo video_id)

async function runSearch() {
  // 2026-08-21 (bug thật "chưa đồng bộ" - card hiện ✅ nhưng số đếm ở panel vẫn 0) - KHOÁ mode
  // ngay từ đầu, dùng "searchMode" xuyên suốt hàm này thay vì đọc lại biến `mode` sống - nếu
  // người dùng đổi tab Chế độ TRONG lúc đang chờ search/submit chạy xong, mọi lệnh gọi API sau
  // đó vẫn tính đúng theo bucket của LẦN SEARCH NÀY, không lẫn sang bucket của mode mới.
  const searchMode = mode;
  const btn = document.getElementById("search-btn");
  btn.disabled = true;
  btn.textContent = "⏳ Đang tìm...";
  steplogEl.textContent = "";
  resultsEl.innerHTML = `<div id="empty-state">Đang tìm kiếm...</div>`;

  // 2026-08-21 (theo yêu cầu người dùng, giống hệt v3: "tự động reset lại khi nhấn lại nút
  // Tìm kiếm") - bấm "Tìm kiếm" LUÔN xoá trắng bucket nộp bài của mode đang chọn (bất kể query
  // giống hay khác lần trước) - KHÔNG chuyển sang bucket ID mới (tránh tạo nhiều bucket "mồ
  // côi"), chỉ xoá nội dung bucket hiện tại. Không chặn search nếu lỗi mạng ở bước này.
  try {
    await fetch(`/api/submissions/${encodeURIComponent(queryKey(searchMode))}`, {
      method: "DELETE", headers: { "X-Session-Id": sessionId },
    });
    if (searchMode === mode) {
      setSubmissionCount(0);
      if (subPanel.classList.contains("open")) refreshSubmissionList();
    }
  } catch { /* phi mạng lúc reset không nên chặn search chính */ }

  const payload = {
    mode: searchMode,
    dense_model: document.getElementById("dense-model").value,
    top_k: Number(topKInput.value),
    ocr_algorithm: document.getElementById("ocr-algorithm").value,
    score_algorithm: document.getElementById("score-algorithm").value,
    distill_model: document.getElementById("distill-model").value,
    multi_clause: document.getElementById("opt-multi-clause").checked,
    use_llm_entity: document.getElementById("opt-llm-entity").checked,
    use_region_clip_rerank: document.getElementById("opt-region-clip").checked,
    filters: buildFilters(),
  };
  if (searchMode === "trake" || searchMode === "temporal") {
    // TRAKE và Temporal dùng CHUNG luồng tìm kiếm nhiều mốc - chỉ khác lúc nộp bài. Mỗi mốc
    // mang theo canvas + AND/OR RIÊNG của nó (theo yêu cầu người dùng - khác v3, nơi AND/OR
    // dùng chung toàn cục cho mọi mốc).
    payload.anchors = [];
    payload.anchor_boxes = [];
    payload.anchor_spatial_op = [];
    anchorState.forEach((a) => {
      const text = (a.text || "").trim();
      if (!text) return;
      payload.anchors.push(text);
      payload.anchor_boxes.push(a.widget && a.widget.hasBoxes() ? a.widget.getBoxes() : null);
      payload.anchor_spatial_op.push(a.spatialOp || "and");
    });
    payload.canvas_w = globalCanvas.W;
    payload.canvas_h = globalCanvas.H;
    payload.max_gap_seconds = Number(maxGapInput.value);
  } else if (searchMode === "qa") {
    payload.query = document.getElementById("qa-event").value.trim();       // "Mô tả sự kiện" -> retrieval
    payload.question = document.getElementById("qa-question").value.trim(); // "Câu hỏi" -> chỉ dùng khi gọi VQA
    payload.use_lvlm = qaUseLvlm.checked;
    payload.vqa_top_n = Number(qaVqaTopN.value);
    payload.vlm_model = document.getElementById("vlm-ocr-model").value; // dùng CHUNG model với "VLM Verify"
  } else {
    payload.query = queryInput.value.trim();
  }
  // Canvas OCR/Object toàn cục — chỉ KIS/Q&A (TRAKE/Temporal dùng canvas + AND/OR riêng/mốc ở trên).
  if ((searchMode === "kis" || searchMode === "qa") && globalCanvas.hasBoxes()) {
    payload.boxes = globalCanvas.getBoxes();
    payload.canvas_w = globalCanvas.W;
    payload.canvas_h = globalCanvas.H;
    payload.spatial_op = document.querySelector('input[name="spatial-op"]:checked').value;
  }

  try {
    const res = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    // 2026-08-21 - trước đây KHÔNG kiểm tra res.ok: request thiếu query/anchors (400) trả về
    // {"detail": "..."} (không có "rows") -> lastRows lặng lẽ thành [] rồi hiện "Không có kết
    // quả", người dùng không biết là do THIẾU INPUT chứ không phải search không ra gì.
    if (!res.ok) {
      resultsEl.innerHTML = `<div id="empty-state">${escapeHtml(data.detail || `Lỗi server (HTTP ${res.status})`)}</div>`;
      return;
    }
    // Temporal: nộp bài dùng 1 frame DUY NHẤT = median của chuỗi mốc (giống hệt v3
    // `_result_row_to_submission`, mode="temporal": statistics.median rồi int() cắt về nguyên -
    // Math.trunc, KHÔNG làm tròn). Tính sẵn ngay đây để card/resultCardId/addToSubmission dùng
    // CHUNG 1 giá trị nhất quán (row.frame_id) - vẫn giữ row.frame_ids để hiển thị cả chuỗi.
    lastRows = (data.rows || []).map((r) =>
      searchMode === "temporal" && r.frame_ids ? { ...r, frame_id: medianFrame(r.frame_ids) } : r
    );
    lastDenseModel = payload.dense_model;
    lastAnchorTexts = payload.anchors && (searchMode === "trake" || searchMode === "temporal") ? payload.anchors : [];
    // Người dùng đã đổi tab Chế độ trong lúc chờ search này chạy xong -> kết quả giờ KHÔNG còn
    // khớp với UI đang hiển thị (vd đang xem KIS nhưng đây là kết quả của lần search QA cũ) -
    // bỏ qua, không render đè lên UI hiện tại (search mới của mode hiện tại sẽ tự chạy riêng).
    if (searchMode !== mode) return;
    lastResultMode = searchMode;
    resultFilterInput.value = "";
    resultFilterBox.classList.toggle("hidden", lastRows.length === 0);
    renderSteplog(data.step_log || [], data.error);
    renderResults(lastRows, searchMode);
    await updateResultsBar(searchMode);
    await markSubmittedCards(searchMode);
  } catch (err) {
    resultsEl.innerHTML = `<div id="empty-state">Lỗi kết nối: ${err}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "🔍 Tìm kiếm";
  }
}

document.getElementById("search-btn").addEventListener("click", runSearch);
queryInput.addEventListener("keydown", (e) => { if (e.key === "Enter") runSearch(); });

// 2026-08-21 (theo yêu cầu người dùng: "xóa phần log step... không cần thiết nữa") - KHÔNG còn
// hiện chi tiết từng bước xử lý (thời gian lọc thô/chưng cất/encode...) nữa - #steplog giờ CHỈ
// còn dùng làm dòng thông báo lỗi/trạng thái chung (autofill, nộp bài lỗi, jump-to-result...).
function renderSteplog(steps, error) {
  steplogEl.innerHTML = error ? `<span class="err">${error}</span>` : "";
}

// ID duy nhất/kết quả (video_id + frame_id hoặc frame_ids) - dùng để nhảy từ danh sách nộp bài
// tới đúng card trong lưới kết quả (xem jumpToResult bên dưới) - CÙNG công thức ở cả 2 nơi.
// Giống hệt statistics.median() + int() bên v3 (cắt về nguyên, KHÔNG làm tròn) - dùng cho
// Temporal (xem runSearch/addToSubmission).
function medianFrame(frameIds) {
  const sorted = [...frameIds].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const med = sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  return Math.trunc(med);
}

// m: mode của item (khi biết) - TRAKE nộp NGUYÊN CHUỖI mốc nên định danh theo frame_ids; Temporal
// nộp 1 frame DUY NHẤT (median, xem addToSubmission) nên định danh theo frame_id như KIS/QA, dù
// row gốc từ search vẫn có frame_ids (hiển thị cả chuỗi cho người xem, xem renderResults).
// Đáp án Q&A là chữ NGƯỜI DÙNG tự gõ (hoặc VLM sinh ra) - phải escape trước khi nhét vào
// innerHTML, nếu không dấu <, & trong đáp án sẽ phá vỡ HTML của panel.
function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function resultCardId(item, m) {
  const useChain = m === "trake" && item.frame_ids;
  const frames = useChain ? item.frame_ids.join("_") : item.frame_id;
  return `card-${item.video_id}-${frames}`;
}

// ---- Lọc kết quả ĐÃ tìm được theo video_id (client-side, không gọi lại search) - dùng chung
// cho mọi mode (KIS/QA/TRAKE/Temporal), xem index.html #result-filter-box. ----
const resultFilterBox = document.getElementById("result-filter-box");
const resultFilterInput = document.getElementById("result-filter");
resultFilterInput.addEventListener("input", () => {
  const q = resultFilterInput.value.trim().toLowerCase();
  const filtered = q ? lastRows.filter((r) => r.video_id.toLowerCase().includes(q)) : lastRows;
  renderResults(filtered, lastResultMode);
  markSubmittedCards(lastResultMode);
  resultsCountEl.textContent = q
    ? `${filtered.length}/${lastRows.length} kết quả (lọc "${resultFilterInput.value.trim()}") — model: ${lastDenseModel}`
    : `${lastRows.length} kết quả — model: ${lastDenseModel}`;
});

function renderResults(rows, resultMode) {
  if (!rows.length) {
    resultsEl.innerHTML = `<div id="empty-state">Không có kết quả.</div>`;
    return;
  }
  resultsEl.innerHTML = "";
  const isChain = (resultMode === "trake" || resultMode === "temporal") && !!(rows[0] && rows[0].frame_ids);
  rows.forEach((row, i) => {
    if (isChain) renderChainCard(row, i, resultMode);
    else renderSingleCard(row, i, resultMode);
  });
}

function renderSingleCard(row, i, resultMode) {
  const wrap = document.createElement("div");
  wrap.className = "card-wrap";
  wrap.id = resultCardId(row, resultMode);
  const frameLabel = row.frame_ids ? row.frame_ids.join(", ") : row.frame_id;
  wrap.innerHTML = `
    <span class="rank">#${i + 1}</span>
    <div class="card">
      <img loading="lazy" src="${row.thumb_url}" alt="${row.video_id}" />
      <div class="meta">
        <span>${row.video_id} · ${frameLabel}</span>
        ${row.score != null ? `<span class="score">${row.score.toFixed(3)}</span>` : ""}
      </div>
      <div class="actions">
        <button class="btn icon-btn add-btn" data-tooltip="Nộp">📤</button>
        ${row.frame_id != null ? `<button class="btn icon-btn shot-btn" data-tooltip="Xem shot">🎬</button>` : ""}
        ${row.frame_id != null ? `<button class="btn icon-btn playback-btn" data-tooltip="Playback">🎯</button>` : ""}
        <button class="btn icon-btn vlm-btn" data-tooltip="VLM Verify">🔍</button>
      </div>
      <div class="vlm-result hidden"></div>
    </div>`;
  const addBtn = wrap.querySelector(".add-btn");
  addBtn.addEventListener("click", (e) => {
    const btn = e.currentTarget;
    if (resultMode === "qa" && !btn.classList.contains("submitted")) {
      openQaAnswerModal(row, btn, resultMode);
    } else {
      addToSubmission(row, btn, null, resultMode);
    }
  });
  wrap.querySelector(".vlm-btn").addEventListener("click", (e) =>
    vlmVerify(row.thumb_url, wrap.querySelector(".vlm-result"), e.target));
  wrap.querySelector("img").addEventListener("click", () => openLightbox(row.thumb_url, row.video_id));
  const shotBtn = wrap.querySelector(".shot-btn");
  if (shotBtn) shotBtn.addEventListener("click", () => openShotPlayer(row.video_id, row.frame_id));
  const playbackBtn = wrap.querySelector(".playback-btn");
  if (playbackBtn) {
    playbackBtn.addEventListener("click", () => {
      // 2026-08-21 (theo yêu cầu người dùng: "giữ history cho toàn bộ quá trình đổi") - gắn lịch
      // sử LÊN `row` (sống suốt vòng đời card, không mất khi đóng/mở lại dialog nhiều lần).
      if (!row._pbHistory) row._pbHistory = [[row.frame_id]];
      window.openPlayback({
        videoId: row.video_id,
        frameIds: [row.frame_id],
        labels: ["Frame"],
        mode: resultMode,
        answer: row.answer_text || "",
        history: row._pbHistory,
        onConfirm: (idx, newFrameId, fps) => {
          // 2026-08-21 (bug thật: "frame này được nộp, nhưng đổi frame qua playback thì nút
          // submit vẫn hiện đã nộp") - frame_id đổi -> identity nộp bài đổi theo (_row_key backend
          // khoá theo frame_id) - entry CŨ trong bucket giờ tham chiếu 1 frame không còn hiển thị
          // nữa (stale). Gỡ entry cũ (toggle-off đúng frame cũ) rồi để nút quay về 📤 "chưa nộp" -
          // người dùng tự bấm Nộp lại cho frame MỚI nếu muốn (không tự nộp hộ).
          if (addBtn.classList.contains("submitted") && row.frame_id !== newFrameId) {
            addToSubmission({ video_id: row.video_id, frame_id: row.frame_id, frame_ids: null }, addBtn, null, resultMode);
          }
          row.frame_id = newFrameId;
          row.thumb_url = `/api/frame?video_id=${encodeURIComponent(row.video_id)}&t=${newFrameId / fps}`;
          wrap.querySelector("img").src = row.thumb_url;
          wrap.querySelector(".meta span").textContent = `${row.video_id} · ${row.frame_id}`;
        },
        onSubmit: (btn, answerText) => {
          if (resultMode === "qa") row.answer_text = answerText;
          addToSubmission(row, addBtn, resultMode === "qa" ? answerText : null, resultMode);
          window.closePlayback();
        },
      });
    });
  }
  resultsEl.appendChild(wrap);
}

// TRAKE/Temporal — giống hệt bố cục v3 (_r_is_temporal): timeline trực quan chấm điểm từng mốc
// theo đúng vị trí thời gian, rồi 1 cột/mốc (ảnh + mô tả mốc + frame/t + nút riêng), nộp 1 lần
// cho CẢ CHUỖI ở dưới cùng (khác KIS/QA — mỗi mốc KHÔNG có nút Nộp riêng).
function renderChainCard(row, i, resultMode) {
  const wrap = document.createElement("div");
  wrap.className = "card-wrap card-wrap-chain";
  wrap.id = resultCardId(row, resultMode);
  const n = row.frame_ids.length;
  const times = row.pts_times || row.frame_ids.map(() => 0);
  const tMax = Math.max(...times) * 1.05 || 1.0;
  // Dùng CHUNG 1 mảng % cho cả chấm timeline lẫn cột mốc bên dưới - để cột mốc thẳng hàng ĐÚNG
  // tâm với chấm thời gian tương ứng (thay vì dàn đều theo flex-gap như trước).
  const pcts = times.map((t) => (100 * t / tMax).toFixed(1));
  const dots = times.map((t, k) =>
    `<div class="chain-dot" style="left:${pcts[k]}%"><span class="chain-dot-mark"></span><span class="chain-dot-label">#${k + 1} ${t.toFixed(1)}s</span></div>`
  ).join("");
  const anchorTexts = lastAnchorTexts || [];
  const cols = row.thumb_urls.map((u, k) => `
    <div class="chain-anchor">
      <img loading="lazy" class="chain-img" data-idx="${k}" src="${u}" alt="${row.video_id} mốc ${k + 1}" />
      <div class="chain-anchor-caption">
        ${anchorTexts[k] ? `<span class="chain-anchor-text">${anchorTexts[k]}</span><br/>` : ""}
        frame <code>${row.frame_ids[k]}</code> · t=${times[k].toFixed(2)}s
      </div>
      <div class="chain-anchor-actions">
        <button class="btn icon-btn chain-shot-btn" data-idx="${k}" data-tooltip="Xem shot">🎬</button>
        <button class="btn icon-btn chain-vlm-btn" data-idx="${k}" data-tooltip="VLM Verify">🔍</button>
      </div>
      <div class="vlm-result hidden"></div>
    </div>`).join("");
  wrap.innerHTML = `
    <span class="rank">#${i + 1}</span>
    <div class="card card-chain">
      <div class="chain-header">
        <span>${row.video_id}</span>
        ${row.score != null ? `<span class="score">score=${row.score.toFixed(3)}</span>` : ""}
      </div>
      <div class="chain-track">
        <div class="chain-timeline">${dots}</div>
        <div class="chain-anchors">${cols}</div>
      </div>
      <div class="chain-footer">
        <button class="btn icon-btn add-btn" data-tooltip="Nộp">📤</button>
        <button class="btn icon-btn chain-playback-btn" data-tooltip="Playback (xem/chỉnh cả chuỗi)">🎯</button>
      </div>
    </div>`;

  const chainAddBtn = wrap.querySelector(".add-btn");
  chainAddBtn.addEventListener("click", (e) => addToSubmission(row, e.currentTarget, null, resultMode));
  wrap.querySelectorAll(".chain-img").forEach((img) => {
    img.addEventListener("click", () => openLightbox(img.src, `${row.video_id} mốc ${Number(img.dataset.idx) + 1}`));
  });
  wrap.querySelectorAll(".chain-shot-btn").forEach((btn) => {
    const k = Number(btn.dataset.idx);
    btn.addEventListener("click", () => openShotPlayer(row.video_id, row.frame_ids[k]));
  });
  wrap.querySelectorAll(".chain-vlm-btn").forEach((btn) => {
    const k = Number(btn.dataset.idx);
    const resultBox = wrap.querySelectorAll(".chain-anchor")[k].querySelector(".vlm-result");
    btn.addEventListener("click", () => vlmVerify(row.thumb_urls[k], resultBox, btn));
  });
  wrap.querySelector(".chain-playback-btn").addEventListener("click", () => {
    // 2026-08-21 (theo yêu cầu người dùng: "giữ history cho toàn bộ quá trình đổi") - 1 mảng lịch
    // sử RIÊNG/mốc, gắn lên `row` để sống xuyên suốt nhiều lần mở/đóng Playback.
    if (!row._pbHistory) row._pbHistory = row.frame_ids.map((f) => [f]);
    window.openPlayback({
      videoId: row.video_id,
      frameIds: [...row.frame_ids],
      labels: anchorTexts.length ? anchorTexts.map((t, k) => `Mốc ${k + 1}: ${t}`) : row.frame_ids.map((_, k) => `Mốc ${k + 1}`),
      mode: resultMode,
      history: row._pbHistory,
      onConfirm: (idx, newFrameId, fps) => {
        // 2026-08-21 (bug thật: "frame này được nộp, nhưng đổi frame qua playback thì nút submit
        // vẫn hiện đã nộp") - TRAKE khoá theo NGUYÊN CHUỖI frame_ids, Temporal khoá theo frame_id
        // (median, xem addToSubmission) - đổi 1 mốc là đổi identity nộp bài -> entry CŨ giờ stale
        // (tham chiếu chuỗi/median không còn hiển thị nữa). Gỡ entry cũ trước khi mutate `row`,
        // để nút quay về 📤 - người dùng tự bấm Nộp lại nếu muốn (không tự nộp hộ).
        if (chainAddBtn.classList.contains("submitted") && row.frame_ids[idx] !== newFrameId) {
          if (resultMode === "trake") {
            addToSubmission({ video_id: row.video_id, frame_id: null, frame_ids: [...row.frame_ids] }, chainAddBtn, null, resultMode);
          } else {
            addToSubmission({ video_id: row.video_id, frame_id: row.frame_id, frame_ids: null }, chainAddBtn, null, resultMode);
          }
        }
        row.frame_ids[idx] = newFrameId;
        row.pts_times[idx] = newFrameId / fps;
        row.thumb_urls[idx] = `/api/frame?video_id=${encodeURIComponent(row.video_id)}&t=${newFrameId / fps}`;
        if (resultMode === "temporal") row.frame_id = medianFrame(row.frame_ids);
        const anchorEl = wrap.querySelectorAll(".chain-anchor")[idx];
        anchorEl.querySelector(".chain-img").src = row.thumb_urls[idx];
        anchorEl.querySelector(".chain-anchor-caption").innerHTML =
          (anchorTexts[idx] ? `<span class="chain-anchor-text">${anchorTexts[idx]}</span><br/>` : "") +
          `frame <code>${newFrameId}</code> · t=${row.pts_times[idx].toFixed(2)}s`;
      },
      // Chỉ Temporal có nút nộp TRONG dialog (TRAKE nộp bằng nút "📤 Nộp" bên ngoài, xem
      // playback.js::renderSubmitArea) - dùng LẠI đúng chainAddBtn để icon ✅/📤 đồng bộ.
      onSubmit: () => { addToSubmission(row, chainAddBtn, null, resultMode); window.closePlayback(); },
    });
  });
  resultsEl.appendChild(wrap);
}

// ---- Lightbox phóng to/thu nhỏ ảnh (giống nút mở rộng của st.image bên Streamlit) ----
const lightbox = document.getElementById("lightbox");
const lightboxImg = document.getElementById("lightbox-img");
let lightboxScale = 1;

function openLightbox(src, alt) {
  lightboxImg.src = src;
  lightboxImg.alt = alt || "";
  lightboxScale = 1;
  lightboxImg.style.transform = "scale(1)";
  lightboxImg.classList.remove("zoomed");
  lightbox.classList.remove("hidden");
}
function closeLightbox() {
  lightbox.classList.add("hidden");
  lightboxImg.src = "";
}
function setLightboxScale(next) {
  lightboxScale = Math.min(4, Math.max(1, next));
  lightboxImg.style.transform = `scale(${lightboxScale})`;
  lightboxImg.classList.toggle("zoomed", lightboxScale > 1);
}

document.getElementById("lightbox-close").addEventListener("click", closeLightbox);
lightbox.addEventListener("click", (e) => { if (e.target === lightbox) closeLightbox(); });
lightboxImg.addEventListener("click", () => setLightboxScale(lightboxScale > 1 ? 1 : 2.5));
lightboxImg.addEventListener("wheel", (e) => {
  e.preventDefault();
  setLightboxScale(lightboxScale + (e.deltaY < 0 ? 0.3 : -0.3));
}, { passive: false });
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !lightbox.classList.contains("hidden")) closeLightbox();
});

// ---- Video player nổi — phát đúng đoạn SHOT chứa frame (giống nút "🎬" bên v3, ranh giới
// shot lấy từ dense_meta.parquet::shot_idx, xem GET /api/shot_clip) ----
const videoPlayer = document.getElementById("video-player");
const vpTitle = document.getElementById("vp-title");
const vpVideo = document.getElementById("vp-video");

function openShotPlayer(videoId, frameId) {
  vpTitle.textContent = `${videoId} · frame ${frameId}`;
  vpVideo.src = `/api/shot_clip?video_id=${encodeURIComponent(videoId)}&frame_id=${frameId}`;
  videoPlayer.classList.remove("hidden");
}
function closeShotPlayer() {
  videoPlayer.classList.add("hidden");
  vpVideo.pause();
  vpVideo.src = "";
}
document.getElementById("vp-close").addEventListener("click", closeShotPlayer);

// ---- VLM Verify (xác minh OCR bằng VLM, lazy on-demand — xem backend/core/vlm_verify.py) ----
async function vlmVerify(thumbUrl, resultBox, btn) {
  const path = new URL(thumbUrl, window.location.origin).searchParams.get("path");
  const model = document.getElementById("vlm-ocr-model").value;

  btn.disabled = true;
  btn.textContent = "⏳";
  resultBox.classList.remove("hidden");
  resultBox.textContent = "Đang gọi VLM đọc chữ...";

  try {
    const res = await fetch("/api/vlm_verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, model }),
    });
    const data = await res.json();
    if (data.error) {
      resultBox.textContent = "Lỗi: " + data.error;
      resultBox.classList.add("err");
    } else {
      resultBox.textContent = data.text ? `Chữ đọc được: "${data.text}"` : "(Không đọc được chữ nào trong ảnh)";
      resultBox.classList.remove("err");
    }
  } catch (err) {
    resultBox.textContent = "Lỗi kết nối: " + err;
    resultBox.classList.add("err");
  } finally {
    btn.disabled = false;
    btn.textContent = "🔍";
  }
}

// ---- Submission bubble (thiết kế lại giống Streamlit — xem online/app.py `_render_submission_
// panel`: mỗi "câu hỏi" có 1 bucket 100-dòng RIÊNG, định danh bằng 1 ID ổn định TÁCH RIÊNG khỏi
// ô "Name" (Name chỉ cosmetic, đổi thoải mái không mất dữ liệu) - xem queryKey() bên dưới) ----
const SUBMISSION_MAX = 100;
const subPill = document.getElementById("sub-pill");
const subPanel = document.getElementById("sub-panel");
const subCount = document.getElementById("sub-count");
const spCountFull = document.getElementById("sp-count-full");
const spName = document.getElementById("sp-name");
const spList = document.getElementById("sp-list");

// mode ("kis"/"qa"/"trake") -> {id, n}: "id" là bucket ĐANG active, "n" là số thứ tự cho lần
// bấm "🆕 Câu hỏi mới" tiếp theo (bucket_kis_1, bucket_kis_2, ...).
const submissionState = {
  kis: { id: "kis_default", n: 1 },
  qa: { id: "qa_default", n: 1 },
  trake: { id: "trake_default", n: 1 },
};

// Temporal dùng CHUNG bucket lưu trữ với KIS (đúng y hệt v3: _FILENAME_MODE_SUFFIX =
// {"KIS":"kis","Temporal":"kis","TRAKE":"trake","Q&A":"qa"}) - vì cả 2 CUỐI CÙNG đều nộp
// "video_id,frame_id" (Temporal rút chuỗi mốc về 1 frame median trước khi nộp, xem
// addToSubmission), không cần bucket riêng.
function modeSuffix(m = mode) { return m === "temporal" ? "kis" : m; }
// 2026-08-21 (bug thật "chưa đồng bộ" - card hiện ✅ nhưng số đếm ở panel vẫn 0): các nơi gọi
// queryKey() bên trong callback BẤT ĐỒNG BỘ (sau khi search/submit đã await xong) đọc biến
// `mode` SỐNG - nếu người dùng đổi tab Chế độ TRONG lúc đang chờ, các lệnh gọi API sau đó lại
// tính nhầm sang bucket của mode MỚI trong khi card/kết quả đang hiển thị vẫn thuộc mode CŨ ->
// lệch. Cho phép truyền `m` tường minh (mode tại lúc search/render, xem renderResults/runSearch)
// để KHÔNG phụ thuộc `mode` sống nữa - chỉ fallback về `mode` hiện tại khi gọi rời rạc (vd nút
// sp-csv/sp-clear trong panel, nơi không có "mode tại thời điểm" nào khác ngoài mode đang xem).
function queryKey(m) { return submissionState[modeSuffix(m || mode)].id; }

subPill.addEventListener("click", () => {
  subPanel.classList.toggle("open");
  if (subPanel.classList.contains("open")) refreshSubmissionList();
});
document.getElementById("sp-close").addEventListener("click", () => subPanel.classList.remove("open"));

document.getElementById("sp-new-query").addEventListener("click", () => {
  const st = submissionState[modeSuffix()];
  st.n += 1;
  st.id = `${modeSuffix()}_${st.n}`;
  spName.value = `query-p1-${st.n}-${modeSuffix()}`;
  refreshSubmissionList();
  // 2026-08-21 (bug thật: "bấm câu hỏi mới thì kết quả mất, nhưng nút nộp trong frame vẫn còn
  // đánh dấu đã nộp") - đổi sang bucket MỚI (rỗng) nhưng các card đang hiển thị trong lưới kết
  // quả vẫn giữ nguyên trạng thái ✅ từ bucket CŨ - chưa có gì gọi lại markSubmittedCards() để
  // kiểm tra lại theo bucket mới. Gọi lại ngay đây để mọi card tự trả về 📤 (bucket mới rỗng).
  markSubmittedCards();
});

async function refreshSubmissionList() {
  const res = await fetch(`/api/submissions/${encodeURIComponent(queryKey())}`, {
    headers: { "X-Session-Id": sessionId },
  });
  const items = res.ok ? await res.json() : [];
  setSubmissionCount(items.length); // đồng bộ luôn cả nhãn nút "Tự động điền" (bug đã gặp: đổi
  // bucket qua "Câu hỏi mới" chỉ set text trực tiếp trước đây, KHÔNG qua setSubmissionCount() ->
  // nút autofill đứng yên ở trạng thái "Đã đủ 100" của bucket CŨ dù bucket mới đang rỗng).
  spList.innerHTML = "";
  items.forEach((it, i) => {
    const label = it.frame_ids ? `${it.video_id} · frame ${it.frame_ids.join(",")}` : `${it.video_id} · frame ${it.frame_id ?? ""}`;
    const tag = (it.mode || mode).toUpperCase();
    const row = document.createElement("div");
    row.className = "sp-item";
    row.innerHTML = `
      <button class="sp-label" title="Nhảy tới kết quả này">
        <span class="sp-icon">🚀</span>${i + 1}. <span class="sp-tag">${tag}</span> ${label}
        ${it.mode === "qa" ? `<span class="sp-answer${it.answer_text ? "" : " empty"}">${it.answer_text ? escapeHtml(it.answer_text) : "⚠ chưa có đáp án"}</span>` : ""}
      </button>
      <button class="btn" data-act="up">↑</button>
      <button class="btn" data-act="down">↓</button>
      <button class="btn" data-act="del">✕</button>`;
    row.querySelector(".sp-label").addEventListener("click", () => jumpToResult(it));
    row.querySelector('[data-act="up"]').addEventListener("click", () => moveItem(i, "up"));
    row.querySelector('[data-act="down"]').addEventListener("click", () => moveItem(i, "down"));
    row.querySelector('[data-act="del"]').addEventListener("click", () => removeItem(i));
    spList.appendChild(row);
  });
}

// Nhảy tới card kết quả tương ứng trong lưới (chỉ tìm được nếu vẫn đang hiển thị đúng kết quả
// đó - giống hạn chế "🔖" bên Streamlit cũ, không lưu lại toàn bộ lịch sử tìm kiếm).
function jumpToResult(item) {
  const el = document.getElementById(resultCardId(item, item.mode));
  if (!el) {
    steplogEl.innerHTML = `<span class="err">Không tìm thấy "${item.video_id} · ${item.frame_ids ? item.frame_ids.join(",") : item.frame_id}" trong kết quả đang hiển thị — thử tìm lại câu hỏi tương ứng.</span>`;
    return;
  }
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.add("flash");
  setTimeout(() => el.classList.remove("flash"), 1200);
}

function setSubmissionCount(n) {
  subCount.textContent = n;
  spCountFull.textContent = `${n}/${SUBMISSION_MAX}`;
  updateAutofillLabel(n);
}

// ---- Tự động điền — giống hệt `_render_autofill_button` bên v3: thêm các dòng CHƯA CÓ trong
// danh sách nộp bài, đúng thứ tự rank hiện có (cao -> thấp), tới khi đủ SUBMISSION_MAX. ----
function updateAutofillLabel(n) {
  const remaining = SUBMISSION_MAX - n;
  if (remaining > 0) {
    autofillBtn.textContent = `⬇️ Tự động điền đủ ${SUBMISSION_MAX} (còn thiếu ${remaining})`;
    autofillBtn.disabled = false;
  } else {
    autofillBtn.textContent = `⬇️ Đã đủ ${SUBMISSION_MAX}`;
    autofillBtn.disabled = true;
  }
}

async function updateResultsBar(resultMode) {
  const m = resultMode || mode;
  if (!lastRows.length) { resultsBar.classList.add("hidden"); return; }
  resultsBar.classList.remove("hidden");
  resultsCountEl.textContent = `${lastRows.length} kết quả — model: ${lastDenseModel}`;
  const res = await fetch(`/api/submissions/${encodeURIComponent(queryKey(m))}`, {
    headers: { "X-Session-Id": sessionId },
  });
  const items = res.ok ? await res.json() : [];
  if (m === mode) setSubmissionCount(items.length); // xem ghi chú ở addToSubmission
}

async function runAutofill() {
  const m = mode; // khoá mode ngay lúc bấm, xuyên suốt hàm (xem ghi chú addToSubmission)
  autofillBtn.disabled = true;
  autofillBtn.textContent = "⏳ Đang thêm...";
  try {
    await runAutofillInner(m);
  } catch (err) {
    // 2026-08-21 (bug thật: hàm này trước đây KHÔNG try/catch - lỗi mạng/server giữa chừng khiến
    // nút kẹt mãi ở "⏳ Đang thêm..." vô thời hạn, không báo gì cho người dùng).
    steplogEl.innerHTML = `<span class="err">Tự động điền lỗi: ${escapeHtml(err.message)}</span>`;
    updateAutofillLabel(Number(subCount.textContent) || 0); // trả nút về nhãn đúng, không kẹt "⏳"
  }
}

async function runAutofillInner(m) {
  // 2026-08-21 (bug thật: "tự động điền chưa hoạt động với Q&A - chưa lấy kết quả top cũng như
  // không đưa đáp án lên") - Q&A của BTC là 1 CÂU TRẢ LỜI DUY NHẤT cho cả 100 dòng (chỉ khác
  // video/frame) nên tính 1 đáp án CHUNG cho cả loạt. Lần đầu port y hệt v3 (_render_autofill_
  // button) - CHỈ tìm đáp án của ĐÚNG dòng rank #1 hiện tại - nhưng thực tế người dùng thường nộp
  // tay 1 dòng bất kỳ (không nhất thiết đúng rank #1 tại THỜI ĐIỂM bấm Tự động điền, vd rank có
  // thể đổi giữa các lần search khác nhau) -> v3 im lặng bỏ qua đáp án đã có, autofill ra RỖNG
  // (đúng lỗi người dùng gặp). Q&A chỉ có ĐÚNG 1 đáp án thật cho toàn bộ submission bất kể rank -
  // nên thông minh hơn: lấy đáp án Q&A ĐÃ NỘP BẤT KỲ nào trong bucket (không ràng buộc đúng
  // rank #1), fallback về answer_text của rank #1 (VQA thật nếu bật) nếu bucket còn trống hoàn
  // toàn.
  let qaDefaultAnswer = null;
  if (m === "qa" && lastRows.length) {
    const res0 = await fetch(`/api/submissions/${encodeURIComponent(queryKey(m))}`, {
      headers: { "X-Session-Id": sessionId },
    });
    const existing = res0.ok ? await res0.json() : [];
    // "rank 1" theo đúng ý người dùng = đáp án ĐẦU TIÊN đã tự nộp (vị trí #1 trong panel nộp
    // bài, existing[0] - mảng lưu đúng thứ tự đã nộp/đã sắp lại bằng nút ↑↓), KHÔNG phải rank
    // điểm search - fallback "bất kỳ dòng Q&A nào có đáp án" nếu dòng #1 lại là KIS/chưa có đáp
    // án (vd người dùng nộp dòng #1 trước khi gõ đáp án).
    const first = existing[0];
    const anyAnswered = (first && first.mode === "qa" && first.answer_text)
      ? first
      : existing.find((r) => r.mode === "qa" && r.answer_text);
    qaDefaultAnswer = anyAnswered ? anyAnswered.answer_text : (lastRows[0].answer_text || "");
  }
  const items = lastRows.map((r) => ({
    video_id: r.video_id, frame_id: r.frame_id ?? null, frame_ids: r.frame_ids ?? null, mode: m,
    answer_text: m === "qa" ? qaDefaultAnswer : null,
  }));
  const res = await apiFetch(`/api/submissions/${encodeURIComponent(queryKey(m))}/autofill`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Session-Id": sessionId },
    body: JSON.stringify({ items }),
  });
  const data = await res.json();
  if (m === mode) {
    setSubmissionCount(data.items.length);
    if (subPanel.classList.contains("open")) refreshSubmissionList();
    await markSubmittedCards(m);
  }
  steplogEl.innerHTML = `Đã tự động thêm ${data.added} dòng.`;
}
autofillBtn.addEventListener("click", runAutofill);

// ---- Form nổi nhập câu trả lời Q&A — CHỈ hiện khi bấm "Nộp" trên 1 frame (không hiện sẵn cho
// cả lưới), gõ xong bấm Enter mới thực sự đưa vào danh sách nộp bài (Esc/bấm ra ngoài = huỷ). ----
const qaAnswerModal = document.getElementById("qa-answer-modal");
const qamTitle = document.getElementById("qam-title");
const qamInput = document.getElementById("qam-input");
let qaModalRow = null;
let qaModalBtn = null;
let qaModalMode = null;

function openQaAnswerModal(row, btn, resultMode) {
  qaModalRow = row;
  qaModalBtn = btn;
  qaModalMode = resultMode;
  const frameLabel = row.frame_ids ? row.frame_ids.join(", ") : row.frame_id;
  qamTitle.textContent = `${row.video_id} · frame ${frameLabel}`;
  qamInput.value = row.answer_text || "";
  qaAnswerModal.classList.remove("hidden");
  qamInput.focus();
  qamInput.select();
}
function closeQaAnswerModal() {
  qaAnswerModal.classList.add("hidden");
  qaModalRow = null;
  qaModalBtn = null;
  qaModalMode = null;
}
qaAnswerModal.addEventListener("click", (e) => { if (e.target === qaAnswerModal) closeQaAnswerModal(); });
qamInput.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    closeQaAnswerModal();
  } else if (e.key === "Enter") {
    const row = qaModalRow, btn = qaModalBtn, resultMode = qaModalMode;
    const answer = qamInput.value.trim();
    closeQaAnswerModal();
    addToSubmission(row, btn, answer, resultMode);
  }
});

// Giống hệt `_render_submit_button` bên v3 - "📤 Nộp" TỰ CHUYỂN "✅" khi đã nộp, bấm lại =
// hoàn tác (xoá khỏi danh sách ngay tại card, không cần mở bảng nộp bài riêng).
function setSubmitBtnState(btn, submitted) {
  btn.classList.toggle("submitted", submitted);
  btn.textContent = submitted ? "✅" : "📤";
  btn.dataset.tooltip = submitted ? "Đã nộp — bấm để hoàn tác" : "Nộp";
}

async function addToSubmission(row, btn, answerText, resultMode) {
  // 2026-08-21 (bug thật "chưa đồng bộ" - card hiện ✅ nhưng số đếm ở panel vẫn 0) - "m" là mode
  // TẠI THỜI ĐIỂM card này được render (resultMode), KHÔNG phải `mode` sống - card thuộc bucket
  // nào là CỐ ĐỊNH theo lúc search ra nó, dù người dùng có đổi tab Chế độ sau đó hay không.
  const m = resultMode || mode;
  // TRAKE nộp NGUYÊN CHUỖI mốc (frame_ids) - Temporal nộp 1 frame DUY NHẤT (median, đã tính sẵn
  // vào row.frame_id ở runSearch) giống hệt KIS/QA - xem ghi chú đầu file _submission_to_csv.
  const item = m === "trake"
    ? { video_id: row.video_id, frame_id: null, frame_ids: row.frame_ids ?? null, mode: m, answer_text: null }
    : { video_id: row.video_id, frame_id: row.frame_id ?? null, frame_ids: null, mode: m, answer_text: answerText ?? null };
  btn.disabled = true;
  try {
    const res = await fetch(`/api/submissions/${encodeURIComponent(queryKey(m))}/toggle`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Session-Id": sessionId },
      body: JSON.stringify(item),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      steplogEl.innerHTML = `<span class="err">${err.detail || "Không nộp được."}</span>`;
      return;
    }
    const data = await res.json();
    setSubmitBtnState(btn, data.submitted);
    // Chỉ cập nhật pill/panel (đại diện cho bucket ĐANG XEM) nếu bucket vừa đổi CHÍNH LÀ bucket
    // của mode hiện tại - nếu người dùng đã đổi tab Chế độ trong lúc thao tác, pill vẫn phải
    // hiện đúng số của mode đang xem, không phải số của bucket vừa nộp (khác mode).
    if (m === mode) {
      setSubmissionCount(data.items.length);
      if (subPanel.classList.contains("open")) refreshSubmissionList();
    }
  } finally {
    btn.disabled = false;
  }
}

// Sau mỗi lần search, đánh dấu sẵn card nào ĐÃ có trong danh sách nộp bài của mode/bucket hiện
// tại (vd người dùng tự động điền rồi quay lại xem) - để nút hiện đúng "✅" ngay từ đầu.
async function markSubmittedCards(resultMode) {
  const res = await fetch(`/api/submissions/${encodeURIComponent(queryKey(resultMode || mode))}`, {
    headers: { "X-Session-Id": sessionId },
  });
  if (!res.ok) return;
  const items = await res.json();
  // Temporal dùng CHUNG bucket lưu trữ với KIS (giống hệt v3, xem modeSuffix()) - bucket có thể
  // lẫn cả 2 loại item, nên tính id theo ĐÚNG mode CỦA TỪNG ITEM (it.mode), không phải mode
  // đang xem lưới kết quả, tránh khớp nhầm.
  const submittedKeys = new Set(items.map((it) => resultCardId(it, it.mode)));
  resultsEl.querySelectorAll(".card-wrap").forEach((wrap) => {
    const btn = wrap.querySelector(".add-btn");
    if (btn) setSubmitBtnState(btn, submittedKeys.has(wrap.id));
  });
}

// 2026-08-21 - cả 3 hàm dưới đây trước KHÔNG hề try/catch (lỗi mạng/index không hợp lệ/server
// crash giữa chừng bay thẳng thành uncaught promise rejection - im lặng trong console, người
// dùng bấm ↑↓/✕/Xoá hết KHÔNG THẤY GÌ xảy ra, cũng không biết là lỗi hay chỉ đang chờ).
async function removeItem(index) {
  try {
    const res = await apiFetch(`/api/submissions/${encodeURIComponent(queryKey())}/${index}`, {
      method: "DELETE", headers: { "X-Session-Id": sessionId },
    });
    const items = await res.json();
    setSubmissionCount(items.length);
    refreshSubmissionList();
  } catch (err) {
    steplogEl.innerHTML = `<span class="err">Không xoá được dòng: ${escapeHtml(err.message)}</span>`;
  }
}

async function moveItem(index, direction) {
  try {
    await apiFetch(`/api/submissions/${encodeURIComponent(queryKey())}/move/${index}?direction=${direction}`, {
      method: "POST", headers: { "X-Session-Id": sessionId },
    });
    refreshSubmissionList();
  } catch (err) {
    steplogEl.innerHTML = `<span class="err">Không đổi thứ tự được: ${escapeHtml(err.message)}</span>`;
  }
}

document.getElementById("sp-clear").addEventListener("click", async () => {
  try {
    await apiFetch(`/api/submissions/${encodeURIComponent(queryKey())}`, {
      method: "DELETE", headers: { "X-Session-Id": sessionId },
    });
    setSubmissionCount(0);
    refreshSubmissionList();
    // 2026-08-21 (bug thật: "bấm xoá hết mà mấy cái nút submit chưa hồi về") - cùng lỗi đã gặp
    // với "🆕 Câu hỏi mới": xoá bucket trên SERVER xong nhưng không có gì gọi lại
    // markSubmittedCards() để báo cho lưới kết quả ĐANG HIỂN THỊ biết bucket giờ rỗng -> card vẫn
    // kẹt ✅ cũ.
    markSubmittedCards();
  } catch (err) {
    steplogEl.innerHTML = `<span class="err">Không xoá hết được: ${escapeHtml(err.message)}</span>`;
  }
});

document.getElementById("sp-csv").addEventListener("click", () => {
  // "Name" thuần cosmetic (chỉ đổi TÊN FILE tải về, không liên quan queryKey() lưu trữ) - giống
  // hệt v3 (_active_submission_key tách riêng khỏi ô "Name").
  const filename = spName.value.trim() || `query-p1-x-${modeSuffix()}`;
  // Điều hướng trình duyệt (KHÔNG qua fetch()) để trình duyệt tự tải file đúng filename từ
  // Content-Disposition - vì vậy KHÔNG gắn được header X-Session-Id, phải gửi qua query param
  // "session_id" thay thế (xem submissions.py::export_csv, bug thật "vẫn chưa tải được").
  const params = new URLSearchParams({ filename, session_id: sessionId });
  window.location.href = `/api/submissions/${encodeURIComponent(queryKey())}/export.csv?${params}`;
});
