"use strict";
// Canvas vẽ khung OCR/Object — port từ v3 (online/objects_canvas_component/index.html, HTML5
// canvas thuần), bỏ giao tiếp postMessage/Streamlit. 2026-08-21: đổi thành FACTORY
// (createCanvasWidget) để tạo được NHIỀU canvas độc lập — 1 canvas TOÀN CỤC cho KIS/Q&A + 1
// canvas RIÊNG/mốc cho TRAKE/Temporal (giống hệt v3: mỗi mốc có canvas riêng, xem
// _render_filter_canvas(f"trake_{i}", full_width=True) trong app.py cũ).
(function () {
  const W = 480, H = 270;
  let LABELS = [];
  fetch("/api/labels").then((r) => r.json()).then((d) => { LABELS = d.labels || []; }).catch(() => {});

  // Giống hệt tier1_filter.py::_strip_accents bên Python (không phân biệt dấu).
  function stripAccents(s) {
    return s.normalize("NFD").replace(/[̀-ͯ]/g, "").replace(/đ/g, "d").replace(/Đ/g, "d").toLowerCase();
  }

  const COLOR = { ocr: "#22c55e", object: "#f5c518" };

  // opts.onChange(boxes) — gọi mỗi lần số khung/nội dung khung đổi (vẽ thêm/sửa/hoàn tác/xoá),
  // để bên ngoài hiện tóm tắt "đang vẽ N khung" trên thanh gấp gọn mà không cần hỏi liên tục.
  // opts.initialBoxes — khung vẽ sẵn lúc khởi tạo (dạng getBoxes() trả về). Cần cho TRAKE/Temporal:
  // mỗi lần thêm/xoá mốc, hàng thẻ Mốc bị render LẠI từ đầu nên canvas cũng bị tạo lại - phải
  // khôi phục đúng khung đã vẽ, nếu không người dùng mất sạch (xem app.js::renderAnchorCards).
  function createCanvasWidget(mountEl, opts = {}) {
    mountEl.innerHTML = `
      <div class="canvas-modebar">
        <button type="button" class="canvas-mode-btn canvas-mode-ocr active"><i class="bi bi-circle-fill" style="color:#22c55e"></i> OCR (gõ chữ tự do)</button>
        <button type="button" class="canvas-mode-btn canvas-mode-object"><i class="bi bi-circle-fill" style="color:#f5c518"></i> Object (chọn nhãn)</button>
      </div>
      <div class="canvas-stage-wrap">
        <div class="canvas-stage"><canvas class="canvas-grid"></canvas></div>
      </div>
      <div class="canvas-toolbar">
        <button type="button" class="btn subtle canvas-undo"><i class="bi bi-arrow-counterclockwise"></i> Hoàn tác</button>
        <button type="button" class="btn subtle canvas-clear"><i class="bi bi-trash3"></i> Xoá hết</button>
        <span class="field-note canvas-hint">Chọn chế độ ở trên rồi kéo chuột vẽ khung.</span>
      </div>`;

    let boxes = (opts.initialBoxes || []).map((b) => ({ ...b }));
    let mode = "ocr";
    let drawing = false, curBox = null;

    const stageWrap = mountEl.querySelector(".canvas-stage-wrap");
    const stage = mountEl.querySelector(".canvas-stage");
    const canvas = mountEl.querySelector(".canvas-grid");
    const ctx = canvas.getContext("2d");
    const modeOcrBtn = mountEl.querySelector(".canvas-mode-ocr");
    const modeObjectBtn = mountEl.querySelector(".canvas-mode-object");

    function resize() {
      canvas.width = W; canvas.height = H;
      canvas.style.width = W + "px"; canvas.style.height = H + "px";
      stage.style.width = W + "px";
      updateScale();
    }
    function updateScale() {
      const availW = stageWrap.getBoundingClientRect().width;
      const scale = availW > 0 ? availW / W : 1;
      stage.style.transform = "scale(" + scale + ")";
      stageWrap.style.height = Math.ceil(H * scale) + "px";
    }
    function updateModeButtons() {
      modeOcrBtn.classList.toggle("active", mode === "ocr");
      modeObjectBtn.classList.toggle("active", mode === "object");
    }
    function boxDisplayLabel(b) {
      if (b.kind === "ocr") return b.text || "";
      if (!b.label) return "";
      return b.minCount > 1 ? (b.label + " ×" + b.minCount) : b.label;
    }
    function draw() {
      if (opts.onChange) opts.onChange(boxes);
      ctx.clearRect(0, 0, W, H);
      ctx.strokeStyle = "rgba(255,255,255,0.06)";
      ctx.lineWidth = 1;
      for (let gx = 0; gx <= W; gx += 20) { ctx.beginPath(); ctx.moveTo(gx + 0.5, 0); ctx.lineTo(gx + 0.5, H); ctx.stroke(); }
      for (let gy = 0; gy <= H; gy += 20) { ctx.beginPath(); ctx.moveTo(0, gy + 0.5); ctx.lineTo(W, gy + 0.5); ctx.stroke(); }
      boxes.forEach((b, i) => drawBox(b, i));
      if (curBox) drawBox(curBox, -1);
      syncOverlays();
    }
    function drawBox(b, idx) {
      const x = Math.min(b.x0, b.x1), y = Math.min(b.y0, b.y1);
      const w = Math.abs(b.x1 - b.x0), h = Math.abs(b.y1 - b.y0);
      const color = COLOR[b.kind] || COLOR.ocr;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);
      const text = boxDisplayLabel(b);
      if (text) {
        ctx.font = "600 12px system-ui, sans-serif";
        // 2026-08-21 (theo yêu cầu người dùng: "dùng icon đẹp và hiện đại hơn" - đã đổi emoji ->
        // bootstrap-icons ở HẦU HẾT nơi khác) - CHỖ NÀY GIỮ NGUYÊN emoji: đây là chip vẽ trực
        // tiếp lên <canvas> 2D qua ctx.fillText(), KHÔNG PHẢI HTML - bootstrap-icons là web font
        // hiển thị qua CSS ::before trên thẻ <i>, không vẽ được vào canvas theo cách này (cần
        // load riêng font PUA codepoint của icon, phức tạp/rủi ro hơn nhiều so với lợi ích).
        const icon = b.kind === "ocr" ? "🔤" : "🔲";
        const label = "#" + (idx + 1) + " " + icon + " " + text;
        const padX = 5;
        const tw = ctx.measureText(label).width + padX * 2;
        const chipY = y - 18 >= 0 ? y - 18 : y;
        ctx.fillStyle = color;
        ctx.fillRect(x, chipY, tw, 17);
        ctx.fillStyle = "#111";
        ctx.fillText(label, x + padX, chipY + 12);
      }
    }
    function syncOverlays() {
      stage.querySelectorAll(".box-del").forEach((el) => el.remove());
      boxes.forEach((b, i) => {
        const x = Math.min(b.x0, b.x1), y = Math.min(b.y0, b.y1);
        const w = Math.abs(b.x1 - b.x0);
        const del = document.createElement("div");
        del.className = "box-del";
        del.textContent = "×";
        del.style.left = (x + w - 8) + "px";
        del.style.top = (y - 8) + "px";
        del.title = "Xoá khung #" + (i + 1);
        del.addEventListener("mousedown", (e) => { e.stopPropagation(); boxes.splice(i, 1); draw(); });
        stage.appendChild(del);
      });
    }

    function openOcrEditor(box, onDone) {
      const x = Math.min(box.x0, box.x1), y = Math.min(box.y0, box.y1);
      const w = Math.max(80, Math.abs(box.x1 - box.x0));
      const wrap = document.createElement("div");
      wrap.className = "box-editor";
      wrap.style.left = x + "px";
      wrap.style.top = (y - 24 >= 0 ? y - 24 : y) + "px";
      const input = document.createElement("input");
      input.className = "box-edit-input";
      input.type = "text";
      input.value = box.text || "";
      input.placeholder = "Chữ trên màn hình…";
      input.style.width = w + "px";
      wrap.appendChild(input);
      stage.appendChild(wrap);
      input.focus(); input.select();
      let done = false;
      function finish() {
        if (done) return;
        done = true;
        box.text = input.value.trim();
        wrap.remove();
        onDone();
      }
      input.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === "Escape") finish(); });
      input.addEventListener("blur", finish);
      wrap.addEventListener("mousedown", (e) => e.stopPropagation());
    }

    function openObjectEditor(box, onDone) {
      const x = Math.min(box.x0, box.x1), y = Math.min(box.y0, box.y1);
      const wrap = document.createElement("div");
      wrap.className = "box-editor";
      wrap.style.left = x + "px";
      wrap.style.top = (y - 26 >= 0 ? y - 26 : y) + "px";

      const searchWrap = document.createElement("div");
      searchWrap.className = "box-edit-search-wrap";
      const search = document.createElement("input");
      search.className = "box-edit-search";
      search.type = "text";
      search.placeholder = "Gõ để tìm nhãn…";
      search.value = box.label || "";
      searchWrap.appendChild(search);

      const dropdown = document.createElement("div");
      dropdown.className = "box-edit-dropdown";
      dropdown.style.display = "none";
      searchWrap.appendChild(dropdown);

      const count = document.createElement("input");
      count.className = "box-edit-count";
      count.type = "number"; count.min = "1"; count.value = box.minCount || 1;
      count.title = "Số lượng tối thiểu";

      wrap.appendChild(searchWrap);
      wrap.appendChild(count);
      stage.appendChild(wrap);
      search.focus(); search.select();

      let filtered = [];
      let highlightIdx = -1;
      function renderDropdown() {
        dropdown.innerHTML = "";
        if (filtered.length === 0) {
          const empty = document.createElement("div");
          empty.className = "box-edit-dropdown-empty";
          empty.textContent = "Không tìm thấy nhãn nào khớp";
          dropdown.appendChild(empty);
        } else {
          filtered.forEach((lb, i) => {
            const item = document.createElement("div");
            item.className = "box-edit-dropdown-item" + (i === highlightIdx ? " highlight" : "");
            item.textContent = lb;
            item.addEventListener("mousedown", (e) => { e.preventDefault(); selectLabel(lb); });
            dropdown.appendChild(item);
          });
        }
        dropdown.style.display = "block";
      }
      function updateFilter() {
        const q = stripAccents(search.value.trim());
        filtered = q
          ? LABELS.filter((lb) => stripAccents(lb).indexOf(q) !== -1).slice(0, 30)
          : LABELS.slice(0, 30);
        highlightIdx = filtered.length ? 0 : -1;
        renderDropdown();
      }
      function selectLabel(lb) {
        box.label = lb;
        search.value = lb;
        dropdown.style.display = "none";
      }
      search.addEventListener("input", updateFilter);
      search.addEventListener("focus", updateFilter);
      search.addEventListener("keydown", (e) => {
        if (e.key === "ArrowDown") {
          e.preventDefault();
          if (filtered.length) { highlightIdx = (highlightIdx + 1) % filtered.length; renderDropdown(); }
        } else if (e.key === "ArrowUp") {
          e.preventDefault();
          if (filtered.length) { highlightIdx = (highlightIdx - 1 + filtered.length) % filtered.length; renderDropdown(); }
        } else if (e.key === "Enter") {
          e.preventDefault();
          if (highlightIdx >= 0 && filtered[highlightIdx]) selectLabel(filtered[highlightIdx]);
        } else if (e.key === "Escape") {
          finish();
        }
      });
      count.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === "Escape") finish(); });

      let done = false;
      function finish() {
        if (done) return;
        done = true;
        if (LABELS.indexOf(box.label) === -1) box.label = "";
        box.minCount = Math.max(1, parseInt(count.value, 10) || 1);
        wrap.remove();
        onDone();
      }
      wrap.addEventListener("mousedown", (e) => e.stopPropagation());
      wrap.addEventListener("focusout", () => {
        setTimeout(() => { if (!wrap.contains(document.activeElement)) finish(); }, 0);
      });
    }

    function stageXY(evt) {
      const r = stage.getBoundingClientRect();
      const scaleX = r.width > 0 ? W / r.width : 1;
      const scaleY = r.height > 0 ? H / r.height : 1;
      return {
        x: Math.max(0, Math.min(W, (evt.clientX - r.left) * scaleX)),
        y: Math.max(0, Math.min(H, (evt.clientY - r.top) * scaleY)),
      };
    }

    stage.addEventListener("mousedown", (e) => {
      if (e.target !== canvas) return;
      const p = stageXY(e);
      drawing = true;
      curBox = { x0: p.x, y0: p.y, x1: p.x, y1: p.y, kind: mode, text: "", label: "", minCount: 1 };
    });
    window.addEventListener("mousemove", (e) => {
      if (!drawing || !curBox) return;
      const p = stageXY(e);
      curBox.x1 = p.x; curBox.y1 = p.y;
      draw();
    });
    window.addEventListener("mouseup", () => {
      if (!drawing || !curBox) return;
      drawing = false;
      const w = Math.abs(curBox.x1 - curBox.x0), h = Math.abs(curBox.y1 - curBox.y0);
      if (w < 8 || h < 8) { curBox = null; draw(); return; }
      const finished = curBox;
      curBox = null;
      boxes.push(finished);
      draw();
      const onDone = () => draw();
      if (finished.kind === "object") openObjectEditor(finished, onDone);
      else openOcrEditor(finished, onDone);
    });

    modeOcrBtn.addEventListener("click", () => { mode = "ocr"; updateModeButtons(); });
    modeObjectBtn.addEventListener("click", () => { mode = "object"; updateModeButtons(); });
    mountEl.querySelector(".canvas-undo").addEventListener("click", () => { boxes.pop(); draw(); });
    mountEl.querySelector(".canvas-clear").addEventListener("click", () => { boxes = []; draw(); });

    updateModeButtons();
    resize();
    draw();
    const ro = window.ResizeObserver ? new ResizeObserver(updateScale) : null;
    if (ro) ro.observe(stageWrap);
    else window.addEventListener("resize", updateScale);

    return {
      W, H,
      getBoxes() {
        return boxes.map((b) => ({
          x0: b.x0, y0: b.y0, x1: b.x1, y1: b.y1, kind: b.kind,
          text: b.text || "", label: b.label || "", minCount: b.minCount || 1,
        }));
      },
      hasBoxes() { return boxes.length > 0; },
      // 2026-08-21 (theo yêu cầu người dùng: "lịch sử tìm kiếm... khôi phục lại trạng thái đó")
      // - canvas TOÀN CỤC (KIS/Q&A) là 1 singleton tạo 1 LẦN lúc tải trang (khác canvas/mốc của
      // TRAKE/Temporal, vốn đã hỗ trợ khôi phục qua opts.initialBoxes vì bị tạo LẠI mỗi lần
      // renderAnchorCards) - cần cách bơm khung đã vẽ vào widget ĐANG SỐNG, không tạo mới được.
      setBoxes(newBoxes) {
        boxes = (newBoxes || []).map((b) => ({ ...b }));
        draw(); // draw() tự gọi opts.onChange(boxes) - badge "N khung" tự cập nhật theo
      },
      destroy() { if (ro) ro.disconnect(); },
    };
  }

  window.createCanvasWidget = createCanvasWidget;
})();
