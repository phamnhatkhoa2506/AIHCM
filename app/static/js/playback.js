"use strict";
// 🎬 Playback — modal xem/chỉnh frame trước khi nộp, port từ v3 (online/app.py::_playback_dialog).
// Giữ đúng các quy tắc v3:
//   - Nudge (-5/+5)/gõ số/kéo slider CHỈ đổi frame XEM THỬ (pending) — frame THẬT SỰ dùng khi
//     nộp (confirmed) CHỈ đổi khi bấm "✅ Xác nhận frame này".
//   - "↩ Về frame gốc" quay về đúng frame hệ thống đề xuất LÚC MỞ dialog LẦN ĐẦU (không phải
//     giá trị đã chỉnh trước đó).
//   - Nhiều mốc (TRAKE/Temporal): timeline chung, hàng nút "#i" chọn mốc đang chỉnh, mỗi mốc có
//     pending RIÊNG (đổi mốc không mất preview đang xem thử của mốc kia).
//   - Q&A: có ô câu trả lời + nút nộp ngay trong dialog. KIS: nút nộp trong dialog. Temporal: nút
//     nộp trong dialog (dùng median). TRAKE: KHÔNG có nút nộp trong dialog — nộp bằng nút "📤 Nộp"
//     bên ngoài (đã tự đọc đúng frame vừa xác nhận, vì onConfirm mutate thẳng vào `row`).
//   - 2026-08-21 (theo yêu cầu người dùng: "thêm history thay đổi cho frame, có thể quay lại
//     những frame trước khi thay đổi") - MỖI mốc có lịch sử RIÊNG các frame đã CHỐT (không phải
//     mọi lần nudge/kéo thử, chỉ những lần thật sự bấm "✅ Xác nhận") - bấm 1 mốc cũ trong lịch
//     sử để quay lại NGAY, không xoá các mốc SAU nó (không phải undo-stack, chỉ là nhật ký).
(function () {
  const PLAYBACK_NUDGE_FRAMES = 5;

  const modal = document.getElementById("playback-modal");
  const closeBtn = document.getElementById("pb-close");
  const timelineEl = document.getElementById("pb-timeline");
  const pickerEl = document.getElementById("pb-anchor-picker");
  const prevBtn = document.getElementById("pb-prev");
  const nextBtn = document.getElementById("pb-next");
  const typedInput = document.getElementById("pb-typed");
  const gotoBtn = document.getElementById("pb-goto");
  const revertBtn = document.getElementById("pb-revert");
  const sliderInput = document.getElementById("pb-slider");
  const previewImg = document.getElementById("pb-preview");
  const previewCaption = document.getElementById("pb-preview-caption");
  const confirmStatus = document.getElementById("pb-confirm-status");
  const confirmBtn = document.getElementById("pb-confirm");
  const historyEl = document.getElementById("pb-history");
  const videoEl = document.getElementById("pb-video");
  const submitArea = document.getElementById("pb-submit-area");

  let st = null; // trạng thái phiên dialog hiện tại
  // 2026-08-21 (theo yêu cầu người dùng: "2 cái này đồng bộ 2 chiều, video có thể tua theo thao
  // tác của người dùng") - `programmaticSeek` phân biệt 2 nguồn đổi currentTime của <video>: TỰ
  // mình set (do người dùng nudge/kéo slider/gõ số bên khung Frame xem thử) vs người dùng tự kéo
  // thanh tua/phát video - tránh vòng lặp phản hồi (mình seek -> bắt sự kiện seeked -> lại seek).
  let programmaticSeek = false;
  // 2026-08-21 (bug thật: "mới bấm vào playback là tự nhiên về frame 0 luôn, thanh video cũng
  // không kéo/tua được") - gán `videoEl.src` MỚI luôn tự reset currentTime về 0 và bắn hàng loạt
  // sự kiện (kể cả "timeupdate"/"seeked" với currentTime=0 tạm thời) TRƯỚC KHI video mới thật sự
  // sẵn sàng - onVideoTimeChange() đọc trúng currentTime=0 tạm bợ đó, HIỂU NHẦM là người dùng vừa
  // tua về 0 -> ghi đè luôn st.pending về 0 thật (đúng y hệt bug đã thấy), rồi vòng lặp
  // pending=0 <-> video liên tục bị syncVideoTime() kéo về 0 khiến thanh tua như bị khoá cứng,
  // không kéo nổi. Chặn bằng cờ `videoReady`: mọi sự kiện video CHỈ được xử lý SAU KHI video MỚI
  // đã báo "loadedmetadata" xong (tức đã load đúng nguồn hiện tại, không phải dư âm từ nguồn cũ).
  let videoReady = false;
  // 2026-08-21 (bug thật VẪN còn sau lần sửa trước: "vẫn về frame 0, vẫn không kéo được thanh
  // video") - gốc sâu hơn: `st.fps` lúc MỚI mở dialog là giá trị ƯỚC LƯỢNG mặc định 25 (chưa kịp
  // gọi xong /api/video_meta) - lần syncVideoTime() ĐẦU TIÊN (gọi ngay trong renderAll() lúc mới
  // mở) tính `t = pending/25` SAI, có thể VƯỢT QUÁ thời lượng video thật -> trình duyệt coi seek
  // này KHÔNG hợp lệ, ÂM THẦM bỏ qua (currentTime vẫn kẹt ở 0, KHÔNG bắn "seeked" để cờ
  // programmaticSeek được tiêu đúng chỗ) - "timeupdate" định kỳ sau đó bắt trúng currentTime=0
  // "trôi nổi" này, hiểu nhầm là người dùng vừa tua về 0 -> ghi đè pending về 0 thật (đúng y hệt
  // bug). Chặn triệt để: KHÔNG seek() lần nào cho tới khi biết CHẮC fps thật (metaLoaded).
  let metaLoaded = false;

  function closePlayback() {
    modal.classList.add("hidden");
    videoEl.pause();
    videoEl.src = "";
    videoReady = false;
    st = null;
  }
  closeBtn.addEventListener("click", closePlayback);
  modal.addEventListener("click", (e) => { if (e.target === modal) closePlayback(); });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !modal.classList.contains("hidden")) closePlayback();
  });

  function renderTimeline() {
    // 2026-08-21 (theo yêu cầu người dùng: "phần dấu chấm frame trên cùng mình thấy chưa khớp
    // cho lắm") - với KIS/QA CHỈ 1 mốc, chuẩn hoá t/tMax của chính nó luôn ra 100% -> chấm bị đẩy
    // dồn về mép phải, trông như "lệch"/vô nghĩa (thanh timeline chỉ có ý nghĩa khi SO SÁNH vị trí
    // NHIỀU mốc với nhau - TRAKE/Temporal). Ẩn hẳn thanh timeline khi chỉ có 1 mốc.
    if (st.confirmed.length <= 1) { timelineEl.innerHTML = ""; return; }
    const times = st.confirmed.map((f) => f / st.fps);
    const tMax = Math.max(...times, 1.0);
    timelineEl.innerHTML = times.map((t, i) => {
      const pct = (100 * t / tMax).toFixed(1);
      const activeCls = i === st.active ? " pb-dot-active" : "";
      return `<div class="pb-dot${activeCls}" style="left:${pct}%"><span class="pb-dot-mark"></span><span class="pb-dot-label">${st.labels[i]}</span></div>`;
    }).join("");
  }

  function renderPicker() {
    if (st.confirmed.length <= 1) { pickerEl.innerHTML = ""; return; }
    pickerEl.innerHTML = st.confirmed.map((_, i) =>
      `<button type="button" class="btn subtle pb-pick-btn${i === st.active ? " active" : ""}" data-idx="${i}">${i === st.active ? "● " : ""}${st.labels[i]}</button>`
    ).join("");
    pickerEl.querySelectorAll(".pb-pick-btn").forEach((btn) => {
      btn.addEventListener("click", () => { st.active = Number(btn.dataset.idx); renderAll(); });
    });
  }

  async function renderPreview() {
    const pending = st.pending[st.active];
    const t = pending / st.fps;
    previewImg.src = `/api/frame?video_id=${encodeURIComponent(st.videoId)}&t=${t}`;
    previewCaption.textContent = `Frame ${pending} · t≈${t.toFixed(1)}s`;
  }

  // Chiều Frame xem thử -> Video: mỗi lần pending đổi (nudge/gõ số/kéo slider/về gốc/nhảy lịch
  // sử/đổi mốc đang chọn), tua luôn <video> tới đúng giây đó.
  function syncVideoTime() {
    if (!videoEl.src || !st || !metaLoaded) return;
    const t = st.pending[st.active] / st.fps;
    if (!Number.isFinite(t)) return;
    programmaticSeek = true;
    try { videoEl.currentTime = t; } catch { programmaticSeek = false; }
  }

  // Chiều Video -> Frame xem thử: người dùng tự kéo thanh tua (hoặc video đang phát) -> cập nhật
  // lại pending + ảnh xem thử theo ĐÚNG giây hiện tại của video (KHÔNG gọi lại syncVideoTime() ở
  // đây - tránh vòng lặp). "seeked" (kéo/tua xong) phản hồi ngay; "timeupdate" (đang phát) chặn
  // bớt tần suất (throttle) để không gọi /api/frame liên tục mỗi vài chục ms.
  let lastVideoSyncTs = 0;
  function onVideoTimeChange(immediate) {
    if (!videoReady) return; // dư âm sự kiện từ lúc video CŨ đang bị thay src, bỏ qua hẳn
    if (programmaticSeek) { programmaticSeek = false; return; }
    if (!st) return;
    // 2026-08-21 - chốt chặn cuối: CHỈ nhận tín hiệu Video -> Frame khi đó là thao tác THẬT của
    // người dùng: hoặc vừa tua xong ("seeked", immediate=true), hoặc video ĐANG PHÁT thật sự.
    // Video đứng yên (paused) mà bắn "timeupdate" thì đó là nhiễu (vd seek bị trình duyệt từ
    // chối, currentTime kẹt 0) - KHÔNG được phép ghi đè frame người dùng đang chọn.
    if (!immediate && videoEl.paused) return;
    const now = Date.now();
    if (!immediate && now - lastVideoSyncTs < 400) return;
    lastVideoSyncTs = now;
    const newPending = Math.max(0, Math.min(st.maxFrame, Math.round(videoEl.currentTime * st.fps)));
    if (newPending === st.pending[st.active]) return;
    st.pending[st.active] = newPending;
    renderControls();
    renderPreview();
    renderConfirmRow();
  }
  videoEl.addEventListener("seeked", () => onVideoTimeChange(true));
  videoEl.addEventListener("timeupdate", () => onVideoTimeChange(false));
  videoEl.addEventListener("loadedmetadata", () => { videoReady = true; syncVideoTime(); });

  function renderConfirmRow() {
    const pending = st.pending[st.active];
    const confirmed = st.confirmed[st.active];
    if (pending !== confirmed) {
      confirmStatus.innerHTML = `⏳ Đang xem thử frame <code>${pending}</code> — frame ĐÃ CHỐT (dùng khi nộp) vẫn là <code>${confirmed}</code>.`;
      confirmBtn.classList.remove("hidden");
    } else {
      confirmStatus.innerHTML = `✅ Frame đã chốt: <code>${confirmed}</code>`;
      confirmBtn.classList.add("hidden");
    }
    const original = st.original[st.active];
    revertBtn.disabled = pending === original;
    revertBtn.textContent = `↩ Về frame gốc (${original})`;
  }

  // Lịch sử các frame ĐÃ CHỐT của mốc đang chọn (không phải mọi lần nudge/kéo, chỉ những lần
  // thật sự bấm "✅ Xác nhận") - bấm 1 mốc cũ QUAY LẠI NGAY (tự confirm luôn, không cần bấm xác
  // nhận lần nữa), giữ nguyên các mốc SAU nó trong danh sách (không xoá "tương lai" như undo-stack).
  function renderHistory() {
    const hist = st.history[st.active];
    const confirmed = st.confirmed[st.active];
    historyEl.innerHTML = hist.map((f, idx) => {
      const isCurrent = f === confirmed && idx === hist.length - 1;
      return `<button type="button" class="pb-history-chip${isCurrent ? " current" : ""}" data-frame="${f}">${idx === 0 ? "gốc " : ""}${f}</button>`;
    }).join("");
    historyEl.querySelectorAll(".pb-history-chip:not(.current)").forEach((chip) => {
      chip.addEventListener("click", () => jumpToHistory(Number(chip.dataset.frame)));
    });
  }

  function jumpToHistory(frameId) {
    st.pending[st.active] = frameId;
    st.confirmed[st.active] = frameId;
    pushHistory(st.active, frameId);
    st.onConfirm(st.active, frameId, st.fps);
    renderTimeline();
    renderPicker();
    renderControls();
    renderPreview();
    renderConfirmRow();
    renderHistory();
    renderSubmitArea();
    syncVideoTime();
  }

  function pushHistory(idx, frameId) {
    const hist = st.history[idx];
    if (hist[hist.length - 1] !== frameId) hist.push(frameId);
  }

  function renderControls() {
    sliderInput.min = 0;
    sliderInput.max = st.maxFrame;
    sliderInput.value = st.pending[st.active];
    typedInput.min = 0;
    typedInput.max = st.maxFrame;
    typedInput.value = st.pending[st.active];
  }

  function renderSubmitArea() {
    submitArea.innerHTML = "";
    if (st.mode === "trake") {
      submitArea.innerHTML = `<p class="field-note">Loại truy vấn hiện tại: <b>TRAKE</b> — nộp đủ cả chuỗi bằng nút "📤 Nộp" BÊN NGOÀI Playback (đã tự đọc đúng frame vừa xác nhận ở đây).</p>`;
      return;
    }
    if (st.mode === "temporal") {
      const median = medianOf(st.confirmed);
      submitArea.innerHTML = `
        <p class="field-note">Loại truy vấn hiện tại: <b>Temporal</b> — frame giữa = <code>${median}</code></p>
        <button type="button" id="pb-submit-btn" class="btn primary">📤 Nộp (Temporal)</button>`;
      document.getElementById("pb-submit-btn").addEventListener("click", (e) => st.onSubmit(e.currentTarget, null));
      return;
    }
    if (st.mode === "qa") {
      submitArea.innerHTML = `
        <div class="qa-field">
          <label class="qa-label">Câu trả lời</label>
          <input type="text" id="pb-answer" value="${(st.answer || "").replace(/"/g, "&quot;")}" />
        </div>
        <button type="button" id="pb-submit-btn" class="btn primary">📤 Nộp</button>`;
      document.getElementById("pb-submit-btn").addEventListener("click", (e) =>
        st.onSubmit(e.currentTarget, document.getElementById("pb-answer").value.trim()));
      return;
    }
    // kis
    submitArea.innerHTML = `<button type="button" id="pb-submit-btn" class="btn primary">📤 Nộp</button>`;
    document.getElementById("pb-submit-btn").addEventListener("click", (e) => st.onSubmit(e.currentTarget, null));
  }

  function medianOf(arr) {
    const sorted = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    const med = sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    return Math.trunc(med);
  }

  function renderAll() {
    renderTimeline();
    renderPicker();
    renderControls();
    renderPreview();
    renderConfirmRow();
    renderHistory();
    renderSubmitArea();
    syncVideoTime(); // đổi mốc đang chọn (TRAKE/Temporal) -> video cũng tua theo mốc mới
  }

  function setPending(v) {
    st.pending[st.active] = Math.max(0, Math.min(st.maxFrame, Math.round(v)));
    renderControls();
    renderPreview();
    renderConfirmRow();
    syncVideoTime();
  }

  prevBtn.addEventListener("click", () => setPending(st.pending[st.active] - PLAYBACK_NUDGE_FRAMES));
  nextBtn.addEventListener("click", () => setPending(st.pending[st.active] + PLAYBACK_NUDGE_FRAMES));
  gotoBtn.addEventListener("click", () => setPending(Number(typedInput.value)));
  revertBtn.addEventListener("click", () => setPending(st.original[st.active]));
  sliderInput.addEventListener("input", () => setPending(Number(sliderInput.value)));

  confirmBtn.addEventListener("click", () => {
    st.confirmed[st.active] = st.pending[st.active];
    pushHistory(st.active, st.confirmed[st.active]);
    st.onConfirm(st.active, st.confirmed[st.active], st.fps);
    renderTimeline();
    renderPicker();
    renderConfirmRow();
    renderHistory();
    renderSubmitArea();
  });

  /**
   * videoId, frameIds (list, thứ tự = anchor), labels (list nhãn hiện trên timeline/nút chọn),
   * mode ("kis"|"qa"|"trake"|"temporal"), answer (chỉ qa), onConfirm(idx, newFrameId) gọi mỗi
   * lần XÁC NHẬN 1 mốc (để cập nhật row/card ngoài dialog), onSubmit(btn, answerText) gọi khi
   * bấm nút nộp trong dialog (KIS/Q&A/Temporal - TRAKE không có nút nộp trong dialog).
   */
  window.openPlayback = async function (opts) {
    // 2026-08-21 (theo yêu cầu người dùng: "giữ history cho toàn bộ quá trình đổi, từ gốc -> lần
    // đổi 1 -> lần đổi 2 -> ...") - nếu caller (app.js) truyền `history` đã lưu từ LẦN MỞ TRƯỚC
    // (gắn trên `row._pbHistory`, sống suốt vòng đời card) thì dùng lại NGUYÊN mảng đó (cùng
    // reference) - pushHistory() mutate thẳng vào mảng này nên tự động "nhớ" xuyên suốt nhiều lần
    // mở/đóng dialog, không phải mỗi lần mở lại reset về [frame hiện tại]. "Frame gốc" (original,
    // dùng cho nút "↩ Về frame gốc") LUÔN LÀ phần tử ĐẦU TIÊN của lịch sử (lần mở đầu tiên, KHÔNG
    // phải giá trị đã chỉnh trước đó) - đúng ý đã ghi ở đầu file.
    const history = opts.history || opts.frameIds.map((f) => [f]);
    st = {
      videoId: opts.videoId,
      labels: opts.labels,
      mode: opts.mode,
      answer: opts.answer || "",
      original: history.map((h) => h[0]),
      confirmed: [...opts.frameIds],
      pending: [...opts.frameIds],
      history,
      active: 0,
      fps: 25,
      maxFrame: Math.max(...opts.frameIds, 1000),
      onConfirm: opts.onConfirm || (() => {}),
      onSubmit: opts.onSubmit || (() => {}),
    };
    videoReady = false; // chặn xử lý sự kiện video cho tới khi nguồn MỚI báo loadedmetadata xong
    metaLoaded = false; // chặn MỌI lần seek() cho tới khi biết chắc fps thật (xem ghi chú trên)
    modal.classList.remove("hidden");
    videoEl.src = `/api/video?video_id=${encodeURIComponent(st.videoId)}`;
    renderAll();

    try {
      const res = await fetch(`/api/video_meta?video_id=${encodeURIComponent(st.videoId)}`);
      const meta = await res.json();
      st.fps = meta.fps || 25;
      st.maxFrame = meta.max_frame || st.maxFrame;
    } catch { /* giữ giá trị ước lượng ban đầu nếu lỗi mạng */ }
    metaLoaded = true;
    renderAll(); // giờ fps đã chắc chắn đúng (hoặc chắc chắn là fallback cuối cùng) - seek an toàn
  };

  window.closePlayback = closePlayback;
})();
