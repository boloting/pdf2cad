(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const MAX_BYTES = 40 * 1024 * 1024;

  const els = {
    drop: $("dropzone"), input: $("file-input"), chip: $("file-chip"), name: $("file-name"),
    size: $("file-size"), clear: $("file-clear"), sample: $("sample-btn"), convert: $("convert-btn"),
    pageField: $("page-field"), page: $("page-select"), scale: $("scale-input"),
    dims: $("opt-dims"), hatch: $("opt-hatch"), blocks: $("opt-blocks"),
    empty: $("state-empty"), loading: $("state-loading"), error: $("state-error"), result: $("state-result"),
    errorMsg: $("error-msg"), retry: $("retry-btn"),
    resultName: $("result-name"), badgeScale: $("badge-scale"), badgeLayers: $("badge-layers"), badgeSize: $("badge-size"),
    download: $("download-btn"), stats: $("stats"), warnings: $("warnings"), layers: $("layers"), entities: $("entities"),
    viewer: $("viewer"), img: $("viewer-img"), legend: $("legend"), zoomIn: $("zoom-in"), zoomOut: $("zoom-out"), zoomFit: $("zoom-fit"),
  };

  let file = null;
  let current = null;
  let view = "highlight";

  const STAT_ORDER = [
    ["线性标注", "var(--dim)"], ["半径/直径标注", "var(--dim)"], ["引线", "var(--dim)"],
    ["图块定义", "var(--block)"], ["图块引用", "var(--block)"],
    ["图案填充", "var(--hatch)"], ["实心填充", "var(--hatch)"],
    ["文字", "var(--text)"], ["圆", "var(--plain)"], ["圆弧", "var(--plain)"], ["样条曲线", "var(--plain)"],
  ];

  function show(state) {
    for (const k of ["empty", "loading", "error", "result"]) els[k].hidden = k !== state;
  }

  function fmtSize(n) {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / 1024 / 1024).toFixed(1)} MB`;
  }

  function fmtScale(s) {
    if (s >= 1) return `1:${+s.toPrecision(6)}`;
    return `${+(1 / s).toPrecision(6)}:1`;
  }

  function setFile(f) {
    if (!f) return;
    if (!/\.pdf$/i.test(f.name) && f.type !== "application/pdf") {
      fail("请选择 PDF 文件。");
      return;
    }
    if (f.size > MAX_BYTES) {
      fail("文件超过 40 MB。");
      return;
    }
    file = f;
    els.name.textContent = f.name;
    els.size.textContent = fmtSize(f.size);
    els.chip.hidden = false;
    els.drop.hidden = true;
    els.pageField.hidden = true;
    els.page.replaceChildren();
    els.convert.disabled = false;
    run(1);
  }

  function clearFile() {
    file = null;
    els.input.value = "";
    els.chip.hidden = true;
    els.drop.hidden = false;
    els.pageField.hidden = true;
    els.convert.disabled = true;
    current = null;
    show("empty");
  }

  function fail(msg) {
    els.errorMsg.textContent = msg;
    show("error");
  }

  async function run(page) {
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file, file.name);
    fd.append("page", String(page));
    const manual = document.querySelector('input[name="scale-mode"]:checked').value === "manual";
    if (manual) {
      const v = parseFloat(els.scale.value);
      if (!(v > 0)) {
        fail("手动比例需要填一个正数，例如 1:100 填 100。");
        return;
      }
      fd.append("scale", String(v));
    }
    fd.append("dimensions", String(els.dims.checked));
    fd.append("hatches", String(els.hatch.checked));
    fd.append("blocks", String(els.blocks.checked));

    els.convert.disabled = true;
    show("loading");
    try {
      const res = await fetch("/api/convert", { method: "POST", body: fd });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(body.detail || `服务器返回 ${res.status}`);
      current = body;
      render(body);
    } catch (e) {
      fail(e instanceof TypeError ? "无法连接服务器，请确认服务正在运行。" : e.message);
    } finally {
      els.convert.disabled = !file;
    }
  }

  function render(r) {
    const rep = r.report;
    els.resultName.textContent = r.filename;
    const src = { dimensions: "由标注推算", manual: "手动指定", default: "未识别，按 1:1" }[rep.scale_source];
    els.badgeScale.textContent = `比例 ${fmtScale(rep.scale)} · ${src}`;
    els.badgeScale.className = "badge" + (rep.scale_source === "dimensions" ? " ok" : "");
    els.badgeLayers.textContent = `${rep.layers.length} 个图层${rep.pdf_layers ? "（来自 PDF）" : ""}`;
    els.badgeSize.textContent = `图幅 ${rep.page_size_mm[0]} × ${rep.page_size_mm[1]} mm`;
    els.download.href = `/api/results/${encodeURIComponent(r.id)}/dxf`;
    els.download.setAttribute("download", r.filename);

    if (rep.page_count > 1 && els.page.options.length !== rep.page_count) {
      els.page.replaceChildren(...Array.from({ length: rep.page_count }, (_, i) => {
        const o = document.createElement("option");
        o.value = String(i + 1);
        o.textContent = `第 ${i + 1} 页`;
        return o;
      }));
    }
    els.pageField.hidden = rep.page_count <= 1;
    els.page.value = String(rep.page + 1);

    els.stats.replaceChildren(...STAT_ORDER.map(([key, color]) => {
      const n = rep.recognized[key] || 0;
      const div = document.createElement("div");
      div.className = "stat" + (n ? "" : " zero");
      const num = document.createElement("div");
      num.className = "stat-num";
      num.textContent = String(n);
      const label = document.createElement("div");
      label.className = "stat-label";
      const sw = document.createElement("i");
      sw.style.background = color;
      label.append(sw, key);
      div.append(num, label);
      return div;
    }));

    els.warnings.replaceChildren(...rep.warnings.map((w) => {
      const li = document.createElement("li");
      li.textContent = w;
      return li;
    }));

    els.layers.replaceChildren(...rep.layers.map((name) => {
      const s = document.createElement("span");
      s.className = "chip";
      s.textContent = name;
      return s;
    }));
    els.entities.replaceChildren(...Object.entries(rep.entities).map(([k, v]) => {
      const tr = document.createElement("tr");
      const a = document.createElement("td");
      a.textContent = k;
      const b = document.createElement("td");
      b.textContent = String(v);
      tr.append(a, b);
      return tr;
    }));

    show("result");
    setView(view, true);
  }

  // --- viewer -------------------------------------------------------------
  const vs = { scale: 1, x: 0, y: 0, w: 0, h: 0 };

  function applyTransform() {
    els.img.style.transform = `translate(${vs.x}px, ${vs.y}px) scale(${vs.scale})`;
    els.zoomFit.textContent = `${Math.round(vs.scale * 100)}%`;
  }

  function fit() {
    const box = els.viewer.getBoundingClientRect();
    if (!vs.w || !box.width) return;
    const pad = 24;
    vs.scale = Math.min((box.width - pad * 2) / vs.w, (box.height - pad * 2) / vs.h);
    vs.x = (box.width - vs.w * vs.scale) / 2;
    vs.y = (box.height - vs.h * vs.scale) / 2;
    applyTransform();
  }

  function zoomAt(factor, cx, cy) {
    const next = Math.min(40, Math.max(0.05, vs.scale * factor));
    const k = next / vs.scale;
    vs.x = cx - (cx - vs.x) * k;
    vs.y = cy - (cy - vs.y) * k;
    vs.scale = next;
    applyTransform();
  }

  function setView(name, refit) {
    view = name;
    for (const t of document.querySelectorAll(".tab")) {
      const on = t.dataset.view === name;
      t.classList.toggle("active", on);
      t.setAttribute("aria-selected", String(on));
    }
    els.legend.hidden = name !== "highlight";
    if (!current) return;
    const keep = !refit && vs.w ? { ...vs } : null;
    els.img.onload = () => {
      // Render the SVG at a generous base width so zooming stays sharp.
      const ratio = els.img.naturalHeight / els.img.naturalWidth || 0.7;
      vs.w = 1600;
      vs.h = 1600 * ratio;
      els.img.style.width = `${vs.w}px`;
      els.img.style.height = `${vs.h}px`;
      if (keep) {
        Object.assign(vs, { scale: keep.scale, x: keep.x, y: keep.y });
        applyTransform();
      } else {
        fit();
      }
    };
    els.img.src = `/api/results/${encodeURIComponent(current.id)}/${name}.svg`;
  }

  let drag = null;
  els.viewer.addEventListener("pointerdown", (e) => {
    drag = { x: e.clientX, y: e.clientY, ox: vs.x, oy: vs.y };
    els.viewer.setPointerCapture(e.pointerId);
    els.viewer.classList.add("dragging");
  });
  els.viewer.addEventListener("pointermove", (e) => {
    if (!drag) return;
    vs.x = drag.ox + e.clientX - drag.x;
    vs.y = drag.oy + e.clientY - drag.y;
    applyTransform();
  });
  const endDrag = () => { drag = null; els.viewer.classList.remove("dragging"); };
  els.viewer.addEventListener("pointerup", endDrag);
  els.viewer.addEventListener("pointercancel", endDrag);
  els.viewer.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = els.viewer.getBoundingClientRect();
    zoomAt(Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top);
  }, { passive: false });
  els.viewer.addEventListener("dblclick", fit);
  const center = () => { const r = els.viewer.getBoundingClientRect(); return [r.width / 2, r.height / 2]; };
  els.zoomIn.addEventListener("click", () => zoomAt(1.4, ...center()));
  els.zoomOut.addEventListener("click", () => zoomAt(1 / 1.4, ...center()));
  els.zoomFit.addEventListener("click", fit);
  window.addEventListener("resize", () => { if (current) fit(); });
  for (const t of document.querySelectorAll(".tab")) t.addEventListener("click", () => setView(t.dataset.view, false));

  // --- input --------------------------------------------------------------
  els.input.addEventListener("change", () => setFile(els.input.files[0]));
  els.drop.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); els.input.click(); }
  });
  for (const ev of ["dragenter", "dragover"]) {
    els.drop.addEventListener(ev, (e) => { e.preventDefault(); els.drop.classList.add("drag"); });
  }
  for (const ev of ["dragleave", "drop"]) {
    els.drop.addEventListener(ev, (e) => { e.preventDefault(); els.drop.classList.remove("drag"); });
  }
  els.drop.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
  document.addEventListener("dragover", (e) => e.preventDefault());
  document.addEventListener("drop", (e) => {
    e.preventDefault();
    if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
  });
  els.clear.addEventListener("click", clearFile);
  els.convert.addEventListener("click", () => run(parseInt(els.page.value || "1", 10)));
  els.retry.addEventListener("click", () => (file ? run(parseInt(els.page.value || "1", 10)) : show("empty")));
  els.page.addEventListener("change", () => run(parseInt(els.page.value, 10)));
  els.scale.addEventListener("focus", () => { document.querySelector('input[name="scale-mode"][value="manual"]').checked = true; });

  els.sample.addEventListener("click", async () => {
    try {
      const res = await fetch("/api/sample");
      if (!res.ok) throw new Error();
      const blob = await res.blob();
      setFile(new File([blob], "示例-安装板.pdf", { type: "application/pdf" }));
    } catch {
      fail("示例图纸加载失败，请刷新页面重试。");
    }
  });
})();
