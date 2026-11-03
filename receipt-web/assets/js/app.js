(() => {
  'use strict';

  /* ───────── Helpers ───────── */
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const LS = {
    get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : v; } catch { return d; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* bỏ qua */ } },
  };

  const MAX_BYTES = 10 * 1024 * 1024;
  const MAX_SIDE = 4096;
  const ACCEPT = ['image/jpeg', 'image/png'];
  const TIMEOUT_MS = 240000;
  const SAMPLE_SRC = 'assets/sample/original.jpg';

  const LABELS = { merchant: 'Cửa hàng', date: 'Ngày', time: 'Giờ', total: 'Tổng thanh toán', address: 'Địa chỉ', phone: 'Điện thoại' };
  const CORE = ['merchant', 'date', 'total', 'address'];

  const cfg = {
    base: LS.get('kie.base', 'http://127.0.0.1:8000'),
    path: LS.get('kie.path', '/analyze'),
  };

  const S = {
    file: null, sample: false, name: '', size: 0,
    srcUrl: null, dims: null,
    result: null,
    view: 'original', zoom: 1, overlay: 'interactive', tab: 'fields',
    ctrl: null, timer: null, t0: 0, cancelled: false, timedOut: false,
    focus: null, renderToken: 0,
  };

  const el = {
    intro: $('#intro'), workspace: $('#workspace'), stage: $('#stage'), canvas: $('#canvas'),
    scan: $('#scan'), results: $('#results'), tip: $('#tip'), toast: $('#toast'),
    fileInput: $('#fileInput'), dropzone: $('#dropzone'), status: $('#status'), statusText: $('#statusText'),
    detectBar: $('#detectBar'), zoomTools: $('#zoomTools'), zoomVal: $('#zoomVal'),
  };

  /* ───────── Toast ───────── */
  let toastT;
  function toast(msg) {
    el.toast.textContent = msg;
    el.toast.classList.add('show');
    clearTimeout(toastT);
    toastT = setTimeout(() => el.toast.classList.remove('show'), 2200);
  }

  async function copyText(text, okMsg = 'Đã sao chép') {
    try { await navigator.clipboard.writeText(text); }
    catch {
      const ta = document.createElement('textarea');
      ta.value = text; ta.style.cssText = 'position:fixed;opacity:0';
      document.body.appendChild(ta); ta.select();
      try { document.execCommand('copy'); } catch { /* bỏ qua */ }
      ta.remove();
    }
    toast(okMsg);
  }

  /* ───────── Format ───────── */
  const nf = new Intl.NumberFormat('vi-VN');
  const fmtMs = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(2).replace('.', ',')} s` : `${Math.round(ms)} ms`);
  const fmtBytes = (b) => (b >= 1048576 ? `${(b / 1048576).toFixed(1).replace('.', ',')} MB` : `${Math.max(1, Math.round(b / 1024))} KB`);
  const prettify = (k) => k.replace(/[_-]+/g, ' ').replace(/^\w/, (c) => c.toUpperCase());

  function displayValue(f) {
    if (f.value == null) return null;
    if (typeof f.value === 'number') return f.key === 'total' ? `${nf.format(f.value)} ₫` : nf.format(f.value);
    const s = String(f.value);
    if (f.key === 'total' && /^[\d.,\s]+$/.test(s.trim())) return `${s.trim()} ₫`;
    return s;
  }

  /* ───────── Chuẩn hóa dữ liệu từ API ───────── */
  function toPoly(p) {
    if (!Array.isArray(p)) return null;
    if (p.length >= 3 && p.every((q) => Array.isArray(q) && q.length >= 2)) return p.map(([x, y]) => [+x, +y]);
    if (p.length === 4 && p.every((n) => typeof n === 'number')) { const [x0, y0, x1, y1] = p; return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]; }
    if (p.length === 8 && p.every((n) => typeof n === 'number')) return [[p[0], p[1]], [p[2], p[3]], [p[4], p[5]], [p[6], p[7]]];
    return null;
  }
  const polyRect = (poly) => {
    const xs = poly.map((q) => q[0]), ys = poly.map((q) => q[1]);
    return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
  };
  const normConf = (c) => (typeof c === 'number' && isFinite(c) ? (c > 1 ? c / 100 : c) : null);

  function normBoxes(raw) {
    if (!Array.isArray(raw)) return [];
    const out = [];
    for (const b of raw) {
      let poly, text = '', conf = null, kv = false;
      if (Array.isArray(b)) { // dạng [poly, [text, conf]]
        poly = toPoly(b[0]);
        if (Array.isArray(b[1])) { text = String(b[1][0] ?? ''); conf = normConf(b[1][1]); }
      } else if (b && typeof b === 'object') {
        poly = toPoly(b.polygon ?? b.points ?? b.bbox ?? b.box ?? b.coordinates);
        text = String(b.text ?? b.transcription ?? '');
        conf = normConf(b.confidence ?? b.score);
        const lab = typeof b.label === 'string' ? b.label.toLowerCase() : '';
        kv = !!(b.is_key_value ?? b.key_value ?? b.is_kv ?? b.highlight ?? b.important) ||
             (lab && !['text', 'other', 'none', 'o'].includes(lab));
      }
      if (poly) out.push({ poly, rect: polyRect(poly), text, conf, kv });
    }
    return out;
  }

  function normFields(raw) {
    const entries = Array.isArray(raw)
      ? raw.map((o) => [o?.name ?? o?.key ?? o?.label ?? '', o])
      : Object.entries(raw || {});
    const map = new Map();
    for (const [k, v] of entries) {
      if (!k) continue;
      let value = v, conf = null, bbox = null;
      if (v && typeof v === 'object' && !Array.isArray(v)) {
        value = v.value ?? v.text ?? null;
        conf = normConf(v.confidence ?? v.score);
        const poly = toPoly(v.bbox ?? v.box ?? v.polygon);
        bbox = poly ? polyRect(poly) : null;
      }
      if (value === '' || value === undefined) value = null;
      map.set(String(k), { key: String(k), value, conf, bbox });
    }
    const list = CORE.map((k) => map.get(k) || { key: k, value: null, conf: null, bbox: null });
    for (const [k, f] of map) if (!CORE.includes(k)) list.push(f);
    return list;
  }

  function imgSrc(images, name) {
    if (!images) return null;
    const b = images[`${name}_jpeg_base64`] ?? images[`${name}_base64`];
    if (b) return b.startsWith('data:') ? b : `data:image/jpeg;base64,${b}`;
    return images[`${name}_url`] ?? null;
  }

  function normResult(raw) {
    const t = raw.timings_ms || {};
    const parts = [
      { k: 'Tiền xử lý', v: +t.preprocessing_ms || 0, c: 'c1' },
      { k: 'Đọc chữ', v: +t.ocr_ms || 0, c: 'c2' },
      { k: 'Trích xuất', v: +t.kie_ms || 0, c: 'c3' },
    ];
    const total = +(raw.processing_time_ms ?? t.total_ms) || parts.reduce((a, p) => a + p.v, 0);
    return {
      raw,
      fields: normFields(raw.fields),
      boxes: normBoxes(raw.boxes),
      text: raw.full_text || '',
      processed: imgSrc(raw.images, 'preprocessed'),
      annotated: imgSrc(raw.images, 'annotated'),
      parts, total,
    };
  }

  /* ───────── Tải kích thước ảnh ───────── */
  const dimCache = new Map();
  function loadDims(src) {
    if (dimCache.has(src)) return Promise.resolve(dimCache.get(src));
    return new Promise((res, rej) => {
      const im = new Image();
      im.onload = () => { const d = { w: im.naturalWidth, h: im.naturalHeight }; dimCache.set(src, d); res(d); };
      im.onerror = () => rej(new Error('Không đọc được ảnh'));
      im.src = src;
    });
  }

  /* ───────── Chọn ảnh ───────── */
  function setMode(mode) {
    el.intro.hidden = mode !== 'intro';
    el.workspace.hidden = mode !== 'work';
    window.scrollTo({ top: 0 });
  }

  function resetResult() {
    S.result = null; S.focus = null; S.tab = 'fields';
    setResState('idle');
    setView('original', true);
  }

  async function handleFile(file) {
    if (!file) return;
    if (!ACCEPT.includes(file.type)) return toast('Chỉ hỗ trợ ảnh JPEG hoặc PNG');
    if (file.size > MAX_BYTES) return toast(`Ảnh ${fmtBytes(file.size)} vượt giới hạn 10 MB`);
    const url = URL.createObjectURL(file);
    let d;
    try { d = await loadDims(url); } catch { URL.revokeObjectURL(url); return toast('Không đọc được ảnh này'); }
    if (Math.max(d.w, d.h) > MAX_SIDE) { URL.revokeObjectURL(url); return toast(`Cạnh ảnh ${Math.max(d.w, d.h)} px vượt giới hạn ${MAX_SIDE} px`); }
    if (S.srcUrl && !S.sample) URL.revokeObjectURL(S.srcUrl);
    Object.assign(S, { file, sample: false, name: file.name, size: file.size, srcUrl: url, dims: d });
    enterWorkspace();
  }

  async function useSample() {
    let d;
    try { d = await loadDims(SAMPLE_SRC); } catch { return toast('Không tìm thấy ảnh mẫu'); }
    if (S.srcUrl && !S.sample) URL.revokeObjectURL(S.srcUrl);
    Object.assign(S, { file: null, sample: true, name: 'hoa-don-mau.jpg', size: 0, srcUrl: SAMPLE_SRC, dims: d });
    enterWorkspace();
  }

  function enterWorkspace() {
    $('#fileName').textContent = S.name;
    $('#fileMeta').textContent = `${S.dims.w}×${S.dims.h}${S.size ? ' · ' + fmtBytes(S.size) : ''}`;
    setMode('work');
    resetResult();
  }

  /* ───────── Trạng thái bảng kết quả ───────── */
  function setResState(st) {
    el.results.dataset.state = st;
    const done = st === 'done' && S.result;
    $$('#viewTabs button').forEach((b) => {
      if (b.dataset.view === 'original') return;
      b.disabled = !done || (b.dataset.view === 'processed' && !S.result.processed) || (b.dataset.view === 'detect' && !S.result.annotated && !S.result.boxes.length);
    });
    el.scan.hidden = st !== 'loading';
  }

  /* ───────── Gọi API ───────── */
  const apiUrl = (p) => `${cfg.base.replace(/\/+$/, '')}${p.startsWith('/') ? p : '/' + p}`;

  async function httpError(res) {
    let detail = '';
    try {
      const j = await res.json();
      const d = j.detail ?? j.message ?? j.error;
      detail = Array.isArray(d) ? d.map((x) => x.msg || x).join('; ') : (typeof d === 'string' ? d : '');
    } catch { /* không phải JSON */ }
    const map = { 413: 'Ảnh quá lớn so với giới hạn của máy chủ.', 415: 'Máy chủ không hỗ trợ định dạng ảnh này.', 422: 'Máy chủ không chấp nhận ảnh này.', 503: 'Máy chủ chưa sẵn sàng, hãy thử lại sau ít giây.' };
    const e = new Error(detail || map[res.status] || `Máy chủ trả về lỗi ${res.status}.`);
    e.title = 'Máy chủ từ chối yêu cầu';
    return e;
  }

  async function analyze() {
    if (S.sample) {
      setResState('loading');
      await new Promise((r) => setTimeout(r, 650));
      return showResult(window.__KIE_SAMPLE__, true);
    }
    if (!S.file) return;

    const ctrl = new AbortController();
    Object.assign(S, { ctrl, cancelled: false, timedOut: false, t0: performance.now() });
    const to = setTimeout(() => { S.timedOut = true; ctrl.abort(); }, TIMEOUT_MS);
    setResState('loading');
    $('#elapsed').textContent = '0,0 s';
    clearInterval(S.timer);
    S.timer = setInterval(() => { $('#elapsed').textContent = `${((performance.now() - S.t0) / 1000).toFixed(1).replace('.', ',')} s`; }, 100);

    try {
      const fd = new FormData();
      fd.append('image', S.file, S.file.name);
      const res = await fetch(apiUrl(cfg.path), { method: 'POST', body: fd, signal: ctrl.signal });
      if (!res.ok) throw await httpError(res);
      const json = await res.json();
      showResult(json, false);
    } catch (e) {
      if (S.cancelled) { setResState('idle'); return; }
      if (S.timedOut) return showError('Quá thời gian chờ', 'Máy chủ không phản hồi sau 4 phút. Hãy thử lại với ảnh nhỏ hơn.');
      if (e.title) return showError(e.title, e.message);
      if (e instanceof SyntaxError) return showError('Kết quả không hợp lệ', 'Máy chủ trả về dữ liệu không đúng định dạng JSON.');
      showError('Không kết nối được máy chủ', `Không gọi được ${apiUrl(cfg.path)}. Hãy kiểm tra máy chủ đã chạy và địa chỉ trong phần cài đặt.`);
      setStatus('off');
    } finally {
      clearTimeout(to); clearInterval(S.timer);
    }
  }

  function showError(title, msg) {
    $('#errTitle').textContent = title;
    $('#errMsg').textContent = msg;
    setResState('error');
  }

  async function showResult(raw, isSample) {
    S.result = normResult(raw);
    S.result.sample = isSample;
    S.focus = null;
    // lấy kích thước ảnh đã chỉnh để vẽ lớp phủ đúng tỉ lệ
    const base = S.result.processed || S.result.annotated;
    if (base) { try { S.result.pdims = await loadDims(base); } catch { /* bỏ qua */ } }
    S.result.overlayOk = !!(S.result.pdims && S.result.boxes.length && S.result.boxes.every((b) =>
      b.rect[2] <= S.result.pdims.w * 1.03 && b.rect[3] <= S.result.pdims.h * 1.03 && b.rect[0] >= -2 && b.rect[1] >= -2));
    if (!S.result.overlayOk) S.overlay = 'server';

    $('#resTime').textContent = `Hoàn tất trong ${fmtMs(S.result.total)}`;
    $('#sampleBadge').hidden = !isSample;
    renderFields(); renderJson(); renderText(); renderTiming();
    setTab('fields');
    setResState('done');
    setView(S.result.annotated || S.result.boxes.length ? 'detect' : 'processed', true);
  }

  /* ───────── Render kết quả ───────── */
  function renderFields() {
    const rows = S.result.fields.map((f, i) => {
      const label = LABELS[f.key] ?? prettify(f.key);
      const v = displayValue(f);
      const has = v !== null;
      const pct = f.conf == null ? null : Math.round(f.conf * 100);
      const cls = f.conf == null ? '' : f.conf >= 0.9 ? '' : f.conf >= 0.75 ? 'mid' : 'low';
      return `<li class="fld${f.key === 'total' ? ' main' : ''}${has ? '' : ' empty'}" data-i="${i}">
        <span class="fld-label">${esc(label)}</span>
        <span class="fld-value">${has ? esc(v) : 'Không tìm thấy'}</span>
        ${has && pct !== null ? `<span class="conf" title="Độ tin cậy"><span class="conf-bar"><i class="${cls}" style="width:${pct}%"></i></span>${pct}%</span>` : ''}
        ${has ? `<span class="fld-tools">
          <button class="btn btn-icon btn-sm" data-act="locate" type="button" aria-label="Xem vị trí trên ảnh" title="Xem vị trí trên ảnh"><svg width="16" height="16"><use href="#i-locate"/></svg></button>
          <button class="btn btn-icon btn-sm" data-act="copy" type="button" aria-label="Sao chép" title="Sao chép"><svg width="16" height="16"><use href="#i-copy"/></svg></button>
        </span>` : ''}
      </li>`;
    }).join('');
    $('#paneFields').innerHTML = `<ul class="fields">${rows}</ul>`;
  }

  function cleanJson() {
    const r = { ...S.result.raw };
    delete r.images;
    return r;
  }
  function renderJson() {
    const txt = JSON.stringify(cleanJson(), null, 2)
      .replace(/\[\s*(-?[\d.]+(?:,\s*-?[\d.]+)*)\s*\]/g, (m, a) => `[${a.replace(/,\s+/g, ', ')}]`);
    $('#jsonOut').innerHTML = esc(txt).replace(
      /(&quot;(?:\\.|[^&]|&(?!quot;))*?&quot;)(\s*:)?|\b(true|false|null)\b|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g,
      (m, s, colon, b, n) => (s ? (colon ? `<span class="j-k">${s}</span>${colon}` : `<span class="j-s">${s}</span>`) : b ? `<span class="j-b">${b}</span>` : `<span class="j-n">${n}</span>`));
  }
  function renderText() {
    $('#textOut').textContent = S.result.text || 'Không có văn bản nào được nhận diện.';
  }
  function renderTiming() {
    const { parts, total } = S.result;
    const sum = parts.reduce((a, p) => a + p.v, 0) || 1;
    $('#timing').innerHTML = `
      <div class="timing-top"><span>Thời gian xử lý</span><b>${fmtMs(total)}</b></div>
      <div class="tbar" role="img" aria-label="Phân bổ thời gian theo từng bước">${parts.map((p) => `<i class="${p.c}" style="flex:${Math.max(p.v, sum * 0.01)}" title="${p.k}: ${fmtMs(p.v)}"></i>`).join('')}</div>
      <ul class="tl">${parts.map((p) => `<li><span><i class="${p.c}"></i>${p.k}</span><b>${fmtMs(p.v)}</b></li>`).join('')}</ul>`;
  }

  function setTab(t) {
    S.tab = t;
    $$('#resTabs button').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.tab === t)));
    $('#paneFields').hidden = t !== 'fields';
    $('#paneJson').hidden = t !== 'json';
    $('#paneText').hidden = t !== 'text';
  }

  /* ───────── Khung xem ảnh ───────── */
  function setView(v, force) {
    if (!force && S.view === v) return;
    S.view = v;
    S.zoom = 1;
    $$('#viewTabs button').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.view === v)));
    el.detectBar.hidden = v !== 'detect' || !S.result?.overlayOk;
    $$('#detectBar [data-overlay]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.overlay === S.overlay)));
    el.zoomTools.classList.toggle('off', v === 'compare');
    renderView();
  }

  function fitWidth(d) {
    const cs = getComputedStyle(el.stage);
    const pw = parseFloat(cs.paddingLeft) + parseFloat(cs.paddingRight);
    const ph = parseFloat(cs.paddingTop) + parseFloat(cs.paddingBottom);
    const sw = el.stage.clientWidth - pw, sh = el.stage.clientHeight - ph;
    return Math.max(120, Math.min(sw, sh * d.w / d.h));
  }

  function applyZoom() {
    el.zoomVal.textContent = S.zoom === 1 ? 'Vừa' : `${Math.round(S.zoom * 100)}%`;
    const w = $('.imgwrap', el.canvas);
    if (w) w.style.width = `${fitWidth({ w: +w.dataset.w, h: +w.dataset.h }) * S.zoom}px`;
  }

  async function renderView() {
    const token = ++S.renderToken;
    const v = S.view, r = S.result;
    el.canvas.className = 'canvas';
    el.canvas.innerHTML = '';
    el.stage.scrollTo(0, 0);

    const single = async (src, overlaySvg) => {
      const d = await loadDims(src);
      if (token !== S.renderToken) return;
      el.canvas.innerHTML = `<div class="imgwrap" data-w="${d.w}" data-h="${d.h}"><img src="${esc(src)}" alt="" draggable="false">${overlaySvg ? overlaySvg(d) : ''}</div>`;
      applyZoom();
      if (overlaySvg) paintFocus();
    };

    if (v === 'original') return single(S.srcUrl);
    if (v === 'processed') return single(r.processed || S.srcUrl);
    if (v === 'detect') {
      if (r.overlayOk && S.overlay === 'interactive') {
        return single(r.processed || r.annotated, (d) => `<svg viewBox="0 0 ${d.w} ${d.h}" preserveAspectRatio="none" aria-label="Các vùng chữ đã nhận diện">${
          r.boxes.map((b, i) => `<polygon class="bx ${b.kv ? 'kv' : 'tx'}" data-i="${i}" points="${b.poly.map((p) => p.join(',')).join(' ')}"/>`).join('')}<g id="fxLayer"></g></svg>`);
      }
      return single(r.annotated || r.processed || S.srcUrl);
    }
    if (v === 'compare') {
      el.canvas.classList.add('fill');
      el.canvas.innerHTML = `<div class="compare">
        <figure><figcaption>Ảnh gốc</figcaption><div class="frame"><img src="${esc(S.srcUrl)}" alt="Ảnh gốc"></div></figure>
        <figure><figcaption>Đã căn chỉnh và làm rõ</figcaption><div class="frame"><img src="${esc(r.processed || S.srcUrl)}" alt="Ảnh đã chỉnh"></div></figure></div>`;
    }
  }

  /* ───────── Làm nổi vùng của một trường ───────── */
  const squash = (s) => String(s).toLowerCase().replace(/\s+/g, '');
  function rectsFor(f) {
    if (f.bbox) return [f.bbox];
    const val = squash(f.value ?? '');
    if (!val) return [];
    const hits = S.result.boxes.filter((b) => {
      const t = squash(b.text);
      return t.length > 2 && (t === val || val.includes(t));
    });
    return hits.map((b) => b.rect);
  }

  function paintFocus() {
    const g = $('#fxLayer');
    if (!g) return;
    g.innerHTML = (S.focus || []).map(([x0, y0, x1, y1]) => `<rect class="fx" x="${x0 - 3}" y="${y0 - 3}" width="${x1 - x0 + 6}" height="${y1 - y0 + 6}" rx="3"/>`).join('');
  }

  async function locate(i) {
    const f = S.result.fields[i];
    S.focus = rectsFor(f);
    if (!S.focus.length) return toast('Chưa xác định được vị trí trên ảnh');
    if (S.view !== 'detect' || S.overlay !== 'interactive') {
      if (S.result.overlayOk) S.overlay = 'interactive';
      setView('detect', true);
      await new Promise((r) => setTimeout(r, 60));
    }
    paintFocus();
    const fx = $('.fx');
    if (fx) fx.scrollIntoView({ block: 'center', inline: 'center', behavior: 'smooth' });
  }

  /* ───────── Gợi ý khi rê chuột lên khung chữ ───────── */
  function showTip(e, html) {
    el.tip.innerHTML = html; el.tip.hidden = false;
    const pad = 14, w = el.tip.offsetWidth, h = el.tip.offsetHeight;
    el.tip.style.left = `${Math.min(e.clientX + pad, innerWidth - w - 8)}px`;
    el.tip.style.top = `${e.clientY + pad + h > innerHeight ? e.clientY - h - pad : e.clientY + pad}px`;
  }

  /* ───────── Kết nối máy chủ ───────── */
  const STATUS = { ok: 'Máy chủ sẵn sàng', warm: 'Máy chủ đang khởi động', off: 'Chưa kết nối máy chủ', checking: 'Đang kiểm tra…' };
  function setStatus(s) { el.status.dataset.s = s; el.statusText.textContent = STATUS[s]; }

  async function ping(base) {
    const c = new AbortController();
    const t = setTimeout(() => c.abort(), 4000);
    try {
      const r = await fetch(`${base.replace(/\/+$/, '')}/health`, { signal: c.signal });
      if (!r.ok) return 'off';
      let j = null; try { j = await r.json(); } catch { /* không có thân JSON */ }
      return j && (j.ready === false || j.status === 'loading' || j.model_loaded === false) ? 'warm' : 'ok';
    } catch { return 'off'; } finally { clearTimeout(t); }
  }
  async function checkHealth() { setStatus('checking'); setStatus(await ping(cfg.base)); }

  const dlg = $('#settings');
  function openSettings() {
    $('#apiBase').value = cfg.base; $('#apiPath').value = cfg.path;
    $('#testRes').textContent = ''; $('#testRes').className = 'test-res';
    dlg.showModal();
  }

  /* ───────── Sự kiện ───────── */
  el.dropzone.addEventListener('click', () => el.fileInput.click());
  el.fileInput.addEventListener('change', () => { handleFile(el.fileInput.files[0]); el.fileInput.value = ''; });
  $('#changeImage').addEventListener('click', () => el.fileInput.click());
  $('#useSample').addEventListener('click', useSample);
  $('#brand').addEventListener('click', (e) => { e.preventDefault(); if (S.ctrl) { S.cancelled = true; S.ctrl.abort(); } setMode('intro'); });

  $('#analyze').addEventListener('click', analyze);
  $('#retry').addEventListener('click', analyze);
  $('#reanalyze').addEventListener('click', analyze);
  $('#cancel').addEventListener('click', () => { S.cancelled = true; S.ctrl?.abort(); setResState('idle'); });
  $('#errSettings').addEventListener('click', openSettings);

  $('#viewTabs').addEventListener('click', (e) => { const b = e.target.closest('button[data-view]'); if (b && !b.disabled) setView(b.dataset.view); });
  $('#resTabs').addEventListener('click', (e) => { const b = e.target.closest('button[data-tab]'); if (b) setTab(b.dataset.tab); });
  $('#detectBar').addEventListener('click', (e) => { const b = e.target.closest('[data-overlay]'); if (b) { S.overlay = b.dataset.overlay; setView('detect', true); } });

  $('#zoomIn').addEventListener('click', () => { S.zoom = Math.min(4, +(S.zoom + 0.25).toFixed(2)); applyZoom(); });
  $('#zoomOut').addEventListener('click', () => { S.zoom = Math.max(0.5, +(S.zoom - 0.25).toFixed(2)); applyZoom(); });
  el.zoomVal.addEventListener('click', () => { S.zoom = 1; applyZoom(); });
  new ResizeObserver(() => { if (S.view !== 'compare') applyZoom(); }).observe(el.stage);

  $('#paneFields').addEventListener('click', (e) => {
    const b = e.target.closest('[data-act]'); if (!b) return;
    const i = +b.closest('.fld').dataset.i, f = S.result.fields[i];
    if (b.dataset.act === 'copy') copyText(String(f.value));
    else locate(i);
  });
  $('#paneFields').addEventListener('mouseover', (e) => {
    const li = e.target.closest('.fld'); if (!li || S.view !== 'detect' || S.overlay !== 'interactive') return;
    const f = S.result.fields[+li.dataset.i];
    if (f.value == null) return;
    S.focus = rectsFor(f); paintFocus();
  });
  $('#paneFields').addEventListener('mouseleave', () => { if (S.focus) { S.focus = null; paintFocus(); } });

  el.stage.addEventListener('mousemove', (e) => {
    const p = e.target.closest?.('.bx');
    if (!p) { el.tip.hidden = true; return; }
    const b = S.result.boxes[+p.dataset.i];
    showTip(e, `${esc(b.text || '—')}${b.conf != null ? `<small>${Math.round(b.conf * 100)}%</small>` : ''}`);
  });
  el.stage.addEventListener('mouseleave', () => { el.tip.hidden = true; });

  $('#copyJson').addEventListener('click', () => copyText(JSON.stringify(cleanJson(), null, 2), 'Đã sao chép JSON'));
  $('#dlJson').addEventListener('click', () => {
    const blob = new Blob([JSON.stringify(cleanJson(), null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${S.name.replace(/\.[^.]+$/, '') || 'ket-qua'}.json`;
    a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });

  // Dán ảnh
  addEventListener('paste', (e) => {
    const it = [...(e.clipboardData?.items || [])].find((x) => x.type.startsWith('image/'));
    if (it) { e.preventDefault(); handleFile(it.getAsFile()); }
  });

  // Kéo thả trên toàn trang
  let depth = 0;
  const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes('Files');
  addEventListener('dragenter', (e) => { if (hasFiles(e)) { depth++; document.body.classList.add('dragging'); } });
  addEventListener('dragleave', (e) => { if (hasFiles(e) && --depth <= 0) { depth = 0; document.body.classList.remove('dragging'); } });
  addEventListener('dragover', (e) => { if (hasFiles(e)) e.preventDefault(); });
  addEventListener('drop', (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault(); depth = 0; document.body.classList.remove('dragging');
    handleFile(e.dataTransfer.files[0]);
  });

  // Cài đặt
  $('#openSettings').addEventListener('click', openSettings);
  el.status.addEventListener('click', openSettings);
  $('#settingsCancel').addEventListener('click', () => dlg.close());
  $('#testConn').addEventListener('click', async () => {
    const out = $('#testRes');
    out.className = 'test-res'; out.textContent = 'Đang kiểm tra…';
    const r = await ping($('#apiBase').value.trim());
    out.className = `test-res ${r === 'off' ? 'bad' : 'ok'}`;
    out.textContent = r === 'ok' ? 'Kết nối được' : r === 'warm' ? 'Máy chủ đang khởi động' : 'Không kết nối được';
  });
  $('#settingsForm').addEventListener('submit', (e) => {
    if (e.submitter?.value !== 'save') return;
    cfg.base = $('#apiBase').value.trim();
    cfg.path = $('#apiPath').value.trim() || '/analyze';
    LS.set('kie.base', cfg.base); LS.set('kie.path', cfg.path);
    checkHealth();
  });

  checkHealth();
})();
