'use strict';

// ── control specification ────────────────────────────────────────────
// Sections are titled lowercase with a period, per the house style.

const SPEC = [
  {t:'source.', rows:[
    {k:'in_path',  type:'file',     l:'input image'},
    {k:'out_path', type:'savefile', l:'output'},
  ]},
  {t:'canvas.', rows:[
    {k:'canvas_w_in', type:'range', l:'width',  min:1, max:96, step:.25, unit:'in'},
    {k:'canvas_h_in', type:'range', l:'height', min:1, max:96, step:.25, unit:'in', on:'!lock_aspect'},
    {k:'lock_aspect', type:'check', l:'lock aspect to source'},
  ]},
  {t:'tone.', rows:[
    {type:'hist'},
    {k:'black_point', type:'range', l:'black point', min:0, max:.95, step:.01},
    {k:'white_point', type:'range', l:'white point', min:.05, max:1, step:.01},
    {k:'contrast',    type:'range', l:'contrast',    min:-1, max:1, step:.02},
    {k:'gamma',       type:'range', l:'gamma',       min:.2, max:8, step:.05},
    {k:'threshold',   type:'range', l:'threshold',   min:0, max:.98, step:.01},
    {k:'pre_blur',    type:'range', l:'pre-blur',    min:0, max:12, step:.25, unit:'px'},
    {k:'invert',      type:'check', l:'invert'},
  ]},
  {t:'marks.', rows:[
    {k:'sampler', type:'cells', l:'sampler',
     opts:[['relaxed','relaxed'],['poisson','poisson'],['classic','classic']]},
    {k:'relax_iterations', type:'range', l:'regularity', min:0, max:60, step:1},
    {k:'max_density', type:'range', l:'max density', min:.01, max:1.2, step:.005},
    {k:'min_density', type:'range', l:'min density', min:0, max:.5, step:.005},
    {k:'dot_radius_pt', type:'range', l:'mark size', min:.02, max:3, step:.01, unit:'pt'},
    {k:'seed', type:'range', l:'seed', min:0, max:999, step:1},
    {k:'scale_with_darkness', type:'check', l:'scale marks with tone'},
    {k:'max_radius_scale', type:'range', l:'max scale', min:1, max:5, step:.05,
     on:'scale_with_darkness'},
  ]},
  {t:'lines.', rows:[
    {k:'line_mode', type:'check', l:'strokes instead of dots'},
    {k:'line_length_factor', type:'range', l:'length', min:.5, max:20, step:.25,
     unit:'x gap', on:'line_mode'},
    {k:'line_taper', type:'check', l:'taper ends', on:'line_mode'},
    {k:'flow_smoothing', type:'range', l:'flow smoothing', min:.5, max:24, step:.5,
     unit:'px', on:'line_mode'},
    {k:'flow_diffusion', type:'range', l:'flow reach', min:0, max:14, step:1,
     on:'line_mode'},
    {k:'flow_bias_angle', type:'range', l:'bias angle', min:0, max:180, step:1,
     unit:'°', on:'line_mode'},
    {k:'flow_bias_strength', type:'range', l:'bias strength', min:0, max:1, step:.02,
     on:'line_mode'},
    {k:'flow_jitter', type:'range', l:'jitter', min:0, max:1, step:.02, on:'line_mode'},
    {k:'flow_perpendicular', type:'check', l:'run across structure', on:'line_mode'},
  ]},
  {t:'colour.', rows:[
    {k:'color_mode', type:'cells', l:'ink',
     opts:[['mono','one colour'],['source','from image']]},
    {k:'ink',   type:'color', l:'ink',   on:'color_mode==mono'},
    {k:'paper', type:'color', l:'paper'},
  ]},
  {t:'output.', rows:[
    {k:'svg_structure', type:'cells', l:'svg',
     opts:[['compound','one path'],['circles','separate']]},
    {k:'paper_background', type:'check', l:'include paper rectangle'},
    {k:'png_dpi', type:'range', l:'png proof', min:0, max:600, step:25, unit:'dpi',
     zero:'off'},
  ]},
];

const NUM_KEYS = new Set();
SPEC.forEach(s => s.rows.forEach(r => {
  if (r.type === 'range') NUM_KEYS.add(r.k);
}));

// ── state ────────────────────────────────────────────────────────────

let P = null;            // current params
let DEFAULTS = null;
let meta = null;         // metadata of the last fit preview
let detailMeta = null;
let crop = [.42, .42, .58, .58];
let seq = 0;
let inflight = false;
let pending = null;

// Everything outside the page goes through the shell: a local HTTP server
// on the desktop, a Pyodide worker in the browser build.
const SHELL = window.STIPPLE_SHELL;

const $ = s => document.querySelector(s);
const el = (tag, cls) => { const n = document.createElement(tag); if (cls) n.className = cls; return n; };

function fmt(k, v) {
  if (typeof v !== 'number') return String(v);
  if (Number.isInteger(v)) return String(v);
  const a = Math.abs(v);
  return v.toFixed(a < 1 ? 2 : a < 10 ? 2 : 1);
}

// A row's `on` names the condition under which it is live; anything else is
// dimmed and inert. Forms: "key", "!key", "key==value". Replaces `dim`, which
// inverted the == form and so dimmed every line control in line mode.
function live(cond) {
  if (!cond) return true;
  const eq = cond.split('==');
  if (eq.length === 2) return P[eq[0]] === eq[1];
  if (cond.startsWith('!')) return !P[cond.slice(1)];
  return !!P[cond];
}

// ── build the control panel ──────────────────────────────────────────

function buildControls() {
  const host = $('#controls');
  host.textContent = '';

  SPEC.forEach(sec => {
    const s = el('section', 'sec');
    const h = el('h2', 'sec__h');
    h.textContent = sec.t;
    s.appendChild(h);
    const body = el('div', 'sec__body');
    s.appendChild(body);
    sec.rows.forEach(r => body.appendChild(makeRow(r)));
    host.appendChild(s);
  });
  syncControls();
}

function makeRow(r) {
  if (r.type === 'hist') {
    const c = el('canvas', 'hist');
    c.id = 'hist'; c.width = 600; c.height = 76;
    return c;
  }

  const row = el('div', 'row');
  row.dataset.k = r.k || '';
  if (r.on) row.dataset.on = r.on;

  if (r.type === 'check') {
    row.className = 'row row--full';
    const lab = el('label', 'check');
    const inp = el('input'); inp.type = 'checkbox'; inp.id = 'c_' + r.k;
    const box = el('span', 'box');
    const txt = el('span', 't'); txt.textContent = r.l;
    lab.append(inp, box, txt);
    inp.addEventListener('change', () => { P[r.k] = inp.checked; changed(false); });
    row.appendChild(lab);
    return row;
  }

  if (r.type === 'cells') {
    row.className = 'row row--full';
    const lab = el('div', 'row__l'); lab.textContent = r.l;
    const cells = el('div', 'cells');
    r.opts.forEach(([val, text]) => {
      const b = el('button'); b.type = 'button'; b.textContent = text;
      b.dataset.val = val; b.dataset.k = r.k;
      b.addEventListener('click', () => { P[r.k] = val; changed(false); });
      cells.appendChild(b);
    });
    row.append(lab, cells);
    return row;
  }

  if (r.type === 'color') {
    const lab = el('div', 'row__l'); lab.textContent = r.l;
    const inp = el('input'); inp.type = 'color'; inp.id = 'c_' + r.k;
    inp.addEventListener('input', () => { P[r.k] = inp.value; changed(true); });
    inp.addEventListener('change', () => { P[r.k] = inp.value; changed(false); });
    row.append(lab, inp);
    return row;
  }

  if (r.type === 'file' || r.type === 'savefile') {
    row.className = 'row row--full';
    const lab = el('div', 'row__l'); lab.textContent = r.l;
    const btn = el('button', 'btn'); btn.type = 'button'; btn.textContent = 'choose…';
    const head = el('div', 'row'); head.append(lab, btn);
    const val = el('div', 'path'); val.id = 'p_' + r.k;
    btn.addEventListener('click', () => pickFile(r.k));
    row.append(head, val);
    return row;
  }

  // range
  row.className = 'row';
  const lab = el('div', 'row__l'); lab.textContent = r.l;
  const val = el('div', 'row__v'); val.id = 'v_' + r.k;
  const inp = el('input'); inp.type = 'range'; inp.id = 'c_' + r.k;
  inp.min = r.min; inp.max = r.max; inp.step = r.step;
  inp.addEventListener('input', () => {
    P[r.k] = parseFloat(inp.value);
    val.textContent = readout(r);
    changed(true);                       // coarse while dragging
  });
  inp.addEventListener('change', () => changed(false));   // refine on release

  // A slider cannot hit gamma 2.60 reliably at three decimal places.
  // Click the number to type it.
  val.dataset.edit = '1';
  val.title = 'click to type a value';
  val.addEventListener('click', () => editValue(r, val, inp));
  row.append(lab, val, inp);
  return row;
}

function editValue(r, val, slider) {
  if (val.querySelector('input')) return;
  const box = el('input');
  box.type = 'text';
  box.value = String(P[r.k]);
  val.textContent = '';
  val.appendChild(box);
  box.focus();
  box.select();

  const commit = (apply) => {
    if (apply) {
      const v = parseFloat(box.value);
      // Clamp out-of-range input rather than rejecting it, so the control
      // cannot be driven outside what it can show.
      if (Number.isFinite(v)) {
        P[r.k] = Math.min(Math.max(v, r.min), r.max);
        slider.value = P[r.k];
      }
    }
    val.textContent = readout(r);
    if (apply) changed(false);
  };
  box.addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); commit(true); }
    if (e.key === 'Escape') { e.preventDefault(); commit(false); }
  });
  box.addEventListener('blur', () => commit(true));
}


function readout(r) {
  const v = P[r.k];
  if (r.zero && !v) return r.zero;
  return fmt(r.k, v) + (r.unit ? ' ' + r.unit : '');
}

function syncControls() {
  SPEC.forEach(sec => sec.rows.forEach(r => {
    if (!r.k) return;
    const c = document.getElementById('c_' + r.k);
    if (c) {
      if (r.type === 'check') c.checked = !!P[r.k];
      else if (c.value !== String(P[r.k])) c.value = P[r.k];
    }
    const v = document.getElementById('v_' + r.k);
    if (v && !v.querySelector('input')) v.textContent = readout(r);
    const p = document.getElementById('p_' + r.k);
    if (p) {
      const full = P[r.k] || '';
      p.textContent = full ? full.split('/').pop() : '—';
      p.title = full;                       // whole path on hover
    }
    document.querySelectorAll(`.cells button[data-k="${r.k}"]`).forEach(b => {
      b.setAttribute('aria-pressed', String(b.dataset.val === P[r.k]));
    });
  }));
  document.querySelectorAll('[data-on]').forEach(n => {
    const ok = live(n.dataset.on);
    n.style.opacity = ok ? '1' : '.34';
    n.style.pointerEvents = ok ? '' : 'none';
  });
}

// ── talking to the engine ────────────────────────────────────────────

function status(msg, kind) {
  const s = $('#status');
  s.textContent = msg;
  s.dataset.busy = kind === 'busy' ? 'true' : 'false';
  s.dataset.err = kind === 'err' ? 'true' : 'false';
}

function changed(coarse) {
  syncControls();
  if (!P.in_path) return;
  schedule(coarse);
}

let timer = null;
function schedule(coarse) {
  clearTimeout(timer);
  timer = setTimeout(() => run(coarse), coarse ? 20 : 60);
}

async function run(coarse) {
  if (inflight) { pending = coarse; return; }
  inflight = true;
  const mine = ++seq;
  status(coarse ? 'drawing…' : 'refining…', 'busy');
  try {
    const fit = await SHELL.preview(P, 'fit', coarse, null);
    if (mine === seq) { meta = fit.meta; drawFit(fit); }
    if (!coarse) {
      const det = await SHELL.preview(P, 'detail', false, cropRect());
      if (mine === seq) { detailMeta = det.meta; drawDetail(det); }
      await loadHistogram();
    }
    if (mine === seq) status(coarse ? 'draft' : 'ready');
  } catch (e) {
    status(String(e.message || e), 'err');
  } finally {
    inflight = false;
    if (pending !== null) { const c = pending; pending = null; run(c); }
  }
}

// ── drawing ──────────────────────────────────────────────────────────

function unpack(res) {
  const m = res.meta;
  const floats = m.count * m.stride;
  const marks = new Float32Array(res.buf, 0, floats);
  let colors = null;
  if (m.colorBytes) colors = new Uint8Array(res.buf, floats * 4, m.colorBytes);
  return {marks, colors};
}

function paint(canvas, res, pxPerPt, wPt, hPt) {
  const m = res.meta;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(wPt * pxPerPt * dpr));
  canvas.height = Math.max(1, Math.round(hPt * pxPerPt * dpr));
  canvas.style.width = (wPt * pxPerPt) + 'px';
  canvas.style.height = (hPt * pxPerPt) + 'px';

  const g = canvas.getContext('2d');
  g.setTransform(dpr * pxPerPt, 0, 0, dpr * pxPerPt, 0, 0);
  g.fillStyle = m.paper;
  g.fillRect(0, 0, wPt, hPt);

  const {marks, colors} = unpack(res);
  const col = i => colors
    ? `rgb(${colors[i*3]},${colors[i*3+1]},${colors[i*3+2]})`
    : m.ink;

  if (m.mode === 'strokes') {
    if (m.taper) {
      g.fillStyle = m.ink;
      for (let i = 0; i < m.count; i++) {
        const o = i * 6;
        const x0=marks[o],y0=marks[o+1],cx=marks[o+2],cy=marks[o+3],x2=marks[o+4],y2=marks[o+5];
        let vx = x2-x0, vy = y2-y0;
        const L = Math.hypot(vx,vy) || 1e-9;
        const nx = -vy/L*m.radius, ny = vx/L*m.radius;
        if (colors) g.fillStyle = col(i);
        g.beginPath();
        g.moveTo(x0,y0);
        g.quadraticCurveTo(cx+nx, cy+ny, x2, y2);
        g.quadraticCurveTo(cx-nx, cy-ny, x0, y0);
        g.fill();
      }
    } else {
      g.lineCap = 'round';
      g.lineWidth = 2 * m.radius;
      g.strokeStyle = m.ink;
      if (!colors) g.beginPath();
      for (let i = 0; i < m.count; i++) {
        const o = i * 6;
        if (colors) { g.beginPath(); g.strokeStyle = col(i); }
        g.moveTo(marks[o], marks[o+1]);
        g.quadraticCurveTo(marks[o+2], marks[o+3], marks[o+4], marks[o+5]);
        if (colors) g.stroke();
      }
      if (!colors) g.stroke();
    }
  } else {
    const scale = m.scaleWithDarkness;
    // A 0.25pt dot on a 28in canvas is under one device pixel in any
    // whole-canvas view: at true radius it vanishes, at a legible radius it
    // over-inks and reads darker than it prints. Draw at the floor and scale
    // opacity by the area actually covered, so total ink is preserved.
    const floor = 0.5 / (pxPerPt * dpr);
    let alpha = 1;
    if (m.radius < floor) alpha = Math.max(0.05, (m.radius * m.radius) / (floor * floor));
    g.globalAlpha = alpha;
    g.fillStyle = m.ink;
    for (let i = 0; i < m.count; i++) {
      const o = i * 3;
      let r = m.radius;
      if (scale) r *= 1 + (m.maxRadiusScale - 1) * marks[o+2];
      if (r < floor) r = floor;
      if (colors) g.fillStyle = col(i);
      g.beginPath();
      g.arc(marks[o], marks[o+1], r, 0, 6.283185307179586);
      g.fill();
    }
    g.globalAlpha = 1;
  }
}

function drawFit(res) {
  const m = res.meta;
  const box = $('#view').getBoundingClientRect();
  const pad = 44;
  const pxPerPt = Math.min((box.width - pad) / m.canvasW, (box.height - pad) / m.canvasH);
  paint($('#cv'), res, pxPerPt, m.canvasW, m.canvasH);
  placeMarquee(pxPerPt);
  showStats(m);
}

function drawDetail(res) {
  const m = res.meta;
  const host = $('.view--detail').getBoundingClientRect();
  const w = Math.max(80, host.width - 44);
  // 1pt = 1/72in, 1 CSS px = 1/96in, so actual size is 96/72 px per point.
  paint($('#cvd'), res, 96 / 72, m.canvasW, m.canvasH);
  $('#cvd').style.width = Math.min(w, m.canvasW * 96 / 72) + 'px';
  $('#cvd').style.height = 'auto';

  const dl = $('#detail-meta');
  dl.textContent = '';
  const rows = [
    ['marks here', m.count.toLocaleString()],
    ['window', (m.canvasW / 72).toFixed(2) + ' × ' + (m.canvasH / 72).toFixed(2) + ' in'],
    ['mark size', (m.radius * 2).toFixed(3) + ' pt'],
  ];
  rows.forEach(([k, v]) => {
    const dt = el('dt'); dt.textContent = k;
    const dd = el('dd'); dd.textContent = v;
    dl.append(dt, dd);
  });
}

function showStats(m) {
  const host = $('#stats');
  host.textContent = '';
  const full = m.fullTarget;
  [['marks', full.toLocaleString()],
   ['canvas', m.fullW.toFixed(2) + ' × ' + m.fullH.toFixed(2) + ' in'],
   ['preview', m.count.toLocaleString() + ' @ ' + (m.scale * 100).toFixed(0) + '%'],
   ['time', (m.elapsed * 1000).toFixed(0) + ' ms']
  ].forEach(([k, v]) => {
    const s = el('div', 'stat');
    const a = el('span', 'lbl'); a.textContent = k;
    const b = el('b'); b.textContent = v;
    s.append(a, b); host.appendChild(s);
  });
}

// ── detail window placement ──────────────────────────────────────────

function cropRect() {
  return crop;
}

function detailSpanFraction() {
  // How much of the canvas fits in the detail pane at actual size.
  const host = $('.view--detail').getBoundingClientRect();
  const wPx = Math.max(80, host.width - 44);
  const wPt = wPx * 72 / 96;
  const frac = meta ? Math.min(1, wPt / (meta.canvasW / meta.scale)) : .12;
  return frac;
}

function placeMarquee() {
  if (!meta) return;
  const cv = $('#cv');
  const box = cv.getBoundingClientRect();
  const host = $('#view').getBoundingClientRect();
  const mq = $('#marquee');
  mq.hidden = false;
  mq.style.left  = (box.left - host.left + crop[0] * box.width) + 'px';
  mq.style.top   = (box.top - host.top + crop[1] * box.height) + 'px';
  mq.style.width = Math.max(2, (crop[2] - crop[0]) * box.width) + 'px';
  mq.style.height= Math.max(2, (crop[3] - crop[1]) * box.height) + 'px';
}

function moveDetail(ev) {
  if (!meta) return;
  const b = $('#cv').getBoundingClientRect();
  const fx = (ev.clientX - b.left) / b.width;
  const fy = (ev.clientY - b.top) / b.height;
  // The detail window is square on the canvas, so its height fraction has to
  // account for the canvas aspect or it comes out stretched.
  const w = detailSpanFraction();
  const h = Math.min(1, w * (meta.canvasW / meta.canvasH));
  const x0 = Math.min(Math.max(fx - w / 2, 0), Math.max(0, 1 - w));
  const y0 = Math.min(Math.max(fy - h / 2, 0), Math.max(0, 1 - h));
  crop = [x0, y0, x0 + w, y0 + h];
  placeMarquee();
  schedule(false);
}

// ── histogram ────────────────────────────────────────────────────────

async function loadHistogram() {
  if (!P.in_path) return;
  let bins;
  try {
    bins = await SHELL.histogram(P.in_path, P.paper);
  } catch { return; }
  const c = $('#hist'); if (!c) return;
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth || 300, h = 38;
  c.width = w * dpr; c.height = h * dpr;
  const g = c.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const peak = Math.max(...bins) || 1;
  g.fillStyle = '#dcdcdc';
  const bw = w / bins.length;
  bins.forEach((v, i) => {
    const bh = Math.round(h * Math.sqrt(v / peak));
    g.fillRect(i * bw, h - bh, Math.max(1, bw - .5), bh);
  });
  // black and white point markers
  g.fillStyle = '#0000ee';
  g.fillRect(P.black_point * w, 0, 1, h);
  g.fillRect(P.white_point * w - 1, 0, 1, h);
}

// ── file pickers, export, presets ────────────────────────────────────

async function pickFile(key) {
  try {
    if (key === 'in_path') {
      const s = await SHELL.pickInput(P.paper);
      if (!s) return;
      P.in_path = s.path;
      if (P.lock_aspect) P.canvas_h_in = P.canvas_w_in * s.height / s.width;
      if (!P.out_path || P.out_path === DEFAULTS.out_path) {
        P.out_path = s.path.replace(/\.[^.\/]+$/, '') + '.stipple.svg';
      }
    } else {
      const path = await SHELL.pickOutput(P.out_path);
      if (!path) return;
      P.out_path = path;
    }
    changed(false);
  } catch (e) { status(String(e.message || e), 'err'); }
}

async function doExport() {
  const b = $('#export');
  b.disabled = true;
  status('exporting…', 'busy');
  try {
    const r = await SHELL.render(P);
    status(`wrote ${r.name} — ${r.marks.toLocaleString()} marks, ${r.mb} MB, ${r.secs}s`);
  } catch (e) {
    status(String(e.message || e), 'err');
  } finally { b.disabled = false; }
}

async function preset(save) {
  try {
    const j = save ? await SHELL.presetSave(P) : await SHELL.presetLoad();
    if (!j || !j.name) return;
    if (!save) { Object.assign(P, j.params); changed(false); }
    status((save ? 'saved ' : 'loaded ') + j.name);
  } catch (e) { status(String(e.message || e), 'err'); }
}

// ── boot ─────────────────────────────────────────────────────────────

(async function init() {
  const info = await SHELL.init();
  DEFAULTS = info.defaults;
  P = Object.assign({}, info.defaults, info.last || {});
  $('#ver').textContent = info.version;

  buildControls();
  $('#export').addEventListener('click', doExport);
  $('#preset-save').addEventListener('click', () => preset(true));
  $('#preset-load').addEventListener('click', () => preset(false));
  $('#reset').addEventListener('click', () => {
    const keep = {in_path: P.in_path, out_path: P.out_path};
    P = Object.assign({}, DEFAULTS, keep);
    changed(false);
  });
  $('#cv').addEventListener('click', moveDetail);

  const shell = document.querySelector('.shell');
  const drawer = (open) => shell.dataset.drawer = open ? 'open' : 'shut';
  $('#drawer-open').addEventListener('click', () => drawer(true));
  $('#drawer-close').addEventListener('click', () => drawer(false));
  window.addEventListener('resize', () => { if (meta) schedule(false); });

  if (P.in_path) { await loadHistogram(); run(false); }
  else status('choose an input image');
})();
