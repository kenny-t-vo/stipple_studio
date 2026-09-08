'use strict';

// The browser shell: the same Python, in a Web Worker, over postMessage.
//
// Files are the only real difference from the desktop. An uploaded image is
// written into the worker's virtual filesystem under /work and referred to by
// path from then on, so params carry the same strings either way and the
// pipeline needs no browser-specific code. Output comes back as bytes and is
// handed to the browser as a download.

window.STIPPLE_SHELL = (() => {
  const WORK = '/work';
  const worker = new Worker('worker.js', {type: 'module'});

  const waiting = new Map();
  let nextId = 1;

  const BOOT = {
    runtime: 'loading python…',
    packages: 'loading numpy…',
    source: 'loading stipple…',
    ready: 'ready',
  };

  function setStatus(msg) {
    const s = document.getElementById('status');
    if (s) s.textContent = msg;
  }

  worker.onmessage = (ev) => {
    const m = ev.data;
    if (m.type === 'boot') { setStatus(BOOT[m.stage] || m.stage); return; }
    const w = waiting.get(m.id);
    if (!w) return;
    waiting.delete(m.id);
    m.ok ? w.resolve(m) : w.reject(new Error(m.error));
  };

  worker.onerror = (e) => setStatus('worker failed: ' + (e.message || e.type));

  function send(name, payload, opts = {}) {
    const id = nextId++;
    return new Promise((resolve, reject) => {
      waiting.set(id, {resolve, reject});
      worker.postMessage({id, name, payload, ...opts},
                         opts.bytes ? [opts.bytes] : []);
    });
  }

  // ── browser file plumbing ──────────────────────────────────────────

  function chooseFile(accept) {
    return new Promise((resolve) => {
      const i = document.createElement('input');
      i.type = 'file';
      i.accept = accept;
      i.hidden = true;
      const done = (f) => { i.remove(); resolve(f); };
      i.addEventListener('change', () => done(i.files[0] || null), {once: true});
      i.addEventListener('cancel', () => done(null), {once: true});
      document.body.appendChild(i);
      i.click();
    });
  }

  function download(name, buf) {
    const url = URL.createObjectURL(new Blob([buf]));
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    a.hidden = true;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  }

  // The virtual filesystem is flat and lives for one page load, so a name
  // collision only has to be avoided within the session.
  let uploads = 0;
  const workPath = (name) =>
    `${WORK}/${++uploads}-${name.replace(/[^\w.\-]+/g, '_')}`;

  // ── persistence ────────────────────────────────────────────────────
  //
  // Paths are dropped: they name files inside a filesystem that no longer
  // exists on the next page load.

  const LAST = 'stipple.last';

  function loadLast() {
    try {
      const raw = JSON.parse(localStorage.getItem(LAST) || '{}');
      delete raw.in_path;
      delete raw.out_path;
      return raw;
    } catch { return {}; }
  }

  function saveLast(params) {
    try {
      const keep = Object.assign({}, params);
      delete keep.in_path;
      delete keep.out_path;
      localStorage.setItem(LAST, JSON.stringify(keep));
    } catch { /* private window, quota, or blocked storage */ }
  }

  return {
    async init() {
      const {meta} = await send('init');
      meta.defaults.out_path = `${WORK}/output.svg`;
      meta.last = loadLast();
      return meta;
    },

    async preview(params, view, coarse, crop) {
      const {meta, buffers} = await send(
        'preview', {params, view, coarse, crop}, {blobs: 1});
      return {meta, buf: buffers[0]};
    },

    histogram: (path, paper) =>
      send('histogram', {path, paper}).then(r => r.meta.bins),

    async pickInput(paper) {
      const f = await chooseFile('image/*,.tif,.tiff');
      if (!f) return null;
      const path = workPath(f.name);
      const bytes = await f.arrayBuffer();
      await send('put', null, {path, bytes});
      const {meta} = await send('source', {path, paper});
      return {path, width: meta.width, height: meta.height};
    },

    // No save dialog without the File System Access API, and the result is a
    // download either way, so this only names the file.
    async pickOutput(current) {
      const now = (current || 'output.svg').split('/').pop();
      const name = window.prompt('Save the export as', now);
      if (!name) return null;
      return `${WORK}/${name.endsWith('.svg') || name.endsWith('.svgz')
                        ? name : name + '.svg'}`;
    },

    async render(params) {
      const {meta, buffers} = await send('render', {params});
      meta.files.forEach((n, i) => download(n, buffers[i]));
      saveLast(params);
      return meta;
    },

    async presetSave(params) {
      const name = window.prompt('Save the preset as', 'preset');
      if (!name) return null;
      const {meta, buffers} = await send(
        'preset_save', {params, name: name.replace(/\.json$/, '')});
      download(meta.files[0], buffers[0]);
      return meta;
    },

    async presetLoad() {
      const f = await chooseFile('.json,application/json');
      if (!f) return null;
      const {meta} = await send('preset_load', {
        text: await f.text(),
        name: f.name.replace(/\.json$/, ''),
      });
      return meta;
    },
  };
})();
