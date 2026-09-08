// Pyodide host. Runs the same Python the desktop build runs, off the main
// thread so a slider drag does not compete with sampling for the UI.

import {loadPyodide} from 'https://cdn.jsdelivr.net/pyodide/v314.0.6/full/pyodide.mjs';

const INDEX = 'https://cdn.jsdelivr.net/pyodide/v314.0.6/full/';

// Where uploaded sources and rendered output live in the virtual filesystem.
// Paths, so params carry the same strings they carry on the desktop.
const WORK = '/work';

let py = null;
let bridge = null;
let booting = null;

function say(stage) {
  self.postMessage({type: 'boot', stage});
}

async function boot() {
  say('runtime');
  py = await loadPyodide({indexURL: INDEX});

  say('packages');
  // numpy and pillow only: scipy is 13MB against numpy's 2.8MB, and
  // stipple.filters / stipple.spatial cover what the pipeline used it for.
  await py.loadPackage(['numpy', 'pillow']);

  say('source');
  const zip = await fetch(new URL('stipple-src.zip', self.location)).then(r => {
    if (!r.ok) throw new Error(`stipple-src.zip: ${r.status}`);
    return r.arrayBuffer();
  });
  py.unpackArchive(zip, 'zip', {extractDir: '/lib/stipple'});

  py.runPython(`
import sys, os
sys.path.insert(0, '/lib/stipple')
os.makedirs('${WORK}', exist_ok=True)
`);
  bridge = py.pyimport('bridge');
  say('ready');
}

function ready() {
  if (!booting) booting = boot();
  return booting;
}

self.onmessage = async (ev) => {
  const {id, name, payload, bytes, path, blobs} = ev.data;
  try {
    await ready();

    // Writing an uploaded file into the filesystem is the one call that is
    // not a bridge call: everything downstream then works on a path.
    if (name === 'put') {
      py.FS.writeFile(path, new Uint8Array(bytes));
      self.postMessage({id, ok: true, meta: {path}});
      return;
    }

    const meta = JSON.parse(bridge.call(name, JSON.stringify(payload || {})));
    if (meta.error) throw new Error(meta.error);

    // Handlers that return several files name them; preview returns one and
    // says so at the call site.
    const n = Array.isArray(meta.files) ? meta.files.length : (blobs || 0);
    const out = [];
    const transfer = [];
    for (let i = 0; i < n; i++) {
      const proxy = bridge.take_blob(i);
      const u8 = proxy.toJs();
      proxy.destroy();
      out.push(u8.buffer);
      transfer.push(u8.buffer);
    }
    self.postMessage({id, ok: true, meta, buffers: out}, transfer);
  } catch (e) {
    self.postMessage({id, ok: false, error: String((e && e.message) || e)});
  }
};
