'use strict';

// The desktop shell: a local Python server answering over HTTP.
//
// app.js reaches everything outside the page through STIPPLE_SHELL, so the
// browser build can swap this for a Web Worker running the same Python
// without touching the interface.

window.STIPPLE_SHELL = (() => {
  async function getJSON(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  }

  async function postJSON(url, body) {
    const r = await fetch(url, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(await r.text());
    return r;
  }

  const q = o => Object.entries(o)
    .map(([k, v]) => k + '=' + encodeURIComponent(v)).join('&');

  return {
    init: () => getJSON('/api/init'),

    async preview(params, view, coarse, crop) {
      const r = await postJSON('/api/preview',
        {params, view, coarse, crop, budget: params.preview_budget});
      return {meta: JSON.parse(r.headers.get('X-Stipple-Meta')),
              buf: await r.arrayBuffer()};
    },

    histogram: (path, paper) =>
      getJSON('/api/histogram?' + q({path, paper})).then(r => r.bins),

    // One call, because the browser shell has the file in hand and would
    // otherwise have to hold it across two.
    async pickInput(paper) {
      const r = await getJSON('/api/browse?save=0&kind=image');
      if (!r.path) return null;
      const s = await getJSON('/api/source?' + q({path: r.path, paper}));
      return {path: r.path, width: s.width, height: s.height};
    },

    async pickOutput() {
      const r = await getJSON('/api/browse?save=1&kind=svg');
      return r.path || null;
    },

    autoTone: params => postJSON('/api/autotone', {params}).then(r => r.json()),

    render: params => postJSON('/api/render', {params}).then(r => r.json()),

    presetSave: params => postJSON('/api/preset', {save: true, params})
      .then(r => r.json()),

    presetLoad: () => postJSON('/api/preset', {save: false}).then(r => r.json()),
  };
})();
