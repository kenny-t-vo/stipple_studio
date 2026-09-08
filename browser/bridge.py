"""Pyodide-side dispatch: the browser build's answer to web/server.py.

The same handlers without HTTP and without native dialogs. The worker writes
whatever file the user chose into the virtual filesystem before calling, so
everything below reads and writes paths exactly as the desktop build does and
the pipeline needs no browser-specific code.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path

from engine import Engine, pack
from stipple import __version__
from stipple.core import generate
from stipple.params import Params

# numpy on WASM, single threaded, runs the sampler about five times slower
# than the desktop does, and relaxation is most of a preview. Measured
# normalised CoV against iteration count on the reference image: 0.288 at 2,
# 0.246 at 8, 0.244 at 20, flat after that. Eight costs under 1% of the
# evenness and halves the wait. The export is not capped.
Engine.FULL = dict(Engine.FULL, relax=8)

ENGINE = Engine()
_NAMES = {f.name for f in fields(Params)}

#: Binary results of the last call, in the order the caller should take them.
_blobs: list[bytes] = []


def _params(raw: dict) -> Params:
    p = Params()
    for k, v in (raw or {}).items():
        if k in _NAMES:
            setattr(p, k, v)
    p.validate()
    return p


def take_blob(i: int = 0) -> bytes:
    """One binary result of the last call. The worker copies it out to JS."""
    return _blobs[i] if i < len(_blobs) else b""


def call(name: str, payload: str) -> str:
    """JSON in, JSON out. Errors come back as {"error": ...}, not exceptions,
    so the worker does not have to unwrap a Python traceback across the
    boundary."""
    _blobs.clear()
    try:
        return json.dumps(_dispatch(name, json.loads(payload or "{}")))
    except Exception as e:                            # noqa: BLE001
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


def _dispatch(name: str, body: dict) -> dict:
    if name == "init":
        return {"version": __version__, "defaults": asdict(Params()), "native": False}

    if name == "histogram":
        return {"bins": ENGINE.histogram(body["path"], body.get("paper", "#ffffff"))}

    if name == "source":
        return ENGINE.source_size(body["path"], body.get("paper", "#ffffff"))

    if name == "preview":
        p = _params(body.get("params"))
        res, k = ENGINE.preview(p, view=body.get("view", "fit"),
                                crop=body.get("crop"),
                                coarse=bool(body.get("coarse")))
        blob, meta = pack(res, k, p)
        _blobs.append(blob)
        return meta

    if name == "render":
        p = _params(body.get("params"))
        if not p.in_path:
            raise ValueError("no input image")
        if not p.out_path:
            raise ValueError("no output path")
        res = generate(p, png_dpi=p.png_dpi or None)

        names = [p.out_path]
        if res.stats.get("png_path"):
            names.append(res.stats["png_path"])
        for n in names:
            _blobs.append(Path(n).read_bytes())

        return {"name": Path(p.out_path).name,
                "files": [Path(n).name for n in names],
                "marks": res.count,
                "mb": round(res.stats.get("bytes", 0) / 1048576, 2),
                "secs": round(res.elapsed, 1)}

    if name == "preset_save":
        p = _params(body.get("params"))
        stem = body.get("name") or "preset"
        _blobs.append((json.dumps({"name": stem, "params": p.to_preset()},
                                  indent=2) + "\n").encode())
        return {"name": stem, "files": [stem + ".json"]}

    if name == "preset_load":
        raw = json.loads(body["text"])
        p = Params().apply_preset(raw.get("params", raw))
        data = {k: v for k, v in asdict(p).items() if k not in Params._JOB_KEYS}
        return {"name": body.get("name") or "preset", "params": data}

    raise ValueError(f"unknown call: {name}")
