"""Local HTTP backend and desktop shell.

The window is pywebview when it is available -- a real macOS window with no
browser chrome -- and falls back to the default browser otherwise. The
server binds to 127.0.0.1 on an ephemeral port and is only ever spoken to
by the page it serves.
"""

from __future__ import annotations

import json
import mimetypes
import socket
import sys
import threading
import webbrowser
from dataclasses import asdict, fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from engine import Engine, pack                      # noqa: E402
from stipple import __version__                      # noqa: E402
from stipple.core import generate                    # noqa: E402
from stipple.params import Params                    # noqa: E402

STATE_DIR = Path.home() / ".local" / "share" / "stipple"
LAST = STATE_DIR / "last.json"
PRESETS = ROOT.parent / "presets"

ENGINE = Engine()
WINDOW = None            # pywebview window, when we have one
_PARAM_NAMES = {f.name for f in fields(Params)}


def _params(raw: dict) -> Params:
    p = Params()
    for k, v in (raw or {}).items():
        if k in _PARAM_NAMES:
            setattr(p, k, v)
    p.validate()
    return p


def _save_last(p: Params) -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        LAST.write_text(json.dumps(asdict(p), indent=1), encoding="utf-8")
    except OSError:
        pass


def _dialog(save: bool, kind: str):
    """Native file dialog through pywebview, when there is a window."""
    if WINDOW is None:
        return None
    import webview
    if save:
        r = WINDOW.create_file_dialog(webview.SAVE_DIALOG, save_filename="output.svg")
    else:
        types = ("Images (*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp;*.webp)",) \
            if kind == "image" else ("Preset (*.json)",)
        r = WINDOW.create_file_dialog(webview.OPEN_DIALOG, file_types=types)
    if not r:
        return None
    return r if isinstance(r, str) else r[0]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):           # keep the terminal quiet
        pass

    # ── plumbing ─────────────────────────────────────────────────────

    def _send(self, code, body=b"", ctype="application/octet-stream", headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _fail(self, exc, code=400):
        self._send(code, str(exc).encode(), "text/plain; charset=utf-8")

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    # ── routes ───────────────────────────────────────────────────────

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self._static("index.html")
            if u.path in ("/app.css", "/app.js"):
                return self._static(u.path.lstrip("/"))

            if u.path == "/api/init":
                last = {}
                if LAST.exists():
                    try:
                        last = json.loads(LAST.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        last = {}
                return self._json({
                    "version": __version__,
                    "defaults": asdict(Params()),
                    "last": last,
                    "native": WINDOW is not None,
                })

            if u.path == "/api/histogram":
                return self._json({"bins": ENGINE.histogram(
                    q["path"], q.get("paper", "#ffffff"))})

            if u.path == "/api/source":
                return self._json(ENGINE.source_size(
                    q["path"], q.get("paper", "#ffffff")))

            if u.path == "/api/browse":
                return self._json({"path": _dialog(q.get("save") == "1",
                                                   q.get("kind", "image"))})

            self._send(404, b"not found", "text/plain")
        except FileNotFoundError as e:
            self._fail(f"file not found: {e}", 404)
        except Exception as e:                        # noqa: BLE001
            self._fail(e, 500)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            body = self._body()

            if u.path == "/api/preview":
                p = _params(body.get("params"))
                res, k = ENGINE.preview(
                    p,
                    view=body.get("view", "fit"),
                    crop=body.get("crop"),
                    coarse=bool(body.get("coarse")),
                )
                data, meta = pack(res, k, p)
                return self._send(200, data, "application/octet-stream",
                                  {"X-Stipple-Meta": json.dumps(meta)})

            if u.path == "/api/render":
                p = _params(body.get("params"))
                if not p.in_path:
                    raise ValueError("no input image")
                if not p.out_path:
                    raise ValueError("no output path")
                res = generate(p, png_dpi=p.png_dpi or None)
                _save_last(p)
                return self._json({
                    "name": Path(p.out_path).name,
                    "path": p.out_path,
                    "marks": res.count,
                    "mb": round(res.stats.get("bytes", 0) / 1048576, 2),
                    "secs": round(res.elapsed, 1),
                })

            if u.path == "/api/preset":
                return self._preset(body)

            self._send(404, b"not found", "text/plain")
        except (ValueError, KeyError) as e:
            self._fail(e, 400)
        except FileNotFoundError as e:
            self._fail(f"file not found: {e}", 404)
        except Exception as e:                        # noqa: BLE001
            self._fail(e, 500)

    def _preset(self, body):
        PRESETS.mkdir(parents=True, exist_ok=True)
        if body.get("save"):
            path = _dialog(True, "preset")
            if not path:
                return self._json({"path": None})
            if not path.endswith(".json"):
                path += ".json"
            p = _params(body.get("params"))
            p.save_preset(path, name=Path(path).stem)
            return self._json({"path": path, "name": Path(path).stem})

        path = _dialog(False, "preset")
        if not path:
            return self._json({"path": None})
        loaded = Params.load_preset(path)
        data = {k: v for k, v in asdict(loaded).items() if k not in Params._JOB_KEYS}
        return self._json({"path": path, "name": Path(path).stem, "params": data})

    def _static(self, name):
        f = ROOT / name
        if not f.is_file():
            return self._send(404, b"not found", "text/plain")
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or name.endswith(".js"):
            ctype += "; charset=utf-8"
        self._send(200, f.read_bytes(), ctype)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(port: int | None = None, open_window: bool = True):
    port = port or _free_port()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"

    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    # flush: stdout is block-buffered when piped, and a launcher that
    # prints nothing until its buffer fills looks like a hang.
    print(f"stipple — {url}", flush=True)

    if not open_window:
        try:
            t.join()
        except KeyboardInterrupt:
            pass
        return

    global WINDOW
    try:
        import webview
    except ImportError:
        print("pywebview not installed; opening the default browser instead.",
              flush=True)
        webbrowser.open(url)
        try:
            t.join()
        except KeyboardInterrupt:
            pass
        return

    WINDOW = webview.create_window("stipple", url, width=1380, height=920,
                                   min_size=(1040, 700), background_color="#ffffff")
    webview.start()          # macOS requires this on the main thread


if __name__ == "__main__":
    args = sys.argv[1:]
    serve(open_window="--no-window" not in args)
