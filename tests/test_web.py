"""Tests for the preview engine and the local HTTP backend."""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from dataclasses import replace
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web"))

from engine import Engine, pack            # noqa: E402
from stipple.params import Params          # noqa: E402


@pytest.fixture(scope="module")
def image(tmp_path_factory):
    d = tmp_path_factory.mktemp("web")
    w, h = 240, 180
    yy, xx = np.mgrid[0:h, 0:w]
    a = (128 + 110 * np.sin(xx / 11.0) * np.cos(yy / 9.0)).astype(np.uint8)
    p = d / "src.png"
    Image.fromarray(a, "L").convert("RGB").save(p)
    return str(p)


@pytest.fixture
def params(image, tmp_path):
    # Big enough that the preview must scale down, which is the case worth
    # testing: at these settings the export runs to ~180k marks.
    return Params(in_path=image, out_path=str(tmp_path / "o.svg"),
                  canvas_w_in=20.0, max_density=0.10, relax_iterations=3)


# ── engine ───────────────────────────────────────────────────────────

def test_coarse_is_cheaper_than_full(params):
    e = Engine()
    coarse, _ = e.preview(params, coarse=True)
    full, _ = e.preview(params, coarse=False)
    assert coarse.count < full.count


def test_preview_preserves_ink_coverage(params):
    """Tone is ink coverage. Scaling the canvas must not change it.

    The first version scaled mark radius along with the canvas, which scales
    coverage by k^4 and left the preview roughly 8x lighter than the print.
    """
    from stipple.core import build
    e = Engine()
    export = build(params)
    prev, k = e.preview(params, coarse=False)
    assert k < 1.0, "this fixture should need scaling down"

    def coverage(res):
        area = res.geom.canvas_w_pt * res.geom.canvas_h_pt
        return res.count * np.pi * params.dot_radius_pt ** 2 / area

    assert coverage(prev) == pytest.approx(coverage(export), rel=0.05)


def test_full_preview_reports_the_exact_export_count(params):
    from stipple.core import build
    e = Engine()
    res, k = e.preview(params, coarse=False)
    _, meta = pack(res, k, params)
    export = build(params)
    assert meta["fullTarget"] == export.target == export.count


def test_detail_view_is_true_scale(params):
    e = Engine()
    res, k = e.preview(params, view="detail", crop=(0.4, 0.4, 0.5, 0.5))
    assert k == 1.0
    assert res.geom.canvas_w_in == pytest.approx(params.canvas_w_in * 0.1, rel=.02)


def test_flow_cache_is_reused_and_correct(params):
    """Cached fields must be identical to freshly computed ones."""
    p = replace(params, line_mode=True)
    e = Engine()
    rgb = e.proxy(p.in_path, p.paper, 400)
    a = e.flow_field(p, rgb)
    b = e.flow_field(p, rgb)
    assert a is b, "second call should hit the cache"

    from stipple.flow import build_field
    from stipple.image import to_luma
    fresh, _ = build_field(to_luma(rgb), smoothing=p.flow_smoothing,
                           diffusion=p.flow_diffusion,
                           bias_angle_deg=p.flow_bias_angle,
                           bias_strength=p.flow_bias_strength,
                           perpendicular=p.flow_perpendicular)
    assert np.array_equal(a, fresh)


def test_flow_cache_invalidates_on_flow_change(params):
    p = replace(params, line_mode=True)
    e = Engine()
    rgb = e.proxy(p.in_path, p.paper, 400)
    a = e.flow_field(p, rgb)
    b = e.flow_field(replace(p, flow_smoothing=p.flow_smoothing + 4), rgb)
    assert not np.array_equal(a, b)


def test_flow_cache_survives_tone_changes(params):
    """The field comes from raw luma, so tone must not invalidate it."""
    p = replace(params, line_mode=True)
    e = Engine()
    rgb = e.proxy(p.in_path, p.paper, 400)
    a = e.flow_field(p, rgb)
    b = e.flow_field(replace(p, gamma=p.gamma + 2.0, black_point=0.3), rgb)
    assert a is b


def test_pack_round_trips_dots(params):
    e = Engine()
    res, k = e.preview(params, coarse=True)
    body, meta = pack(res, k, params)
    assert meta["mode"] == "dots" and meta["stride"] == 3
    arr = np.frombuffer(body, dtype=np.float32).reshape(-1, 3)
    assert len(arr) == meta["count"]
    assert np.allclose(arr[:, :2], res.points, atol=1e-3)


def test_pack_round_trips_strokes(params):
    e = Engine()
    res, k = e.preview(replace(params, line_mode=True), coarse=True)
    body, meta = pack(res, k, replace(params, line_mode=True))
    assert meta["mode"] == "strokes" and meta["stride"] == 6
    arr = np.frombuffer(body, dtype=np.float32).reshape(-1, 6)
    assert np.allclose(arr[:, 0:2], res.strokes.p0, atol=1e-3)


def test_pack_appends_colour_bytes(params):
    e = Engine()
    p = replace(params, color_mode="source")
    res, k = e.preview(p, coarse=True)
    body, meta = pack(res, k, p)
    assert meta["hasColor"] and meta["colorBytes"] == meta["count"] * 3
    assert len(body) == meta["count"] * meta["stride"] * 4 + meta["colorBytes"]


# ── server ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def base_url(tmp_path_factory):
    # Point the server's state directory at a temp dir; /api/render persists
    # whatever params it is given, and that must not land in the real one.
    import os
    os.environ["STIPPLE_STATE_DIR"] = str(tmp_path_factory.mktemp("state"))
    import server as S
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.status, r.read(), dict(r.headers)


def _post(url, obj):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, r.read(), dict(r.headers)


@pytest.mark.parametrize("path,ctype", [
    ("/", "text/html"), ("/app.css", "text/css"), ("/app.js", "javascript"),
])
def test_static_files_served(base_url, path, ctype):
    st, body, h = _get(base_url + path)
    assert st == 200 and len(body) > 200
    assert ctype in h["Content-Type"]


def test_init_exposes_every_param(base_url):
    st, body, _ = _get(base_url + "/api/init")
    d = json.loads(body)
    from dataclasses import fields
    assert set(d["defaults"]) == {f.name for f in fields(Params)}


def test_preview_returns_binary_and_meta(base_url, params):
    from dataclasses import asdict
    st, body, h = _post(base_url + "/api/preview",
                        {"params": asdict(params), "coarse": True})
    assert st == 200
    meta = json.loads(h["X-Stipple-Meta"])
    assert meta["count"] > 0
    assert len(body) == meta["count"] * meta["stride"] * 4


def test_render_does_not_touch_the_real_saved_settings(base_url):
    """Running the suite must not overwrite the user's last-used params."""
    import server as S
    assert "pytest" in str(S.STATE_DIR) or "tmp" in str(S.STATE_DIR).lower()
    real = Path.home() / ".local" / "share" / "stipple" / "last.json"
    assert S.LAST != real


def test_render_writes_the_file(base_url, params, tmp_path):
    from dataclasses import asdict
    out = tmp_path / "web.svgz"
    p = asdict(replace(params, out_path=str(out)))
    st, body, _ = _post(base_url + "/api/render", {"params": p})
    assert st == 200
    j = json.loads(body)
    assert out.exists() and j["marks"] > 0 and j["name"] == "web.svgz"


def test_histogram_and_source(base_url, image):
    from urllib.parse import quote
    st, body, _ = _get(f"{base_url}/api/histogram?path={quote(image)}&paper=%23ffffff")
    assert st == 200 and len(json.loads(body)["bins"]) == 96
    st, body, _ = _get(f"{base_url}/api/source?path={quote(image)}&paper=%23ffffff")
    assert json.loads(body) == {"width": 240, "height": 180}


def test_invalid_params_are_client_errors(base_url, params):
    from dataclasses import asdict
    bad = asdict(replace(params, sampler="nope"))
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base_url + "/api/preview", {"params": bad})
    assert e.value.code == 400


def test_missing_file_is_404(base_url, params):
    from dataclasses import asdict
    bad = asdict(replace(params, in_path="/no/such/image.png"))
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base_url + "/api/preview", {"params": bad})
    assert e.value.code == 404


def test_render_without_input_is_rejected(base_url, params):
    from dataclasses import asdict
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base_url + "/api/render", {"params": asdict(replace(params, in_path=""))})
    assert e.value.code == 400
