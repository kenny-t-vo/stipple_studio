"""SVG and PNG output."""

from __future__ import annotations

import gzip
import io
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from .density import Geometry
from .params import Params, PT_PER_INCH

_RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS


# ── ordering ─────────────────────────────────────────────────────────

def boustrophedon(pts: np.ndarray, band_pt: float) -> np.ndarray:
    """Order points in serpentine bands: left-to-right, then right-to-left.

    Two payoffs. Consecutive points end up physically close, so emitting
    relative path deltas produces small numbers and a materially smaller
    file. And an editor (or a pen) walking the path in order does far less
    jumping around than with unsorted points.
    """
    if len(pts) == 0:
        return np.zeros(0, dtype=np.int64)
    band = np.floor(pts[:, 1] / max(band_pt, 1e-6)).astype(np.int64)
    x = pts[:, 0].copy()
    x[band % 2 == 1] *= -1.0          # reverse direction on odd bands
    return np.lexsort((x, band))


# ── colour ───────────────────────────────────────────────────────────

def quantize_colors(rgb: np.ndarray, levels: int = 6):
    """Snap per-point colours to a coarse cube so they group into few paths.

    Emitting one element per point with its own fill would defeat the whole
    point of the compound path. Quantising to `levels` steps per channel
    yields at most levels^3 groups, each of which becomes one path.
    """
    q = np.clip((rgb * (levels - 1)).round().astype(np.int32), 0, levels - 1)
    keys = (q[:, 0] * levels + q[:, 1]) * levels + q[:, 2]
    out = q.astype(np.float32) / (levels - 1)
    return keys, out


def _hex(rgb: Iterable[float]) -> str:
    r, g, b = (int(round(float(c) * 255)) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


# ── SVG ──────────────────────────────────────────────────────────────

def _num(v: float, places: int = 1) -> str:
    """Shortest unambiguous SVG number: no trailing zeros, no leading zero."""
    s = f"{v:.{places}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s.startswith("0.") and len(s) > 2:
        s = s[1:]
    elif s.startswith("-0.") and len(s) > 3:
        s = "-" + s[2:]
    return s or "0"


def _arc(r: float) -> str:
    """The two-arc circle suffix for radius r, as a relative path fragment."""
    rr, d2 = _num(r, 3), _num(2.0 * r, 3)
    neg = d2 if d2.startswith("-") else "-" + d2
    return f"a{rr},{rr} 0 1 0 {d2},0a{rr},{rr} 0 1 0{neg},0"


def _dot_path(buf, pts: np.ndarray, radii: np.ndarray) -> None:
    """Append circles to `buf` as one compound path.

    Coordinates are relative to the previous dot, which after serpentine
    ordering keeps them to one or two digits. Radius is constant unless
    dots scale with tone, so the arc fragment is built once and reused.
    Precision is 0.1pt: 1/720 inch, finer than any printer resolves.
    """
    if len(pts) == 0:
        return
    const_r = float(radii[0])
    uniform = bool(np.all(radii == radii[0]))
    suffix = _arc(const_r) if uniform else ""

    px = py = 0.0
    out = []
    for i, (x, y) in enumerate(pts):
        out.append("m")
        out.append(_num(x - px))
        out.append(",")
        out.append(_num(y - py))
        out.append(suffix if uniform else _arc(float(radii[i])))
        px, py = x, y
    buf.write("".join(out))


def write_svg(
    path: str,
    geom: Geometry,
    pts: np.ndarray,
    dark: np.ndarray,
    p: Params,
    colors: np.ndarray | None = None,
    strokes: list | None = None,
) -> dict:
    """Write the SVG. Returns a small stats dict."""
    w_in = geom.canvas_w_pt / PT_PER_INCH
    h_in = geom.canvas_h_pt / PT_PER_INCH

    buf = io.StringIO()
    buf.write(
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg"\n'
        f'     width="{w_in:.4f}in" height="{h_in:.4f}in"\n'
        f'     viewBox="0 0 {geom.canvas_w_pt:.2f} {geom.canvas_h_pt:.2f}">\n'
    )
    if p.paper_background:
        buf.write(f'  <rect x="0" y="0" width="{geom.canvas_w_pt:.2f}" '
                  f'height="{geom.canvas_h_pt:.2f}" fill="{p.paper}"/>\n')

    if strokes is not None:
        _write_strokes(buf, strokes, p, colors)
    elif len(pts):
        order = boustrophedon(pts, band_pt=max(4.0 * p.dot_radius_pt, 2.0))
        pts, dark = pts[order], dark[order]
        if colors is not None:
            colors = colors[order]

        radii = np.full(len(pts), p.dot_radius_pt, dtype=np.float64)
        if p.scale_with_darkness:
            radii = p.dot_radius_pt * (1.0 + (p.max_radius_scale - 1.0) * dark)

        if p.color_mode == "source" and colors is not None:
            keys, snapped = quantize_colors(colors)
            buf.write('  <g stroke="none">\n')
            for k in np.unique(keys):
                m = keys == k
                buf.write(f'    <path fill="{_hex(snapped[m][0])}" d="')
                _dot_path(buf, pts[m], radii[m])
                buf.write('"/>\n')
            buf.write('  </g>\n')
        elif p.svg_structure == "compound":
            buf.write(f'  <path fill="{p.ink}" stroke="none" d="')
            _dot_path(buf, pts, radii)
            buf.write('"/>\n')
        else:
            buf.write(f'  <g fill="{p.ink}" stroke="none">\n')
            for (x, y), r in zip(pts, radii):
                buf.write(f'    <circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.3f}"/>\n')
            buf.write('  </g>\n')

    buf.write('</svg>\n')
    data = buf.getvalue().encode("utf-8")
    if path.lower().endswith(".svgz"):
        # Illustrator opens gzipped SVG directly, and the payload is almost
        # all repeated path syntax, so it compresses hard.
        with gzip.open(path, "wb", compresslevel=6) as f:
            f.write(data)
        written = len(Path(path).read_bytes())
    else:
        with open(path, "wb") as f:
            f.write(data)
        written = len(data)
    return {"bytes": written, "raw_bytes": len(data), "marks": len(pts)}


def _write_strokes(buf, strokes, p: Params, colors) -> None:
    """Placeholder until the flow-field mode lands in phase 2."""
    raise NotImplementedError("stroke rendering arrives with the flow field")


# ── PNG proof ────────────────────────────────────────────────────────

def render_png(
    path: str,
    geom: Geometry,
    pts: np.ndarray,
    dark: np.ndarray,
    p: Params,
    dpi: int = 150,
    supersample: int = 2,
    colors: np.ndarray | None = None,
) -> dict:
    """Rasterise a proof so tone and density can be judged without Illustrator."""
    scale = dpi / PT_PER_INCH
    w = max(1, int(round(geom.canvas_w_pt * scale)))
    h = max(1, int(round(geom.canvas_h_pt * scale)))
    ss = max(1, int(supersample))

    img = Image.new("RGB", (w * ss, h * ss), p.paper)
    draw = ImageDraw.Draw(img)

    radii = np.full(len(pts), p.dot_radius_pt, dtype=np.float64)
    if p.scale_with_darkness:
        radii = p.dot_radius_pt * (1.0 + (p.max_radius_scale - 1.0) * dark)

    use_color = p.color_mode == "source" and colors is not None
    fill = p.ink
    s = scale * ss
    for i, ((x, y), r) in enumerate(zip(pts, radii)):
        cx, cy, rr = x * s, y * s, max(r * s, 0.35)
        if use_color:
            fill = _hex(colors[i])
        draw.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=fill)

    if ss > 1:
        img = img.resize((w, h), _RESAMPLE)
    img.save(path, dpi=(dpi, dpi))
    return {"width": w, "height": h, "dpi": dpi}
