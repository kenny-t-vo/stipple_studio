"""The pipeline: image in, marks out."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from . import image as I
from . import render as R
from . import sample as S
from . import flow as F
from .strokes import build_strokes
from .density import Geometry, density_field, geometry_for, spacing_field, target_count
from .params import Params

Progress = Callable[[str, float], None] | None


@dataclass
class Result:
    points: np.ndarray                  # (N, 2) canvas points
    darkness: np.ndarray                # (N,)  darkness at each point
    colors: np.ndarray | None           # (N, 3) source colour, if requested
    strokes: object | None              # Strokes, when line mode is on
    geom: Geometry
    target: int
    elapsed: float
    stats: dict = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.points)


def _sample_at(field_2d: np.ndarray, pts: np.ndarray, geom: Geometry) -> np.ndarray:
    h, w = field_2d.shape[:2]
    ix = np.clip(((pts[:, 0] - geom.img_x_pt) / geom.img_w_pt * w).astype(np.int32), 0, w - 1)
    iy = np.clip(((pts[:, 1] - geom.img_y_pt) / geom.img_h_pt * h).astype(np.int32), 0, h - 1)
    return field_2d[iy, ix]


def build(p: Params, *, max_edge: int | None = None,
          crop: tuple[float, float, float, float] | None = None,
          rgb: np.ndarray | None = None,
          theta: np.ndarray | None = None,
          progress: Progress = None) -> Result:
    """Run the pipeline and return the marks, without writing anything.

    `max_edge` downscales the source first. The GUI uses it for preview: the
    same code path at a smaller scale, so what you see is the real algorithm
    rather than a stand-in.

    `crop` is (x0, y0, x1, y1) in normalised source coordinates. The canvas
    shrinks to match, so marks come out at their true size -- that is what
    makes the detail view a real 1:1 view of the export rather than a
    scaled-down impression of it.
    """
    p.validate()
    t0 = time.time()

    if rgb is None:
        rgb = I.load_rgb(p.in_path, paper=p.paper)

    if crop:
        h0, w0 = rgb.shape[:2]
        x0, y0, x1, y1 = crop
        cx0, cx1 = sorted((int(x0 * w0), int(x1 * w0)))
        cy0, cy1 = sorted((int(y0 * h0), int(y1 * h0)))
        cx1, cy1 = max(cx1, cx0 + 1), max(cy1, cy0 + 1)
        rgb = rgb[cy0:cy1, cx0:cx1]
        # Scale from the RESOLVED canvas, not from the raw fields. Under a
        # locked aspect the height is derived from the source and canvas_h_in
        # is never read, so it can hold anything; scaling that stale value
        # gave the crop a canvas of the wrong shape and floated the image in
        # a band of empty paper.
        base = geometry_for(p, h0, w0)
        p = replace(p,
                    canvas_w_in=base.canvas_w_in * (cx1 - cx0) / w0,
                    canvas_h_in=base.canvas_h_in * (cy1 - cy0) / h0,
                    lock_aspect=False)

    if max_edge:
        rgb = I.downscale(rgb, max_edge)
    src_h, src_w = rgb.shape[:2]

    luma = I.to_luma(rgb)
    darkness = I.prepare(
        luma,
        invert=p.invert, pre_blur=p.pre_blur,
        black_point=p.black_point, white_point=p.white_point,
        contrast=p.contrast, gamma=p.gamma,
    )

    geom = geometry_for(p, src_h, src_w)
    dens = density_field(darkness, p)
    target = target_count(dens, geom)

    pts = S.sample(p, dens, geom, target, progress=progress)
    dark_at = _sample_at(darkness, pts, geom) if len(pts) else np.zeros(0, dtype=np.float32)
    colors = _sample_at(rgb, pts, geom) if (p.color_mode == "source" and len(pts)) else None

    strokes = None
    if p.line_mode and len(pts):
        # The field is derived from raw luma, before any tone curve, so it
        # survives every levels/gamma/threshold change and the caller can
        # hand back a cached one. It is ~70% of a line-mode preview.
        if theta is None:
            theta, _ = F.build_field(
                luma,
                smoothing=p.flow_smoothing,
                diffusion=p.flow_diffusion,
                bias_angle_deg=p.flow_bias_angle,
                bias_strength=p.flow_bias_strength,
                perpendicular=p.flow_perpendicular,
            )
        strokes = build_strokes(
            pts, dark_at, theta, geom,
            base_r=p.dot_radius_pt,
            spacing=spacing_field(dens, 2.0 * p.dot_radius_pt * p.min_sep_factor),
            length_factor=p.line_length_factor,
            jitter=p.flow_jitter,
            seed=p.seed,
        )

    return Result(
        points=pts, darkness=dark_at, colors=colors, strokes=strokes, geom=geom,
        target=target, elapsed=time.time() - t0,
        stats={"source": (src_w, src_h), "sampler": p.sampler},
    )


def generate(p: Params, *, png_dpi: int | None = None, progress: Progress = None) -> Result:
    """Full-quality run, written to `p.out_path`."""
    res = build(p, progress=progress)
    svg = R.write_svg(p.out_path, res.geom, res.points, res.darkness, p,
                      colors=res.colors, strokes=res.strokes)
    res.stats.update(svg)

    if png_dpi:
        png_path = p.out_path.rsplit(".", 1)[0] + ".proof.png"
        res.stats["png"] = R.render_png(
            png_path, res.geom, res.points, res.darkness, p,
            dpi=png_dpi, colors=res.colors, strokes=res.strokes,
        )
        res.stats["png_path"] = png_path

    return res
