"""The pipeline: image in, marks out."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import image as I
from . import render as R
from . import sample as S
from .density import Geometry, density_field, geometry_for, target_count
from .params import Params

Progress = Callable[[str, float], None] | None


@dataclass
class Result:
    points: np.ndarray                  # (N, 2) canvas points
    darkness: np.ndarray                # (N,)  darkness at each point
    colors: np.ndarray | None           # (N, 3) source colour, if requested
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


def build(p: Params, *, max_edge: int | None = None, progress: Progress = None) -> Result:
    """Run the pipeline and return the marks, without writing anything.

    `max_edge` downscales the source first. The GUI uses it for preview: the
    same code path at a smaller scale, so what you see is the real algorithm
    rather than a stand-in.
    """
    p.validate()
    t0 = time.time()

    rgb = I.load_rgb(p.in_path, paper=p.paper)
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

    return Result(
        points=pts, darkness=dark_at, colors=colors, geom=geom,
        target=target, elapsed=time.time() - t0,
        stats={"source": (src_w, src_h), "sampler": p.sampler},
    )


def generate(p: Params, *, png_dpi: int | None = None, progress: Progress = None) -> Result:
    """Full-quality run, written to `p.out_path`."""
    res = build(p, progress=progress)
    svg = R.write_svg(p.out_path, res.geom, res.points, res.darkness, p, colors=res.colors)
    res.stats.update(svg)

    if png_dpi:
        png_path = p.out_path.rsplit(".", 1)[0] + ".proof.png"
        res.stats["png"] = R.render_png(
            png_path, res.geom, res.points, res.darkness, p,
            dpi=png_dpi, colors=res.colors,
        )
        res.stats["png_path"] = png_path

    return res
