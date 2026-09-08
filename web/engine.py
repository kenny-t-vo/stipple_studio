"""Preview engine: the pipeline at interactive speed.

Two views, both produced by the same code as the export.

  fit     the whole canvas, scaled down until the mark count is small
          enough to draw quickly. Marks scale with it, so the result is a
          faithful reduction.
  detail  a crop rendered at full canvas scale, so marks come out at the
          size they will print. A 0.25pt dot on a 28in canvas is invisible
          in any whole-canvas view.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path

import numpy as np

from stipple import flow as F
from stipple import image as I
from stipple.core import build
from stipple.density import density_field, geometry_for, target_count
from stipple.params import Params


class Engine:
    """Holds the decoded source image so slider moves do not re-read it."""

    #: mark budgets and proxy sizes for the two preview qualities. Coarse
    #: runs while a slider is moving and must stay well inside a frame; full
    #: runs once the control is released.
    COARSE = dict(budget=7_000, proxy_edge=560, relax=4, diffusion=3)
    FULL = dict(budget=26_000, proxy_edge=1100, relax=None, diffusion=None)

    def __init__(self, budget: int = 26_000, proxy_edge: int = 1100):
        self.budget = budget
        self.proxy_edge = proxy_edge
        self._lock = threading.Lock()
        self._key = None
        self._full = None      # full-resolution RGB
        self._proxy = {}       # edge -> downscaled RGB
        self._flow = {}        # flow-field cache, see flow_field()

    # ── source ───────────────────────────────────────────────────────

    def load(self, path: str, paper: str):
        st = Path(path).stat()
        key = (path, st.st_mtime_ns, st.st_size, paper)
        with self._lock:
            if key != self._key:
                self._full = I.load_rgb(path, paper=paper)
                self._proxy = {}
                self._flow = {}
                self._key = key
            return self._full

    def proxy(self, path: str, paper: str, edge: int):
        full = self.load(path, paper)
        with self._lock:
            got = self._proxy.get(edge)
            if got is None:
                got = I.downscale(full, edge)
                self._proxy[edge] = got
            return got

    def histogram(self, path: str, paper: str, bins: int = 96):
        return I.histogram(I.to_luma(self.proxy(path, paper, self.proxy_edge)), bins=bins)

    def auto_tone(self, p: Params) -> dict:
        """Levels and gamma for the source, at full resolution.

        Not the proxy: gamma is solved to hold mean darkness, and mean
        darkness is the mark count, which `exact_target` measures on the full
        image. Solving on the proxy moved the count by about 1%.
        """
        luma = I.to_luma(self.load(p.in_path, p.paper))
        black, white, gamma = I.auto_levels(
            luma, invert=p.invert, pre_blur=p.pre_blur,
            black_point=p.black_point, white_point=p.white_point,
            contrast=p.contrast, gamma=p.gamma,
        )
        return {"black_point": round(black, 4), "white_point": round(white, 4),
                "gamma": gamma}

    def source_size(self, path: str, paper: str):
        full = self.load(path, paper)
        return {"width": int(full.shape[1]), "height": int(full.shape[0])}

    # ── previews ─────────────────────────────────────────────────────

    def _fit_scale(self, p: Params, proxy: np.ndarray, budget: int) -> float:
        """How far to shrink the canvas so the mark count stays affordable."""
        dark = I.prepare(
            I.to_luma(proxy),
            invert=p.invert, pre_blur=0.0,
            black_point=p.black_point, white_point=p.white_point,
            contrast=p.contrast, gamma=p.gamma,
        )
        geom = geometry_for(p, *proxy.shape[:2])
        n = target_count(density_field(dark, p), geom)
        if n <= budget:
            return 1.0, n
        return float(np.sqrt(budget / max(n, 1))), n

    def exact_target(self, p: Params, full: np.ndarray) -> int:
        """Mark count for the real export, from the full-resolution source."""
        dark = I.prepare(
            I.to_luma(full),
            invert=p.invert, pre_blur=p.pre_blur,
            black_point=p.black_point, white_point=p.white_point,
            contrast=p.contrast, gamma=p.gamma,
        )
        geom = geometry_for(p, *full.shape[:2])
        return target_count(density_field(dark, p), geom)

    def flow_field(self, p: Params, rgb: np.ndarray):
        """Cached edge-tangent field for this image and these flow settings.

        Costs about 70% of a line-mode preview and depends on no tone or sampling
        parameter -- it is built from raw luma. Dragging density, length, seed or
        mark size in line mode reuses it.
        """
        if not p.line_mode:
            return None
        key = (rgb.shape, round(p.flow_smoothing, 4), int(p.flow_diffusion),
               round(p.flow_bias_angle, 3), round(p.flow_bias_strength, 4),
               bool(p.flow_perpendicular))
        with self._lock:
            got = self._flow.get(key)
        if got is None:
            got, _ = F.build_field(
                I.to_luma(rgb),
                smoothing=p.flow_smoothing, diffusion=p.flow_diffusion,
                bias_angle_deg=p.flow_bias_angle,
                bias_strength=p.flow_bias_strength,
                perpendicular=p.flow_perpendicular,
            )
            with self._lock:
                if len(self._flow) > 6:
                    self._flow.clear()
                self._flow[key] = got
        return got

    def preview(self, p: Params, view: str = "fit", crop=None,
                coarse: bool = False, budget: int | None = None):
        q = self.COARSE if coarse else self.FULL
        # The coarse pass has to stay inside a frame, so it keeps its own
        # budget whatever the interface asks for.
        want = q["budget"] if coarse or not budget else int(budget)
        # Independent, so a build can cap one without the other.
        if q["relax"] is not None:
            p = replace(p, relax_iterations=min(p.relax_iterations, q["relax"]))
        if q["diffusion"] is not None:
            p = replace(p, flow_diffusion=min(p.flow_diffusion, q["diffusion"]))

        if view == "detail" and crop:
            # True scale: full canvas, small window onto it.
            src = self.load(p.in_path, p.paper)
            res = build(p, rgb=src, crop=tuple(crop))
            res.stats["full_target"] = res.target
            res.stats["full_w_in"] = res.geom.canvas_w_in
            res.stats["full_h_in"] = res.geom.canvas_h_in
            return res, 1.0

        full = self.load(p.in_path, p.paper)
        proxy = self.proxy(p.in_path, p.paper, q["proxy_edge"])

        k, full_target = self._fit_scale(p, proxy, want)
        # pre_blur is measured in source pixels, so it must shrink with the proxy.
        px_scale = proxy.shape[1] / max(full.shape[1], 1)
        # Shrink the canvas but NOT the marks. Density is dots per point
        # squared, so a canvas scaled by k carries k^2 marks; scaling the
        # radius too would scale ink coverage by k^4 -- 8x lighter than the
        # print at k = 0.35. Holding mark size fixed keeps coverage, and
        # therefore tone, exact. Texture comes out magnified by 1/k, so
        # sub-point marks stay visible in a whole-canvas view.
        scaled = replace(
            p,
            canvas_w_in=p.canvas_w_in * k,
            canvas_h_in=p.canvas_h_in * k,
            pre_blur=p.pre_blur * px_scale,
        )
        res = build(scaled, rgb=proxy, theta=self.flow_field(scaled, proxy))
        if not coarse:
            # The proxy's mean darkness drifts a couple of percent from the
            # full image's, so a count derived from it is an estimate. Compute
            # it properly on the refine pass; it costs a fraction of the
            # sampling, and the count must match the export.
            full_target = self.exact_target(p, full)
        # Resolve the canvas rather than reporting the fields: under a locked
        # aspect canvas_h_in is never read by the pipeline and holds whatever
        # it last held, so the readout would show a stale height.
        shown = geometry_for(p, *full.shape[:2])
        res.stats["full_target"] = full_target
        res.stats["full_w_in"] = shown.canvas_w_in
        res.stats["full_h_in"] = shown.canvas_h_in
        return res, k


# ── wire format ──────────────────────────────────────────────────────
#
# Marks go to the browser as raw float32 rather than JSON: 26k strokes is
# 624KB of binary against several megabytes of decimal text, and the
# browser can hand the buffer straight to a typed array.

def pack(res, k: float, p: Params) -> tuple[bytes, dict]:
    g = res.geom
    meta = {
        "mode": "strokes" if res.strokes is not None else "dots",
        "count": int(res.count),
        "target": int(res.target),
        "canvasW": g.canvas_w_pt,
        "canvasH": g.canvas_h_pt,
        "imgX": g.img_x_pt, "imgY": g.img_y_pt,
        "imgW": g.img_w_pt, "imgH": g.img_h_pt,
        "radius": p.dot_radius_pt,
        "scale": k,
        "elapsed": round(res.elapsed, 3),
        "ink": p.ink, "paper": p.paper,
        "taper": bool(p.line_taper),
        "scaleWithDarkness": bool(p.scale_with_darkness),
        "maxRadiusScale": p.max_radius_scale,
        "hasColor": res.colors is not None,
        "fullTarget": int(res.stats.get("full_target", res.target)),
        "fullW": float(res.stats.get("full_w_in", res.geom.canvas_w_in)),
        "fullH": float(res.stats.get("full_h_in", res.geom.canvas_h_in)),
    }

    if res.strokes is not None:
        st = res.strokes
        arr = np.empty((len(st), 6), dtype=np.float32)
        arr[:, 0:2] = st.p0
        arr[:, 2:4] = st.ctrl
        arr[:, 4:6] = st.p2
        meta["stride"] = 6
    else:
        arr = np.empty((res.count, 3), dtype=np.float32)
        arr[:, 0:2] = res.points
        arr[:, 2] = res.darkness
        meta["stride"] = 3

    body = arr.tobytes()
    if res.colors is not None:
        rgb = (np.clip(res.colors, 0, 1) * 255).astype(np.uint8)
        body += rgb.tobytes()
        meta["colorBytes"] = int(rgb.size)
    return body, meta
