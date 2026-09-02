"""Canvas geometry and the density field that drives sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import Params, PT_PER_INCH


@dataclass(frozen=True)
class Geometry:
    """Where the image sits on the canvas, in points.

    When the aspect ratio is locked the image fills the canvas exactly.
    When unlocked the image is fitted (never cropped) and centred on both
    axes, leaving paper-coloured margins.
    """
    canvas_w_pt: float
    canvas_h_pt: float
    img_x_pt: float
    img_y_pt: float
    img_w_pt: float
    img_h_pt: float

    @property
    def canvas_w_in(self) -> float:
        return self.canvas_w_pt / PT_PER_INCH

    @property
    def canvas_h_in(self) -> float:
        return self.canvas_h_pt / PT_PER_INCH

    @property
    def image_area_pt2(self) -> float:
        return self.img_w_pt * self.img_h_pt

    @property
    def has_margins(self) -> bool:
        return self.img_x_pt > 1e-6 or self.img_y_pt > 1e-6


def geometry_for(p: Params, src_h: int, src_w: int) -> Geometry:
    """Resolve canvas and image placement for a source of `src_w` x `src_h`."""
    aspect = src_h / float(src_w)
    cw = p.canvas_w_in * PT_PER_INCH

    if p.lock_aspect:
        ch = cw * aspect
        return Geometry(cw, ch, 0.0, 0.0, cw, ch)

    ch = p.canvas_h_in * PT_PER_INCH
    scale = min(cw / src_w, ch / src_h)
    iw, ih = src_w * scale, src_h * scale
    return Geometry(cw, ch, (cw - iw) / 2.0, (ch - ih) / 2.0, iw, ih)


def density_field(darkness: np.ndarray, p: Params) -> np.ndarray:
    """Darkness in [0,1] -> density in dots per point squared."""
    d = p.min_density + darkness * (p.max_density - p.min_density)
    if p.threshold > 0:
        d = np.where(darkness >= p.threshold, d, 0.0)
    return d.astype(np.float32)


def target_count(dens: np.ndarray, geom: Geometry) -> int:
    """Expected dot count: the density field integrated over the image area."""
    return int(round(float(dens.mean()) * geom.image_area_pt2))


#: Random disk packing saturates well below the hexagonal limit, so a disk
#: radius of 1/sqrt(d) yields fewer than d points per unit area. Measured on
#: the reference image: an uncorrected radius filled 87% of the target count,
#: with the shortfall entirely in the darkest regions, which saturate first
#: and flattened tone. 0.72 was the largest value that still reached 100%.
PACKING_ALPHA = 0.72


def spacing_field(dens: np.ndarray, floor: float,
                  alpha: float = PACKING_ALPHA) -> np.ndarray:
    """Per-pixel minimum spacing in points, derived from local density.

    For a locally uniform point set of density d, spacing goes as 1/sqrt(d),
    scaled by `alpha` for packing inefficiency. Used as a per-point Poisson-disk
    radius, this grades spacing with tone. One constant separation everywhere
    leaves light regions clumpy and dark regions short of the floor.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(dens > 0, alpha / np.sqrt(np.maximum(dens, 1e-12)), np.inf)
    return np.maximum(r, floor).astype(np.float32)
