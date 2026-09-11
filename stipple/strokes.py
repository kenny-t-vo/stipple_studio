"""Turn seed points into short curved strokes that follow the flow field."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .density import Geometry


@dataclass
class Strokes:
    """Quadratic Beziers: start, control, end, plus a width for each."""
    p0: np.ndarray      # (N, 2)
    ctrl: np.ndarray    # (N, 2)
    p2: np.ndarray      # (N, 2)
    width: np.ndarray   # (N,)

    def __len__(self) -> int:
        return len(self.p0)


def _dir_at(theta: np.ndarray, x: np.ndarray, y: np.ndarray, geom: Geometry):
    h, w = theta.shape
    ix = np.clip(((x - geom.img_x_pt) / geom.img_w_pt * w).astype(np.int32), 0, w - 1)
    iy = np.clip(((y - geom.img_y_pt) / geom.img_h_pt * h).astype(np.int32), 0, h - 1)
    t = theta[iy, ix]
    return np.cos(t), np.sin(t)


def _align(dx, dy, rx, ry):
    """Flip direction vectors that oppose a reference.

    The field stores an orientation, not a heading: theta and theta+pi are the
    same line. Integrating without fixing the sign at every step makes
    streamlines reverse on themselves and strokes collapse.
    """
    flip = (dx * rx + dy * ry) < 0.0
    return np.where(flip, -dx, dx), np.where(flip, -dy, dy)


def _walk(x0, y0, dx0, dy0, step, steps, theta, geom, sign):
    """Integrate a streamline from each seed, midpoint method, in lockstep."""
    x, y = x0.copy(), y0.copy()
    rx, ry = sign * dx0, sign * dy0

    for k in range(steps):
        d1x, d1y = _align(*_dir_at(theta, x, y, geom), rx, ry)
        hx, hy = x + 0.5 * step * d1x, y + 0.5 * step * d1y
        d2x, d2y = _align(*_dir_at(theta, hx, hy, geom), d1x, d1y)
        x = x + step * d2x
        y = y + step * d2y
        rx, ry = d2x, d2y

    return x, y


def build_strokes(
    pts: np.ndarray,
    dark: np.ndarray,
    theta: np.ndarray,
    geom: Geometry,
    *,
    base_r: float,
    spacing: np.ndarray,
    length_factor: float,
    jitter: float,
    seed: int,
    steps: int = 4,
) -> Strokes:
    """Grow a curved stroke through each seed point, along the flow field.

    Each stroke is integrated outward in both directions from its seed, so the
    seed stays at the centre and the point set's blue-noise spacing still
    governs the texture. Length scales with local darkness and per-stroke noise:
    dark regions give long strokes, light regions give marks barely longer than
    dots.
    """
    n = len(pts)
    if n == 0:
        empty = np.zeros((0, 2))
        return Strokes(empty, empty, empty, np.zeros(0))

    rng = np.random.default_rng(seed ^ 0x5713)
    x0, y0 = pts[:, 0], pts[:, 1]

    dx, dy = _dir_at(theta, x0, y0, geom)
    if jitter > 0:
        a = rng.normal(0.0, float(jitter), n)
        ca, sa = np.cos(a), np.sin(a)
        dx, dy = dx * ca - dy * sa, dx * sa + dy * ca

    # Half-length per stroke: tone, times a per-stroke draw so the field
    # does not look mechanically uniform.
    noise = 0.35 + 0.65 * rng.random(n)
    h_sp, w_sp = spacing.shape
    six = np.clip(((x0 - geom.img_x_pt) / geom.img_w_pt * w_sp).astype(np.int32), 0, w_sp - 1)
    siy = np.clip(((y0 - geom.img_y_pt) / geom.img_h_pt * h_sp).astype(np.int32), 0, h_sp - 1)
    local = spacing[siy, six]
    local = np.where(np.isfinite(local), local, 0.0)
    half_len = 0.5 * np.maximum(dark, 0.0) * length_factor * local * noise
    step = half_len / max(steps, 1)

    fx, fy = _walk(x0, y0, dx, dy, step, steps, theta, geom, +1.0)
    bx, by = _walk(x0, y0, dx, dy, step, steps, theta, geom, -1.0)

    p0 = np.column_stack([bx, by])
    p2 = np.column_stack([fx, fy])

    # Quadratic control point placed so the curve interpolates the seed,
    # which is the streamline's midpoint: B(0.5) = (P0 + 2C + P2) / 4.
    ctrl = 2.0 * pts - 0.5 * (p0 + p2)

    return Strokes(p0=p0, ctrl=ctrl, p2=p2, width=np.full(n, 2.0 * base_r))
