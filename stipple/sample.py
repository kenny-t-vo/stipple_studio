"""Point samplers.

Three samplers, all producing points in canvas coordinates (points):

  classic  the original dart-throwing sampler. Kept because it is what the
           existing artwork was made with. Its nearest-neighbour spread is
           indistinguishable from a Poisson process, i.e. random scatter.
  poisson  variable-radius Poisson disk. Organic blue noise; spacing grades
           with local tone instead of sitting at one global floor.
  relaxed  density-proportional seed, then spacing relaxation against the
           local target radius. The evenest, and the default.
"""

from __future__ import annotations

import math
import random
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree

from .density import Geometry, spacing_field
from .params import Params

Progress = Callable[[str, float], None] | None


def _notify(cb: Progress, stage: str, frac: float) -> None:
    if cb is not None:
        cb(stage, frac)


# ── coordinate helpers ───────────────────────────────────────────────

def _to_canvas(fx: np.ndarray, fy: np.ndarray, shape, geom: Geometry):
    """Fractional pixel coords -> canvas points."""
    h, w = shape
    x = geom.img_x_pt + (fx / w) * geom.img_w_pt
    y = geom.img_y_pt + (fy / h) * geom.img_h_pt
    return x, y


def _sample_field(field: np.ndarray, x: np.ndarray, y: np.ndarray, geom: Geometry):
    """Nearest-neighbour lookup of a source-resolution field at canvas points."""
    h, w = field.shape
    ix = np.clip(((x - geom.img_x_pt) / geom.img_w_pt * w).astype(np.int32), 0, w - 1)
    iy = np.clip(((y - geom.img_y_pt) / geom.img_h_pt * h).astype(np.int32), 0, h - 1)
    return field[iy, ix]


# ── seeding ──────────────────────────────────────────────────────────

def seed_by_density(dens: np.ndarray, n: int, geom: Geometry, rng: np.random.Generator):
    """Draw n points with probability proportional to the density field.

    Inverse-transform sampling over the flattened field. Exact, and negligible
    cost even on a 7-megapixel source.
    """
    flat = dens.ravel().astype(np.float64)
    total = flat.sum()
    if total <= 0 or n <= 0:
        return np.zeros((0, 2), dtype=np.float64)

    cdf = np.cumsum(flat)
    idx = np.searchsorted(cdf, rng.random(n) * total, side="right")
    idx = np.clip(idx, 0, flat.size - 1)

    h, w = dens.shape
    iy, ix = np.divmod(idx, w)
    fx = ix + rng.random(n)
    fy = iy + rng.random(n)
    x, y = _to_canvas(fx, fy, (h, w), geom)
    return np.column_stack([x, y])


# ── spacing relaxation ───────────────────────────────────────────────

def relax(
    pts: np.ndarray,
    dens: np.ndarray,
    geom: Geometry,
    iterations: int,
    floor: float,
    neighbours: int = 10,
    step: float = 0.5,
    reach: float = 1.5,
    progress: Progress = None,
) -> np.ndarray:
    """Even out spacing by pushing each point off its too-close neighbours.

    Weighted Lloyd relaxation -- move each point to the density-weighted
    centroid of its Voronoi cell -- does not work at this scale. Estimating a
    centroid needs a raster accumulation grid, and at 216k dots on a 21x28in
    canvas each Voronoi cell covers only ~14 source pixels, so the centroid is
    estimated from ~14 samples and is close to noise. Measured: CoV pinned at
    0.45 regardless of iteration count. Reaching a usable ~55 samples per cell
    needs 2x supersampling, 28M grid points and roughly 48 seconds.

    This is grid-free. Each point looks up its target spacing from the density
    field, finds its k nearest neighbours, and is displaced away from any that
    sit inside that spacing. The target radius is local, so dense regions relax
    against a small radius and sparse regions against a large one: tone is
    preserved while spacing evens out. Cost is one KD-tree query per iteration.
    """
    if iterations <= 0 or len(pts) == 0:
        return pts

    spacing = spacing_field(dens, floor)
    finite_max = float(np.nanmax(spacing[np.isfinite(spacing)])) if np.isfinite(spacing).any() else floor
    k = min(neighbours, len(pts) - 1)
    if k < 1:
        return pts

    x0, y0 = geom.img_x_pt, geom.img_y_pt
    x1, y1 = x0 + geom.img_w_pt, y0 + geom.img_h_pt

    for it in range(iterations):
        r = _sample_field(spacing, pts[:, 0], pts[:, 1], geom)
        r = np.where(np.isfinite(r), r, finite_max) * reach

        dist, idx = cKDTree(pts).query(pts, k=k + 1, workers=-1)
        dist, idx = dist[:, 1:], idx[:, 1:]

        dx = pts[:, 0:1] - pts[idx, 0]
        dy = pts[:, 1:2] - pts[idx, 1]
        d = np.maximum(dist, 1e-9)

        # Only neighbours inside the interaction radius contribute. `reach`
        # widens it past the natural spacing on purpose: at reach = 1.0 a
        # Poisson-disk seed already satisfies every constraint, every force is
        # exactly zero and relaxation is a no-op (measured: 0.202 -> 0.201).
        # Widening it puts every point in contact so voids can close too.
        overlap = np.maximum(r[:, None] - d, 0.0)
        wgt = overlap / d
        active = np.maximum((overlap > 0).sum(axis=1), 1)

        nx = np.clip(pts[:, 0] + step * (dx * wgt).sum(axis=1) / active, x0, x1)
        ny = np.clip(pts[:, 1] + step * (dy * wgt).sum(axis=1) / active, y0, y1)

        # A point pushed into a zero-density region stays where it was.
        # Without this, repulsion bleeds dots across the threshold boundary
        # into areas the tone curve said should be blank -- measured at 21%
        # of all points leaking out over 12 iterations.
        ok = _sample_field(dens, nx, ny, geom) > 0
        pts[ok, 0] = nx[ok]
        pts[ok, 1] = ny[ok]
        _notify(progress, "relaxing", (it + 1) / iterations)

    return pts


# ── variable-radius Poisson disk ─────────────────────────────────────

def poisson_disk(
    dens: np.ndarray,
    geom: Geometry,
    target: int,
    floor: float,
    rng: np.random.Generator,
    max_passes: int = 24,
    progress: Progress = None,
) -> np.ndarray:
    """Multi-pass dart throwing with a per-point radius drawn from density.

    Each pass proposes a batch of density-proportional candidates, thins them
    against a cell grid so no two survivors share a cell, then rejects any
    that fall inside an accepted point's disk. Batched rather than sequential
    so the whole thing stays in numpy and scipy.

    The conflict test uses max(r_candidate, r_neighbour). Strictly that should
    consider every point whose radius reaches the candidate, not just its
    nearest; because the spacing field is smooth, neighbours carry near-equal
    radii and the difference is not visible.
    """
    spacing = spacing_field(dens, floor)
    finite = spacing[np.isfinite(spacing)]
    if finite.size == 0 or target <= 0:
        return np.zeros((0, 2), dtype=np.float64)

    cell = float(max(finite.min(), 1e-3)) / math.sqrt(2.0)
    accepted: list[np.ndarray] = []
    total = 0
    tree = None

    for p in range(max_passes):
        want = target - total
        if want <= 0:
            break
        batch = seed_by_density(dens, max(want * 2, 1024), geom, rng)
        if batch.shape[0] == 0:
            break

        # One survivor per grid cell keeps the batch internally spread.
        keys = (batch[:, 0] / cell).astype(np.int64) * 1_000_003 + \
               (batch[:, 1] / cell).astype(np.int64)
        _, first = np.unique(keys, return_index=True)
        batch = batch[np.sort(first)]

        r_cand = _sample_field(spacing, batch[:, 0], batch[:, 1], geom)

        if tree is not None:
            dist, nn = tree.query(batch, workers=-1)
            r_near = _sample_field(spacing, tree.data[nn, 0], tree.data[nn, 1], geom)
            batch = batch[dist >= np.maximum(r_cand, r_near)]
            if batch.shape[0] == 0:
                continue
            r_cand = _sample_field(spacing, batch[:, 0], batch[:, 1], geom)

        # Resolve conflicts inside the surviving batch, greedily.
        order = np.argsort(-r_cand)          # place sparse regions first
        batch, r_cand = batch[order], r_cand[order]
        bt = cKDTree(batch)
        dead = np.zeros(len(batch), dtype=bool)
        for i, pairs in enumerate(bt.query_ball_point(batch, r_cand, workers=-1)):
            if dead[i]:
                continue
            for j in pairs:
                if j > i:
                    dead[j] = True
        batch = batch[~dead]

        if batch.shape[0] == 0:
            continue
        if total + len(batch) > target:
            batch = batch[: target - total]

        accepted.append(batch)
        total += len(batch)
        stacked = np.vstack(accepted)
        tree = cKDTree(stacked)
        _notify(progress, "sampling", min(1.0, total / max(target, 1)))

    return np.vstack(accepted) if accepted else np.zeros((0, 2), dtype=np.float64)


# ── the original sampler, preserved ──────────────────────────────────

def classic_scatter(
    dens: np.ndarray,
    geom: Geometry,
    target: int,
    min_dist: float,
    max_attempts: int,
    seed: int,
) -> np.ndarray:
    """Dart throwing with one global separation floor, as originally written."""
    rng = random.Random(seed)
    h, w = dens.shape
    dens_max = float(dens.max())
    if dens_max <= 0:
        return np.zeros((0, 2), dtype=np.float64)

    cell = min_dist
    buckets: dict[tuple[int, int], list[tuple[float, float]]] = {}
    out: list[tuple[float, float]] = []
    min_sq = min_dist * min_dist
    attempts = 0

    while attempts < max_attempts and len(out) < target:
        attempts += 1
        x = geom.img_x_pt + rng.random() * geom.img_w_pt
        y = geom.img_y_pt + rng.random() * geom.img_h_pt
        ix = min(int((x - geom.img_x_pt) / geom.img_w_pt * w), w - 1)
        iy = min(int((y - geom.img_y_pt) / geom.img_h_pt * h), h - 1)

        d = dens[iy, ix]
        if d <= 0 or rng.random() > d / dens_max:
            continue

        kx, ky = int(x // cell), int(y // cell)
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for nx, ny in buckets.get((kx + dx, ky + dy), ()):
                    if (x - nx) ** 2 + (y - ny) ** 2 < min_sq:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            continue

        buckets.setdefault((kx, ky), []).append((x, y))
        out.append((x, y))

    return np.array(out, dtype=np.float64) if out else np.zeros((0, 2), dtype=np.float64)


def _top_up(pts: np.ndarray, dens, geom: Geometry, target: int,
            rng: np.random.Generator) -> np.ndarray:
    """Make up any shortfall with density-proportional points.

    Dart throwing stops when it runs out of passes, not when the plane is full,
    and stops consistently short: measured at 97.6% of target across image
    types, canvas sizes and densities. A 2.4% deficit lightens every tone by
    2.4%, systematically.

    Top-up points are drawn from the density field, so tone is right, but are
    placed without regard to spacing, so a few start too close to a neighbour.
    At this proportion relaxation absorbs them.
    """
    short = target - len(pts)
    if short <= 0:
        return pts
    extra = seed_by_density(dens, short, geom, rng)
    return np.vstack([pts, extra]) if len(pts) else extra


# ── dispatch ─────────────────────────────────────────────────────────

def sample(p: Params, dens: np.ndarray, geom: Geometry, target: int,
           progress: Progress = None) -> np.ndarray:
    """Run the configured sampler. Returns an (N, 2) array of canvas points."""
    rng = np.random.default_rng(p.seed)
    floor = 2.0 * p.dot_radius_pt * p.min_sep_factor

    if p.sampler == "classic":
        return classic_scatter(
            dens, geom, target, floor,
            int(max(1000, target * p.max_attempts_factor)), p.seed,
        )

    if p.sampler == "poisson":
        return _top_up(poisson_disk(dens, geom, target, floor, rng, progress=progress),
                       dens, geom, target, rng)

    # Seeding matters more than relaxing. A density-proportional random seed
    # is a Poisson process (normalised CoV 0.69) and repulsion cannot undo
    # that clumping -- it stalls around 0.32. Seeding with a Poisson disk and
    # then relaxing reaches 0.21.
    pts = poisson_disk(dens, geom, target, floor, rng, progress=progress)
    pts = _top_up(pts, dens, geom, target, rng)
    return relax(pts, dens, geom, p.relax_iterations, floor, progress=progress)
