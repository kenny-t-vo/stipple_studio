"""The numpy stand-ins for the parts of scipy the pipeline uses.

scipy is optional so the browser build can leave it out of the Pyodide
payload. These tests hold the two backends to the same output, which is what
lets the desktop and browser builds claim to be the same tool.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stipple import filters, flow, image, spatial
from stipple.core import build
from stipple.params import Params

needs_scipy = pytest.mark.skipif(not filters.HAVE_SCIPY,
                                 reason="nothing to compare against")


@pytest.fixture
def field():
    """Clustered points, so local density varies the way an image makes it."""
    rng = np.random.default_rng(11)
    return np.vstack([rng.random((4000, 2)) * [300.0, 400.0],
                      rng.normal([80.0, 120.0], 6.0, size=(4000, 2))])


# ── filters ──────────────────────────────────────────────────────────

@needs_scipy
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sigma", [0.1, 0.5, 1.0, 2.0, 3.6, 11.66, 25.0])
def test_gaussian_matches_scipy(dtype, sigma):
    from scipy.ndimage import gaussian_filter as reference

    a = np.random.default_rng(0).random((61, 47)).astype(dtype)
    got = filters._gaussian_filter_np(a, sigma, mode="nearest")
    assert got.dtype == a.dtype
    assert np.allclose(got, reference(a, sigma, mode="nearest"), rtol=0, atol=2e-6)


@needs_scipy
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_sobel_matches_scipy(axis):
    from scipy.ndimage import sobel as reference

    a = np.random.default_rng(1).random((53, 39)).astype(np.float32)
    got = filters._sobel_np(a, axis=axis, mode="nearest")
    assert np.allclose(got, reference(a, axis=axis, mode="nearest"), rtol=0, atol=2e-6)


def test_filters_reject_other_edge_modes():
    a = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        filters._gaussian_filter_np(a, 1.0, mode="reflect")
    with pytest.raises(ValueError):
        filters._sobel_np(a, mode="reflect")


# ── neighbour index ──────────────────────────────────────────────────

@needs_scipy
def test_knn_within_matches_the_tree(field):
    radius = np.random.default_rng(2).uniform(1.0, 6.0, size=len(field))
    tree = spatial._KDTreeIndex(field).knn_within(radius, 10)
    grid = spatial._GridIndex(field).knn_within(radius, 10)

    # Padding sits in the same slots, and the neighbours found are the same
    # ones in the same order. Distances differ by at most an ulp.
    finite = np.isfinite(tree[0])
    assert np.array_equal(finite, np.isfinite(grid[0]))
    assert np.array_equal(tree[1][finite], grid[1][finite])
    assert np.allclose(tree[0][finite], grid[0][finite], rtol=0, atol=1e-12)


@needs_scipy
def test_nearest_and_ball_match_the_tree(field):
    rng = np.random.default_rng(3)
    q = rng.random((500, 2)) * [300.0, 400.0]
    tree, grid = spatial._KDTreeIndex(field), spatial._GridIndex(field)

    td, ti = tree.nearest(q)
    gd, gi = grid.nearest(q)
    assert np.array_equal(ti, gi)
    assert np.allclose(td, gd, rtol=0, atol=1e-12)

    radii = rng.uniform(0.5, 9.0, size=len(q))
    for a, b in zip(tree.ball(q, radii), grid.ball(q, radii)):
        assert sorted(int(i) for i in a) == sorted(int(i) for i in b)


@needs_scipy
def test_nearest_handles_queries_outside_the_point_cloud():
    pts = np.random.default_rng(4).random((200, 2)) * 50.0
    q = np.array([[-9e4, -9e4], [9e4, 9e4], [25.0, 25.0]])
    assert np.allclose(spatial._KDTreeIndex(pts).nearest(q)[0],
                       spatial._GridIndex(pts).nearest(q)[0], rtol=0, atol=1e-12)


@pytest.mark.parametrize("pts", [
    np.zeros((0, 2)),
    np.array([[1.0, 1.0]]),
    np.tile([2.0, 3.0], (5, 1)),            # every point in one spot
])
def test_grid_survives_degenerate_point_sets(pts):
    g = spatial._GridIndex(pts)
    assert g.gw >= 1 and g.gh >= 1
    d, _ = g.nearest(np.array([[0.0, 0.0]]))
    assert d.shape == (1,)
    assert len(g.ball(np.array([[0.0, 0.0]]), np.array([1.0]))) == 1


def test_the_factory_follows_the_flag(monkeypatch):
    pts = np.random.default_rng(5).random((32, 2))
    monkeypatch.setattr(spatial, "HAVE_SCIPY", False)
    assert isinstance(spatial.neighbours(pts), spatial._GridIndex)


# ── the whole pipeline ───────────────────────────────────────────────

@needs_scipy
@pytest.mark.parametrize("line_mode", [False, True])
def test_pipeline_agrees_with_scipy_absent(tmp_path, monkeypatch, line_mode):
    """Same marks either way. Only the last ulp of a distance differs, which
    reaches the canvas at 1e-13 points on a 288pt canvas."""
    rng = np.random.default_rng(6)
    a = (rng.random((180, 140)) * 160 + np.linspace(0, 95, 140)).clip(0, 255)
    src = tmp_path / "src.png"
    Image.fromarray(a.astype(np.uint8), "L").convert("RGB").save(src)

    p = Params(in_path=str(src), canvas_w_in=4.0, max_density=0.35,
               relax_iterations=6, pre_blur=1.5, contrast=0.2, line_mode=line_mode)
    with_scipy = build(p)

    monkeypatch.setattr(spatial, "HAVE_SCIPY", False)
    monkeypatch.setattr(image, "gaussian_filter", filters._gaussian_filter_np)
    monkeypatch.setattr(flow, "gaussian_filter", filters._gaussian_filter_np)
    monkeypatch.setattr(flow, "sobel", filters._sobel_np)
    without = build(p)

    assert without.count == with_scipy.count
    assert np.allclose(without.points, with_scipy.points, rtol=0, atol=1e-6)
    assert np.array_equal(without.darkness, with_scipy.darkness)
    if line_mode:
        assert np.allclose(without.strokes.p0, with_scipy.strokes.p0, rtol=0, atol=1e-6)
        assert np.allclose(without.strokes.p2, with_scipy.strokes.p2, rtol=0, atol=1e-6)


@needs_scipy
def test_an_infinite_radius_reaches_every_point(field, recwarn):
    """spacing_field is inf wherever density is zero, and the nearest-pixel
    lookup can land there, so the samplers do pass inf through.

    An inf radius must be turned into a ring width before the cast to int, not
    after. Casting a float out of int64 range is undefined; this numpy
    saturates upward, which lands on the right ring by luck and warns while
    doing it, but WASM traps on the same conversion and numpy emulates it.
    """
    q = field[:20]
    radii = np.full(len(q), np.inf)
    tree = spatial._KDTreeIndex(field).ball(q, radii)
    grid = spatial._GridIndex(field).ball(q, radii)
    for a, b in zip(tree, grid):
        assert len(a) == len(field)
        assert sorted(int(i) for i in a) == sorted(int(i) for i in b)

    r = np.where(np.arange(len(field)) % 500 == 0, np.inf, 3.0)
    td, ti = spatial._KDTreeIndex(field).knn_within(r, 10)
    gd, gi = spatial._GridIndex(field).knn_within(r, 10)
    finite = np.isfinite(td)
    assert np.array_equal(finite, np.isfinite(gd))
    assert np.array_equal(ti[finite], gi[finite])

    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]
