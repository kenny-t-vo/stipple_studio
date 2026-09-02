"""Tests for the stipple pipeline."""

from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stipple.core import build, generate
from stipple.density import density_field, geometry_for, spacing_field
from stipple.image import load_rgb, prepare, to_luma
from stipple.params import Params
from stipple.render import boustrophedon


# ── fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def gradient(tmp_path):
    """A horizontal black-to-white gradient."""
    w, h = 200, 160
    a = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    p = tmp_path / "gradient.png"
    Image.fromarray(a, "L").convert("RGB").save(p)
    return str(p)


@pytest.fixture
def base(gradient, tmp_path):
    return Params(in_path=gradient, out_path=str(tmp_path / "out.svg"),
                  canvas_w_in=4.0, max_density=0.05, relax_iterations=3)


# ── image handling ───────────────────────────────────────────────────

def test_alpha_is_composited_onto_paper_not_dropped(tmp_path):
    """The original bug: .convert('RGB') discarded alpha, so fully
    transparent pixels kept whatever RGB sat underneath and were stippled
    as if solid."""
    rgba = np.zeros((10, 10, 4), dtype=np.uint8)
    rgba[..., :3] = 0        # black underneath
    rgba[..., 3] = 0         # ...but fully transparent
    p = tmp_path / "clear.png"
    Image.fromarray(rgba, "RGBA").save(p)

    rgb = load_rgb(str(p), paper="#ffffff")
    assert np.allclose(rgb, 1.0), "transparent pixels must read as paper, not black"


def test_partial_alpha_blends(tmp_path):
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[..., 3] = 128
    p = tmp_path / "half.png"
    Image.fromarray(rgba, "RGBA").save(p)
    rgb = load_rgb(str(p), paper="#ffffff")
    assert 0.4 < float(rgb.mean()) < 0.6


def test_grayscale_and_16bit_load(tmp_path):
    g = tmp_path / "g.png"
    Image.fromarray(np.full((8, 8), 128, np.uint8), "L").save(g)
    assert load_rgb(str(g)).shape == (8, 8, 3)

    i16 = tmp_path / "i.png"
    Image.fromarray(np.full((8, 8), 30000, np.uint16)).save(i16)
    out = load_rgb(str(i16))
    assert out.shape == (8, 8, 3) and 0.0 <= out.min() <= out.max() <= 1.0


def test_invert_and_gamma_move_darkness_the_right_way():
    luma = np.array([[0.2, 0.8]], dtype=np.float32)
    d = prepare(luma, gamma=1.0)
    assert d[0, 0] > d[0, 1], "darker input must yield higher darkness"
    inv = prepare(luma, gamma=1.0, invert=True)
    assert inv[0, 0] < inv[0, 1]


def test_levels_clip_and_expand():
    luma = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    d = prepare(luma, gamma=1.0, black_point=0.25, white_point=0.75)
    assert d[0, 0] == pytest.approx(1.0)
    assert d[0, 2] == pytest.approx(0.0)


# ── geometry ─────────────────────────────────────────────────────────

def test_locked_aspect_matches_source():
    p = Params(canvas_w_in=10.0, lock_aspect=True)
    g = geometry_for(p, src_h=200, src_w=100)
    assert g.canvas_h_in == pytest.approx(20.0)
    assert not g.has_margins


def test_unlocked_aspect_fits_and_centres():
    p = Params(canvas_w_in=10.0, canvas_h_in=10.0, lock_aspect=False)
    g = geometry_for(p, src_h=200, src_w=100)      # tall source, square canvas
    assert g.img_h_pt == pytest.approx(g.canvas_h_pt)          # fitted, not cropped
    assert g.img_x_pt == pytest.approx((g.canvas_w_pt - g.img_w_pt) / 2)
    assert g.img_y_pt == pytest.approx(0.0)
    assert g.has_margins


def test_spacing_grades_with_density():
    dens = np.array([[0.01, 0.25]], dtype=np.float32)
    sp = spacing_field(dens, floor=0.1)
    assert sp[0, 0] > sp[0, 1], "sparse regions must get wider spacing"


def test_threshold_blanks_light_regions():
    dark = np.array([[0.1, 0.9]], dtype=np.float32)
    d = density_field(dark, Params(threshold=0.5, min_density=0.01, max_density=0.2))
    assert d[0, 0] == 0.0 and d[0, 1] > 0.0


# ── samplers ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("sampler", ["relaxed", "poisson", "classic"])
def test_sampler_stays_inside_the_image_rect(base, sampler):
    base.sampler = sampler
    r = build(base)
    g = r.geom
    assert len(r.points) > 100
    assert r.points[:, 0].min() >= g.img_x_pt - 1e-6
    assert r.points[:, 0].max() <= g.img_x_pt + g.img_w_pt + 1e-6
    assert r.points[:, 1].min() >= g.img_y_pt - 1e-6
    assert r.points[:, 1].max() <= g.img_y_pt + g.img_h_pt + 1e-6


@pytest.mark.parametrize("sampler", ["relaxed", "poisson", "classic"])
def test_sampler_is_deterministic(base, sampler):
    base.sampler = sampler
    a, b = build(base).points, build(base).points
    assert np.array_equal(a, b)


def test_different_seeds_differ(base):
    a = build(base).points
    base.seed += 1
    assert not np.array_equal(a, build(base).points)


def test_relaxation_improves_spacing_uniformity(base):
    """The whole point of the rewrite."""
    base.sampler = "classic"
    classic = build(base).points
    base.sampler = "relaxed"
    relaxed = build(base).points

    from scipy.spatial import cKDTree

    def cov(p):
        d, _ = cKDTree(p).query(p, k=2)
        nn = d[:, 1]
        return nn.std() / nn.mean()

    assert cov(relaxed) < cov(classic) * 0.75


def test_relaxation_does_not_leak_into_blank_regions(base):
    """Repulsion must not push dots across the threshold boundary."""
    base.threshold = 0.5
    base.relax_iterations = 25
    r = build(base)
    luma = to_luma(load_rgb(base.in_path))
    dark = prepare(luma, gamma=base.gamma)
    dens = density_field(dark, base)
    h, w = dens.shape
    g = r.geom
    ix = np.clip(((r.points[:, 0] - g.img_x_pt) / g.img_w_pt * w).astype(int), 0, w - 1)
    iy = np.clip(((r.points[:, 1] - g.img_y_pt) / g.img_h_pt * h).astype(int), 0, h - 1)
    assert float((dens[iy, ix] <= 0).mean()) < 0.01


def test_density_tracks_tone(base):
    """A left-to-right gradient must produce a left-to-right dot gradient."""
    base.threshold = 0.0
    r = build(base)
    mid = r.geom.img_x_pt + r.geom.img_w_pt / 2
    left = int((r.points[:, 0] < mid).sum())
    right = len(r.points) - left
    assert left > right * 2, "dark half must carry far more dots"


# ── output ───────────────────────────────────────────────────────────

def test_svg_is_wellformed_and_sized(base):
    generate(base)
    root = ET.parse(base.out_path).getroot()
    assert root.tag.endswith("svg")
    assert root.get("width", "").endswith("in")
    assert len(root.get("viewBox", "").split()) == 4


def test_compound_path_is_a_single_element(base):
    base.svg_structure = "compound"
    generate(base)
    root = ET.parse(base.out_path).getroot()
    ns = "{http://www.w3.org/2000/svg}"
    assert len(root.findall(f"{ns}path")) == 1
    assert root.findall(f"{ns}circle") == []


def test_circles_structure_emits_one_element_each(base):
    base.svg_structure = "circles"
    r = generate(base)
    root = ET.parse(base.out_path).getroot()
    ns = "{http://www.w3.org/2000/svg}"
    assert len(root.find(f"{ns}g").findall(f"{ns}circle")) == r.count


def test_svgz_round_trips(base, tmp_path):
    import gzip
    base.out_path = str(tmp_path / "out.svgz")
    generate(base)
    with gzip.open(base.out_path, "rb") as f:
        assert ET.fromstring(f.read().decode()).tag.endswith("svg")


def test_svgz_is_much_smaller(base, tmp_path):
    plain = generate(base).stats["bytes"]
    base.out_path = str(tmp_path / "out.svgz")
    assert generate(base).stats["bytes"] < plain / 4


def test_source_colour_groups_into_paths(base, tmp_path):
    base.color_mode = "source"
    base.out_path = str(tmp_path / "colour.svg")
    generate(base)
    root = ET.parse(base.out_path).getroot()
    ns = "{http://www.w3.org/2000/svg}"
    paths = root.find(f"{ns}g").findall(f"{ns}path")
    assert 1 <= len(paths) <= 216
    assert all(p.get("fill", "").startswith("#") for p in paths)


def test_png_proof_written(base, tmp_path):
    r = generate(base, png_dpi=72)
    assert Path(r.stats["png_path"]).exists()


def test_boustrophedon_shortens_travel(base):
    pts = build(base).points
    order = boustrophedon(pts, band_pt=2.0)
    def travel(p):
        return float(np.abs(np.diff(p, axis=0)).sum())
    assert travel(pts[order]) < travel(pts) * 0.5


# ── params and presets ───────────────────────────────────────────────

def test_preset_round_trip(tmp_path, base):
    base.gamma, base.sampler, base.ink = 3.3, "poisson", "#123456"
    f = tmp_path / "p.json"
    base.save_preset(f, name="test")
    loaded = Params.load_preset(f)
    assert (loaded.gamma, loaded.sampler, loaded.ink) == (3.3, "poisson", "#123456")


def test_preset_does_not_carry_job_paths(tmp_path, base):
    f = tmp_path / "p.json"
    base.save_preset(f)
    assert Params.load_preset(f).in_path == ""


@pytest.mark.parametrize("kw", [
    {"sampler": "nope"}, {"color_mode": "nope"}, {"gamma": 0},
    {"canvas_w_in": 0}, {"black_point": 0.9, "white_point": 0.2},
    {"contrast": 5}, {"min_density": 1.0, "max_density": 0.5},
])
def test_validation_rejects_bad_params(kw):
    with pytest.raises(ValueError):
        Params(**kw).validate()


# ── CLI ──────────────────────────────────────────────────────────────

def _cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "stipple.cli", *args],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True, text=True,
    )


def test_cli_renders(gradient, tmp_path):
    out = tmp_path / "cli.svg"
    r = _cli("render", gradient, str(out), "--width-in", "3", "--max-density", "0.03")
    assert r.returncode == 0, r.stderr
    assert out.exists()


def test_cli_boolean_flags_can_be_switched_off(gradient, tmp_path):
    """The original CLI paired store_true with a config-supplied default, so a
    flag defaulting to True was impossible to disable."""
    out = tmp_path / "cli.svg"
    r = _cli("render", gradient, str(out), "--no-lock-aspect",
             "--width-in", "6", "--height-in", "2", "--max-density", "0.03")
    assert r.returncode == 0, r.stderr
    assert "aspect unlocked" in r.stdout


def test_cli_batch(gradient, tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    for i in range(3):
        Image.open(gradient).save(folder / f"img{i}.png")
    out = tmp_path / "out"
    r = _cli("batch", str(folder), "--out-dir", str(out),
             "--width-in", "2", "--max-density", "0.03")
    assert r.returncode == 0, r.stderr
    assert len(list(out.glob("*.svg"))) == 3


def test_cli_rejects_bad_sampler(gradient, tmp_path):
    assert _cli("render", gradient, str(tmp_path / "x.svg"),
                "--sampler", "bogus").returncode != 0


# ── flow field ───────────────────────────────────────────────────────

from stipple.flow import build_field  # noqa: E402
from stipple.strokes import build_strokes  # noqa: E402
from stipple.density import Geometry  # noqa: E402


def _edge(angle_deg, n=80):
    """A step edge through the centre whose tangent is at `angle_deg`.

    The boundary runs along (cos a, sin a), so its normal is (-sin a, cos a)
    and the tangent the structure tensor recovers should be `angle_deg`.
    """
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    a = np.deg2rad(angle_deg)
    d = (xx - n / 2) * (-np.sin(a)) + (yy - n / 2) * np.cos(a)
    return (d > 0).astype(np.float32)


def _stripes(angle_deg, n=80, period=8):
    """Repeating bands with tangent at `angle_deg`."""
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    a = np.deg2rad(angle_deg)
    d = xx * (-np.sin(a)) + yy * np.cos(a)
    return ((d % period) < period / 2).astype(np.float32)


@pytest.mark.parametrize("deg", [0, 30, 60, 90, 135])
def test_tangent_runs_along_the_edge_not_across_it(deg):
    th, coh = build_field(_edge(deg), smoothing=3.0, diffusion=0, bias_strength=0.0)
    inner = th[30:50, 30:50]
    got = np.rad2deg(np.arctan2(np.sin(2 * inner).mean(), np.cos(2 * inner).mean()) / 2) % 180
    assert min(abs(got - deg), 180 - abs(got - deg)) < 8


def test_coherence_high_on_structure_low_on_flat():
    _, coh_edge = build_field(_edge(45), smoothing=3.0, diffusion=0, bias_strength=0.0)
    _, coh_flat = build_field(np.full((60, 60), 0.5, np.float32),
                              smoothing=3.0, diffusion=0, bias_strength=0.0)
    assert coh_edge[30:50, 30:50].mean() > 0.8
    assert coh_flat.mean() < 0.05


def test_flat_regions_fall_back_to_the_bias_angle():
    flat = np.full((60, 60), 0.5, np.float32)
    th, _ = build_field(flat, smoothing=3.0, diffusion=2,
                        bias_angle_deg=30.0, bias_strength=1.0)
    assert abs((np.rad2deg(th.mean()) % 180) - 30.0) < 1.0


def test_diffusion_carries_direction_into_a_blank_region():
    """Half structured, half featureless: the blank half must inherit.

    A flat neighbourhood gives arctan2(0, 0), a constant -- so the blank half
    already looks perfectly self-consistent without any diffusion. The test
    that means something is whether it agrees with the structured half.
    """
    ANGLE = 30.0
    a = np.full((80, 80), 0.5, np.float32)
    a[:, :35] = _stripes(ANGLE, 80)[:, :35]

    blank = (slice(20, 60), slice(58, 76))

    def error_vs_source(field):
        s = field[blank]
        got = np.rad2deg(np.arctan2(np.sin(2 * s).mean(),
                                    np.cos(2 * s).mean()) / 2) % 180
        return min(abs(got - ANGLE), 180 - abs(got - ANGLE))

    no_diff, _ = build_field(a, smoothing=3.0, diffusion=0, bias_strength=0.0)
    diff, _ = build_field(a, smoothing=3.0, diffusion=10, bias_strength=0.0)

    assert error_vs_source(no_diff) > 30.0, "blank half should start off wrong"
    assert error_vs_source(diff) < error_vs_source(no_diff) / 2


def test_perpendicular_rotates_by_ninety_degrees():
    a = _edge(0)
    th1, _ = build_field(a, smoothing=3.0, diffusion=0, bias_strength=0.0)
    th2, _ = build_field(a, smoothing=3.0, diffusion=0, bias_strength=0.0,
                         perpendicular=True)
    d = np.rad2deg(np.abs(th2 - th1)).mean() % 180
    assert abs(d - 90) < 1e-3


# ── strokes ──────────────────────────────────────────────────────────

def _flat_field(angle_deg, n=100):
    return np.full((n, n), np.deg2rad(angle_deg), np.float32)


def test_strokes_follow_the_field():
    g = Geometry(200, 200, 0, 0, 200, 200)
    pts = np.array([[100.0, 100.0], [60.0, 40.0]])
    st = build_strokes(pts, np.ones(2), _flat_field(30), g,
                       base_r=1.0, spacing=np.full((100, 100), 4.0, np.float32),
                       length_factor=4.0, jitter=0.0, seed=1)
    v = st.p2 - st.p0
    ang = np.rad2deg(np.arctan2(v[:, 1], v[:, 0])) % 180
    assert np.allclose(ang, 30.0, atol=2.0)


def test_stroke_length_scales_with_darkness():
    g = Geometry(200, 200, 0, 0, 200, 200)
    pts = np.array([[100.0, 100.0], [100.0, 100.0]])
    st = build_strokes(pts, np.array([1.0, 0.25]), _flat_field(0), g,
                       base_r=1.0, spacing=np.full((100, 100), 4.0, np.float32),
                       length_factor=4.0, jitter=0.0, seed=7)
    ln = np.hypot(*(st.p2 - st.p0).T)
    assert ln[0] > ln[1] * 2.0


def test_zero_darkness_gives_a_degenerate_stroke():
    g = Geometry(200, 200, 0, 0, 200, 200)
    st = build_strokes(np.array([[50.0, 50.0]]), np.zeros(1), _flat_field(0), g,
                       base_r=1.0, spacing=np.full((100, 100), 4.0, np.float32),
                       length_factor=4.0, jitter=0.0, seed=1)
    assert float(np.hypot(*(st.p2 - st.p0)[0])) < 1e-6


def test_seed_point_is_the_curve_midpoint():
    """Strokes grow outward from the seed, so blue-noise spacing still governs."""
    g = Geometry(200, 200, 0, 0, 200, 200)
    pts = np.array([[100.0, 100.0], [70.0, 130.0]])
    st = build_strokes(pts, np.ones(2), _flat_field(45), g,
                       base_r=1.0, spacing=np.full((100, 100), 4.0, np.float32),
                       length_factor=4.0, jitter=0.0, seed=3)
    assert np.allclose((st.p0 + 2 * st.ctrl + st.p2) / 4, pts)


def test_strokes_are_deterministic(base):
    base.line_mode = True
    a, b = build(base).strokes, build(base).strokes
    assert np.array_equal(a.p0, b.p0) and np.array_equal(a.p2, b.p2)


# ── stroke output ────────────────────────────────────────────────────

def test_line_mode_svg_is_wellformed(base):
    base.line_mode = True
    generate(base)
    root = ET.parse(base.out_path).getroot()
    ns = "{http://www.w3.org/2000/svg}"
    paths = root.find(f"{ns}g").findall(f"{ns}path")
    assert len(paths) >= 1
    assert all("q" in p.get("d", "") for p in paths)


def test_taper_fills_and_plain_strokes(base):
    ns = "{http://www.w3.org/2000/svg}"
    base.line_mode = True

    base.line_taper = True
    generate(base)
    p = ET.parse(base.out_path).getroot().find(f"{ns}g").find(f"{ns}path")
    assert p.get("fill") == "#000000" and p.get("stroke") == "none"

    base.line_taper = False
    generate(base)
    p = ET.parse(base.out_path).getroot().find(f"{ns}g").find(f"{ns}path")
    assert p.get("fill") == "none" and p.get("stroke-linecap") == "round"


def test_line_mode_png_proof(base, tmp_path):
    base.line_mode = True
    r = generate(base, png_dpi=72)
    assert Path(r.stats["png_path"]).exists()


def test_cli_line_mode(gradient, tmp_path):
    out = tmp_path / "lines.svg"
    r = _cli("render", gradient, str(out), "--line-mode", "--width-in", "3",
             "--max-density", "0.03")
    assert r.returncode == 0, r.stderr
    assert out.exists()
