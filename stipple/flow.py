"""Direction field derived from image structure.

Stroke direction comes from a structure tensor. Its minor eigenvector points
along local image structure -- the direction in which the image changes
least. Its eigenvalue gap gives how directional the neighbourhood is, which
matters because large flat areas (sky, smooth walls) have no direction and
would otherwise produce noise.
"""

from __future__ import annotations

import numpy as np

from .filters import gaussian_filter, sobel


def structure_tensor(luma: np.ndarray, sigma: float):
    """Return smoothed tensor components (Jxx, Jxy, Jyy).

    Smoothing the tensor rather than the gradients keeps the field coherent:
    gradient vectors of opposite sign cancel when averaged, the outer products
    do not.
    """
    gx = sobel(luma, axis=1, mode="nearest")
    gy = sobel(luma, axis=0, mode="nearest")
    s = max(float(sigma), 0.1)
    return (
        gaussian_filter(gx * gx, s, mode="nearest"),
        gaussian_filter(gx * gy, s, mode="nearest"),
        gaussian_filter(gy * gy, s, mode="nearest"),
    )


def tangent_and_coherence(jxx, jxy, jyy):
    """Minor-eigenvector angle, and how much to trust it.

    Coherence is (l1 - l2) / (l1 + l2): 1 where the neighbourhood has one
    clear direction, 0 where it is isotropic and the angle is meaningless.
    """
    diff = jxx - jyy
    root = np.sqrt(diff * diff + 4.0 * jxy * jxy)
    l1 = 0.5 * (jxx + jyy + root)
    l2 = 0.5 * (jxx + jyy - root)

    # Minor eigenvector, i.e. along the structure rather than across it.
    theta = 0.5 * np.arctan2(2.0 * jxy, diff) + np.pi / 2.0

    total = l1 + l2
    coh = np.where(total > 1e-12, (l1 - l2) / np.maximum(total, 1e-12), 0.0)
    return theta.astype(np.float32), np.clip(coh, 0.0, 1.0).astype(np.float32)


def diffuse(theta, coh, passes: int, sigma: float = 2.0,
            growth: float = 1.8, max_sigma: float = 64.0):
    """Propagate direction from confident regions into flat ones.

    Angles cannot be averaged directly: they wrap, and stroke direction is
    undirected, so theta and theta+pi are the same. Averaging the doubled-angle
    vector (cos 2t, sin 2t) handles both.

    The blur radius grows by `growth` each pass. At constant sigma the reach
    after n passes is sigma*sqrt(n) -- four passes at sigma 2 travel about six
    pixels, too short to cross a large flat region, which then falls back to the
    bias angle. Growing the radius makes the reach geometric: the same four
    passes carry direction roughly ten times further at about the same cost.
    """
    if passes <= 0:
        return theta

    # Weighted vectors: each pixel contributes in proportion to how much its
    # direction can be trusted. Blurring these and reading the angle back out
    # at the end IS a weighted average -- arctan2 is scale invariant, so no
    # division by the accumulated weight is needed. An earlier version did
    # divide, per pass, guarded by max(w, 1e-6); in the middle of a blank
    # region that divides numerical dust by a millionth, rescales it to unit
    # length, and records it as fully confident. The garbage then propagated,
    # leaving flat areas stranded halfway between the real direction and the
    # arctan2(0, 0) default.
    c2 = np.cos(2.0 * theta) * coh
    s2 = np.sin(2.0 * theta) * coh

    for i in range(int(passes)):
        sig = min(sigma * (growth ** i), max_sigma)
        c2 = gaussian_filter(c2, sig, mode="nearest")
        s2 = gaussian_filter(s2, sig, mode="nearest")

    filled = 0.5 * np.arctan2(s2, c2)

    # Pixels that had a direction of their own keep it; only the uncertain
    # ones take what diffused in.
    cx = coh * np.cos(2.0 * theta) + (1.0 - coh) * np.cos(2.0 * filled)
    cy = coh * np.sin(2.0 * theta) + (1.0 - coh) * np.sin(2.0 * filled)
    return (0.5 * np.arctan2(cy, cx)).astype(np.float32)


def build_field(
    luma: np.ndarray,
    *,
    smoothing: float = 4.0,
    diffusion: int = 4,
    bias_angle_deg: float = 90.0,
    bias_strength: float = 0.5,
    perpendicular: bool = False,
):
    """Full field: (theta, coherence), both at source resolution.

    Directional regions use the tangent. Flat regions fall back toward
    `bias_angle_deg`, mixed in proportion to the missing confidence, so they get
    a consistent grain instead of noise.
    """
    jxx, jxy, jyy = structure_tensor(luma, smoothing)
    theta, coh = tangent_and_coherence(jxx, jxy, jyy)
    theta = diffuse(theta, coh, diffusion)

    if bias_strength > 0:
        bias = np.deg2rad(float(bias_angle_deg))
        # Blend in doubled-angle space, weighted by lack of coherence.
        k = np.clip(bias_strength * (1.0 - coh), 0.0, 1.0)
        cx = (1.0 - k) * np.cos(2 * theta) + k * np.cos(2 * bias)
        cy = (1.0 - k) * np.sin(2 * theta) + k * np.sin(2 * bias)
        theta = (0.5 * np.arctan2(cy, cx)).astype(np.float32)

    if perpendicular:
        theta = theta + np.float32(np.pi / 2.0)

    return theta, coh


def direction_at(theta: np.ndarray, ix: np.ndarray, iy: np.ndarray):
    """Unit vectors for the field sampled at integer pixel indices."""
    t = theta[iy, ix]
    return np.cos(t), np.sin(t)
