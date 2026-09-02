"""Image loading, normalisation and tonal preparation.

Everything downstream consumes a float32 array in [0, 1] where 0 is black.
This module is the only place that knows about file formats, bit depths,
colour spaces or alpha.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter

# Rec.709 luma weights, applied to sRGB-encoded values (the usual
# "luminosity" convention in image editors).
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# Pillow >= 9.1 moved the resampling enum; support both.
_RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS


def _hex_to_rgb01(color: str) -> np.ndarray:
    s = color.lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        raise ValueError(f"bad hex colour: {color!r}")
    return np.array([int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)


def load_rgb(path: str, paper: str = "#ffffff") -> np.ndarray:
    """Load any image as float32 RGB in [0, 1], honouring EXIF orientation.

    Transparency is composited onto `paper`, not discarded. Dropping the alpha
    channel leaves transparent regions holding whatever RGB sits underneath,
    which then gets stippled as solid.
    """
    img = Image.open(path)

    # EXIF orientation: phone and camera files are frequently rotated.
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    # Palette images can carry transparency; go via RGBA to keep it.
    if img.mode == "P":
        img = img.convert("RGBA" if "transparency" in img.info else "RGB")

    # High bit depth integer modes normalise by their own maximum.
    if img.mode in ("I", "I;16", "I;16B", "I;16L", "I;16N"):
        arr = np.asarray(img).astype(np.float32)
        peak = 65535.0 if arr.max() > 255 else 255.0
        g = np.clip(arr / peak, 0.0, 1.0)
        return np.repeat(g[:, :, None], 3, axis=2)

    if img.mode == "F":
        arr = np.asarray(img, dtype=np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        g = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
        return np.repeat(g[:, :, None], 3, axis=2)

    has_alpha = img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info
    img = img.convert("RGBA" if has_alpha else "RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0

    if arr.shape[2] == 4:
        rgb, a = arr[:, :, :3], arr[:, :, 3:4]
        arr = rgb * a + _hex_to_rgb01(paper)[None, None, :] * (1.0 - a)

    return np.ascontiguousarray(arr[:, :, :3])


def to_luma(rgb: np.ndarray) -> np.ndarray:
    """RGB in [0,1] -> single-channel brightness in [0,1]."""
    return np.clip(rgb @ _LUMA, 0.0, 1.0)


def prepare(
    luma: np.ndarray,
    *,
    invert: bool = False,
    pre_blur: float = 0.0,
    black_point: float = 0.0,
    white_point: float = 1.0,
    contrast: float = 0.0,
    gamma: float = 2.2,
) -> np.ndarray:
    """Brightness -> darkness in [0,1], where 1 means "as dark as it gets".

    Blur first, so levels and gamma operate on clean values. Otherwise sensor
    noise is amplified into the density field.
    """
    v = np.asarray(luma, dtype=np.float32)

    if invert:
        v = 1.0 - v

    if pre_blur > 0:
        v = gaussian_filter(v, sigma=float(pre_blur), mode="nearest")

    # Levels
    span = max(white_point - black_point, 1e-6)
    v = np.clip((v - black_point) / span, 0.0, 1.0)

    # Contrast about mid grey. k runs 0..inf, symmetric about c=0.
    if contrast != 0.0:
        c = float(np.clip(contrast, -0.999, 0.999))
        k = (1.0 + c) / (1.0 - c)
        v = np.clip((v - 0.5) * k + 0.5, 0.0, 1.0)

    # Darkness curve (matches the original script's convention).
    return np.clip(1.0 - np.power(v, float(gamma)), 0.0, 1.0)


def downscale(arr: np.ndarray, max_edge: int) -> np.ndarray:
    """Reduce an array so its longest edge is at most `max_edge` pixels."""
    h, w = arr.shape[:2]
    if max(h, w) <= max_edge:
        return arr
    scale = max_edge / float(max(h, w))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))

    if arr.ndim == 2:
        im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), "L")
        out = np.asarray(im.resize((nw, nh), _RESAMPLE), dtype=np.float32) / 255.0
    else:
        im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), "RGB")
        out = np.asarray(im.resize((nw, nh), _RESAMPLE), dtype=np.float32) / 255.0
    return out


def histogram(luma: np.ndarray, bins: int = 128) -> list[int]:
    """Counts for the GUI's levels display."""
    counts, _ = np.histogram(np.clip(luma, 0.0, 1.0), bins=bins, range=(0.0, 1.0))
    return counts.astype(int).tolist()
