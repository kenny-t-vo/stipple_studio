"""Separable filters, using scipy when it is installed.

The pipeline takes two functions from scipy.ndimage. The numpy versions here
match them, so the browser build can leave scipy out of the Pyodide payload
and still produce the same output.
"""

from __future__ import annotations

import numpy as np

__all__ = ["gaussian_filter", "sobel", "HAVE_SCIPY"]


def _correlate1d(arr: np.ndarray, weights, axis: int) -> np.ndarray:
    """One separable correlation pass with edge replication.

    scipy.ndimage.correlate1d at origin 0: the tap at index j reads
    input[i + j - len(w)//2]. Accumulates in float64 and casts back, as the C
    version does.
    """
    w = np.asarray(weights, dtype=np.float64)
    r = len(w) // 2
    pad = [(0, 0)] * arr.ndim
    pad[axis] = (r, r)
    padded = np.pad(arr.astype(np.float64, copy=False), pad, mode="edge")

    n = arr.shape[axis]
    out = np.zeros(arr.shape, dtype=np.float64)
    sl: list = [slice(None)] * arr.ndim
    for j, wj in enumerate(w):
        if wj == 0.0:
            continue
        sl[axis] = slice(j, j + n)
        out += wj * padded[tuple(sl)]
    return out.astype(arr.dtype, copy=False)


def _gaussian_kernel1d(sigma: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 / (sigma * sigma) * x * x)
    return k / k.sum()


def _gaussian_filter_np(input, sigma, mode: str = "nearest",
                        truncate: float = 4.0) -> np.ndarray:
    """scipy.ndimage.gaussian_filter, mode='nearest' only."""
    if mode != "nearest":
        raise ValueError(f"only mode='nearest' is implemented, got {mode!r}")
    arr = np.asarray(input)
    sd = float(sigma)
    radius = int(truncate * sd + 0.5)   # scipy's rounding, not ceil
    if radius < 1:
        return arr.copy()
    k = _gaussian_kernel1d(sd, radius)
    for axis in range(arr.ndim):
        arr = _correlate1d(arr, k, axis)
    return arr


def _sobel_np(input, axis: int = -1, mode: str = "nearest") -> np.ndarray:
    """scipy.ndimage.sobel, mode='nearest' only."""
    if mode != "nearest":
        raise ValueError(f"only mode='nearest' is implemented, got {mode!r}")
    arr = np.asarray(input)
    axis %= arr.ndim
    out = _correlate1d(arr, [-1.0, 0.0, 1.0], axis)
    for other in range(arr.ndim):
        if other != axis:
            out = _correlate1d(out, [1.0, 2.0, 1.0], other)
    return out


try:
    from scipy.ndimage import gaussian_filter, sobel   # noqa: F401
    HAVE_SCIPY = True
except ImportError:                                    # pragma: no cover
    gaussian_filter = _gaussian_filter_np
    sobel = _sobel_np
    HAVE_SCIPY = False
