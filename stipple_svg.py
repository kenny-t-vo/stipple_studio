#!/usr/bin/env python3
"""
Raster image → stochastic stipple dots → SVG

Generates a stippled (dot-based) representation of a raster image,
output as SVG compatible with Adobe Illustrator and other vector editors.

Usage:
    python3 stipple_svg.py input.png output.svg
    python3 stipple_svg.py --gui
    python3 stipple_svg.py input.png output.svg --combine-paths --scale-with-darkness
"""

import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

try:
    import numpy as np
    HAS_NUMPY = True
except Exception:
    np = None
    HAS_NUMPY = False

try:
    from PIL import Image, ImageOps
except Exception as exc:
    raise SystemExit("Pillow is required: python3 -m pip install pillow") from exc


# ═══════════════════════════════════════════════════════════════════════════
#  DEFAULTS — edit these values to change default behavior
# ═══════════════════════════════════════════════════════════════════════════
DEFAULTS = dict(
    canvas_size_in      = 28.0,     # longest edge of output canvas (inches)
    dot_radius_pt       = 0.25,     # base dot radius (points; 72pt = 1in)
    min_density         = 0.0,      # dots/pt² in lightest areas
    max_density         = 0.225,    # dots/pt² in darkest areas
    gamma               = 4.05,     # brightness gamma curve
    invert              = False,    # invert before stippling
    threshold           = 0.55,     # darkness cutoff 0–1 (below → no dots)
    seed                = 39,       # random seed
    max_attempts_factor = 20.0,     # sampling attempts = target × factor
    min_sep_factor      = 1.2,      # min gap = 2 × radius × factor
    scale_with_darkness = False,    # make dots bigger in darker areas
    max_radius_scale    = 1.5,      # max radius multiplier (when scaling on)
    combine_paths       = False,    # one compound path (much faster in Illustrator)
    line_mode           = False,    # short lines instead of dots (grass/field effect)
    line_length_factor  = 3.0,      # max line length = factor × dot_radius_pt
)

PT_PER_INCH = 72


# ─── Parameters ───────────────────────────────────────────────────────

@dataclass
class Params:
    in_path: str = ""
    out_path: str = ""
    canvas_size_in: float = DEFAULTS["canvas_size_in"]
    dot_radius_pt: float = DEFAULTS["dot_radius_pt"]
    min_density: float = DEFAULTS["min_density"]
    max_density: float = DEFAULTS["max_density"]
    gamma: float = DEFAULTS["gamma"]
    invert: bool = DEFAULTS["invert"]
    threshold: float = DEFAULTS["threshold"]
    seed: int = DEFAULTS["seed"]
    max_attempts_factor: float = DEFAULTS["max_attempts_factor"]
    min_sep_factor: float = DEFAULTS["min_sep_factor"]
    scale_with_darkness: bool = DEFAULTS["scale_with_darkness"]
    max_radius_scale: float = DEFAULTS["max_radius_scale"]
    combine_paths: bool = DEFAULTS["combine_paths"]
    line_mode: bool = DEFAULTS["line_mode"]
    line_length_factor: float = DEFAULTS["line_length_factor"]


# ─── Fallback 2D array (when numpy is not installed) ──────────────────

class Array2D:
    __slots__ = ("data", "width", "height", "_max", "_mean")

    def __init__(self, data: List[float], width: int, height: int):
        self.data = data
        self.width = width
        self.height = height
        self._max = None
        self._mean = None

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)

    def __getitem__(self, idx: Tuple[int, int]) -> float:
        y, x = idx
        return self.data[y * self.width + x]

    def max(self) -> float:
        if self._max is None:
            self._max = max(self.data) if self.data else 0.0
        return self._max

    def mean(self) -> float:
        if self._mean is None:
            self._mean = (sum(self.data) / len(self.data)) if self.data else 0.0
        return self._mean


# ─── Image processing ─────────────────────────────────────────────────

def load_luma(path: str, invert: bool):
    """Load image as brightness values 0–1 (0=black, 1=white)."""
    img = Image.open(path).convert("RGB")
    gray = ImageOps.grayscale(img)
    if HAS_NUMPY:
        arr = np.asarray(gray, dtype=np.float32) / 255.0
        if invert:
            arr = 1.0 - arr
        return arr
    w, h = gray.size
    data = [v / 255.0 for v in gray.getdata()]
    if invert:
        data = [1.0 - v for v in data]
    return Array2D(data, w, h)


def apply_gamma(arr, gamma: float):
    """Convert brightness → darkness with gamma correction. Returns 0–1 (1=dark)."""
    if HAS_NUMPY:
        b = np.clip(arr, 0.0, 1.0)
        return np.clip(1.0 - np.power(b, gamma), 0.0, 1.0)
    out = []
    for v in arr.data:
        v = max(0.0, min(1.0, v))
        out.append(max(0.0, min(1.0, 1.0 - v ** gamma)))
    return Array2D(out, arr.width, arr.height)


def density_map(darkness, min_d: float, max_d: float, thr: float):
    """Map darkness 0–1 → density (dots/pt²), with optional threshold cutoff."""
    if HAS_NUMPY:
        d = min_d + darkness * (max_d - min_d)
        if thr > 0:
            d = np.where(darkness >= thr, d, 0.0)
        return d
    scale = max_d - min_d
    if thr > 0:
        data = [min_d + v * scale if v >= thr else 0.0 for v in darkness.data]
    else:
        data = [min_d + v * scale for v in darkness.data]
    return Array2D(data, darkness.width, darkness.height)


# ─── Spatial hash for fast neighbor queries ───────────────────────────

class SpatialHash:
    def __init__(self, cell_size: float):
        self.cell = cell_size
        self.buckets: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}

    def _key(self, x: float, y: float) -> Tuple[int, int]:
        return (int(x // self.cell), int(y // self.cell))

    def nearby(self, x: float, y: float) -> List[Tuple[float, float]]:
        kx, ky = self._key(x, y)
        pts = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                b = self.buckets.get((kx + dx, ky + dy))
                if b:
                    pts.extend(b)
        return pts

    def add(self, x: float, y: float):
        self.buckets.setdefault(self._key(x, y), []).append((x, y))


# ─── Point sampling ──────────────────────────────────────────────────

def sample_points(
    dens,
    darkness,
    out_w_pt: float,
    out_h_pt: float,
    min_dist: float,
    target_count: int,
    max_attempts: int,
    rng: random.Random,
) -> List[Tuple[float, float, float]]:
    """Sample stipple dot positions. Returns list of (x_pt, y_pt, darkness_01)."""
    h, w = dens.shape
    sh = SpatialHash(cell_size=min_dist)
    pts: List[Tuple[float, float, float]] = []
    dens_max = float(dens.max())
    if dens_max <= 0:
        return pts

    min_dist_sq = min_dist * min_dist
    attempts = 0

    while attempts < max_attempts and len(pts) < target_count:
        attempts += 1

        x_pt = rng.random() * out_w_pt
        y_pt = rng.random() * out_h_pt

        ix = min(int((x_pt / out_w_pt) * w), w - 1)
        iy = min(int((y_pt / out_h_pt) * h), h - 1)

        local_d = dens[iy, ix]
        if local_d <= 0:
            continue

        if rng.random() > local_d / dens_max:
            continue

        # Min distance check
        ok = True
        for nx, ny in sh.nearby(x_pt, y_pt):
            if (x_pt - nx) ** 2 + (y_pt - ny) ** 2 < min_dist_sq:
                ok = False
                break
        if not ok:
            continue

        sh.add(x_pt, y_pt)
        dark_val = float(darkness[iy, ix])
        pts.append((x_pt, y_pt, dark_val))

    return pts


# ─── Direction noise for line mode ────────────────────────────────────

def _hash01(x: float, y: float) -> float:
    """Deterministic pseudo-random 0–1 from coordinates (GLSL-style)."""
    v = math.sin(x * 12.9898 + y * 78.233) * 43758.5453
    return v - math.floor(v)


def _angle_noise(nx: float, ny: float, seed: float = 0.0) -> float:
    """Smooth angle field from layered sinusoids. nx/ny normalised ~[0,1]."""
    s = seed * 17.31
    a  = math.sin(nx * 6.3 + 1.7 + s) * math.cos(ny * 7.1 + 2.3 + s)
    a += 0.5 * math.sin(nx * 13.7 + 3.1 + s * 0.7) * math.cos(ny * 15.3 + 0.8 + s)
    a += 0.25 * math.sin(nx * 27.1 + 5.2 + s * 1.3) * math.cos(ny * 23.9 + 4.1 + s)
    return a * math.pi


# ─── SVG output ───────────────────────────────────────────────────────

def write_svg(
    path: str,
    width_pt: float,
    height_pt: float,
    base_r: float,
    points: List[Tuple[float, float, float]],
    scale_with_darkness: bool,
    max_radius_scale: float,
    combine_paths: bool,
    line_mode: bool = False,
    line_length_factor: float = 3.0,
    seed: int = 0,
):
    """Write dots (or lines) to SVG. Points are (x, y, darkness_01)."""
    width_in = width_pt / PT_PER_INCH
    height_in = height_pt / PT_PER_INCH

    header = (
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg"\n'
        f'     width="{width_in:.4f}in" height="{height_in:.4f}in"\n'
        f'     viewBox="0 0 {width_pt:.2f} {height_pt:.2f}">\n'
    )

    def dot_radius(dark: float) -> float:
        if scale_with_darkness:
            return base_r * (1.0 + (max_radius_scale - 1.0) * dark)
        return base_r

    def _line_endpoints(x, y, dark):
        """Compute (x1,y1,x2,y2) for a line centred on (x,y)."""
        nx, ny = x / width_pt, y / height_pt
        angle = _angle_noise(nx, ny, float(seed))
        # per-point jitter so it doesn't look too mechanical
        angle += (_hash01(x * 7.3, y * 11.7) - 0.5) * 0.6
        # length varies with darkness AND per-point noise (some stay dot-like)
        noise = _hash01(x * 3.1, y * 5.7)
        half_len = dark * line_length_factor * base_r * noise
        dx = math.cos(angle) * half_len
        dy = math.sin(angle) * half_len
        return x - dx, y - dy, x + dx, y + dy

    with open(path, "w", encoding="utf-8") as f:
        f.write(header)

        if line_mode:
            stroke_w = 2 * base_r
            if combine_paths:
                f.write(f'  <path stroke="#000000" stroke-width="{stroke_w:.3f}" '
                        f'stroke-linecap="round" fill="none" d="\n')
                for x, y, dark in points:
                    x1, y1, x2, y2 = _line_endpoints(x, y, dark)
                    f.write(f'M{x1:.2f},{y1:.2f}L{x2:.2f},{y2:.2f}\n')
                f.write('"/>\n')
            else:
                f.write(f'  <g stroke="#000000" stroke-width="{stroke_w:.3f}" '
                        f'stroke-linecap="round" fill="none">\n')
                for x, y, dark in points:
                    x1, y1, x2, y2 = _line_endpoints(x, y, dark)
                    f.write(f'    <line x1="{x1:.2f}" y1="{y1:.2f}" '
                            f'x2="{x2:.2f}" y2="{y2:.2f}"/>\n')
                f.write('  </g>\n')

        elif combine_paths:
            f.write('  <path fill="#000000" stroke="none" d="\n')
            for x, y, dark in points:
                r = dot_radius(dark)
                d2 = 2 * r
                f.write(f'M{x - r:.2f},{y:.2f}'
                        f'a{r:.3f},{r:.3f} 0 1,0 {d2:.3f},0'
                        f'a{r:.3f},{r:.3f} 0 1,0 {-d2:.3f},0z\n')
            f.write('"/>\n')
        else:
            f.write('  <g fill="#000000" stroke="none">\n')
            if scale_with_darkness:
                for x, y, dark in points:
                    r = dot_radius(dark)
                    f.write(f'    <circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.3f}"/>\n')
            else:
                for x, y, _ in points:
                    f.write(f'    <circle cx="{x:.2f}" cy="{y:.2f}" r="{base_r:.3f}"/>\n')
            f.write('  </g>\n')

        f.write('</svg>\n')


# ─── Core generation logic ────────────────────────────────────────────

def generate(p: Params) -> dict:
    """Run the full stipple pipeline. Returns stats dict."""
    if not HAS_NUMPY:
        print("numpy not found — using pure-Python fallback (slower).")

    rng = random.Random(p.seed)

    luma = load_luma(p.in_path, invert=p.invert)
    darkness = apply_gamma(luma, p.gamma)
    dens = density_map(darkness, p.min_density, p.max_density, p.threshold)

    # Output dimensions: longest edge = canvas_size_in
    h_px, w_px = dens.shape
    canvas_pt = p.canvas_size_in * PT_PER_INCH
    if w_px >= h_px:
        out_w_pt = canvas_pt
        out_h_pt = canvas_pt * (h_px / w_px)
    else:
        out_h_pt = canvas_pt
        out_w_pt = canvas_pt * (w_px / h_px)

    # Target dot count from integrated density
    area_pt2 = out_w_pt * out_h_pt
    avg_density = float(dens.mean())
    target = int(avg_density * area_pt2)

    min_dist = 2.0 * p.dot_radius_pt * p.min_sep_factor
    max_attempts = int(max(1000, target * p.max_attempts_factor))

    print(f"Canvas: {out_w_pt / PT_PER_INCH:.2f} x {out_h_pt / PT_PER_INCH:.2f} in "
          f"({out_w_pt:.0f} x {out_h_pt:.0f} pt)")
    print(f"Target dots: {target:,} | max attempts: {max_attempts:,}")

    pts = sample_points(
        dens=dens,
        darkness=darkness,
        out_w_pt=out_w_pt,
        out_h_pt=out_h_pt,
        min_dist=min_dist,
        target_count=target,
        max_attempts=max_attempts,
        rng=rng,
    )

    write_svg(
        path=p.out_path,
        width_pt=out_w_pt,
        height_pt=out_h_pt,
        base_r=p.dot_radius_pt,
        points=pts,
        scale_with_darkness=p.scale_with_darkness,
        max_radius_scale=p.max_radius_scale,
        combine_paths=p.combine_paths,
        line_mode=p.line_mode,
        line_length_factor=p.line_length_factor,
        seed=p.seed,
    )

    stats = dict(
        dots=len(pts),
        width_in=out_w_pt / PT_PER_INCH,
        height_in=out_h_pt / PT_PER_INCH,
        width_pt=out_w_pt,
        height_pt=out_h_pt,
        target=target,
    )
    print(f"Wrote {stats['dots']:,} dots to {p.out_path}")
    return stats


# ─── CLI ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Raster → stochastic stipple dots → SVG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input", nargs="?", default="Input.png",
                    help="Input raster image")
    ap.add_argument("output", nargs="?", default="output.svg",
                    help="Output SVG path")
    ap.add_argument("--gui", action="store_true",
                    help="Launch minimal GUI instead of CLI")
    ap.add_argument("--canvas-size-in", type=float,
                    default=DEFAULTS["canvas_size_in"],
                    help=f"Longest edge in inches (default: {DEFAULTS['canvas_size_in']})")
    ap.add_argument("--dot-radius-pt", type=float,
                    default=DEFAULTS["dot_radius_pt"],
                    help=f"Dot radius in points (default: {DEFAULTS['dot_radius_pt']})")
    ap.add_argument("--min-density", type=float,
                    default=DEFAULTS["min_density"],
                    help=f"Min density dots/pt² (default: {DEFAULTS['min_density']})")
    ap.add_argument("--max-density", type=float,
                    default=DEFAULTS["max_density"],
                    help=f"Max density dots/pt² (default: {DEFAULTS['max_density']})")
    ap.add_argument("--gamma", type=float,
                    default=DEFAULTS["gamma"],
                    help=f"Brightness gamma (default: {DEFAULTS['gamma']})")
    ap.add_argument("--invert", action="store_true",
                    help="Invert image before stippling")
    ap.add_argument("--threshold", type=float,
                    default=DEFAULTS["threshold"],
                    help=f"Darkness threshold 0–1 (default: {DEFAULTS['threshold']})")
    ap.add_argument("--seed", type=int,
                    default=DEFAULTS["seed"],
                    help=f"Random seed (default: {DEFAULTS['seed']})")
    ap.add_argument("--max-attempts-factor", type=float,
                    default=DEFAULTS["max_attempts_factor"],
                    help=f"Attempt multiplier (default: {DEFAULTS['max_attempts_factor']})")
    ap.add_argument("--min-sep-factor", type=float,
                    default=DEFAULTS["min_sep_factor"],
                    help=f"Separation factor (default: {DEFAULTS['min_sep_factor']})")
    ap.add_argument("--scale-with-darkness", action="store_true",
                    default=DEFAULTS["scale_with_darkness"],
                    help="Scale dot radius larger in darker areas")
    ap.add_argument("--max-radius-scale", type=float,
                    default=DEFAULTS["max_radius_scale"],
                    help=f"Max radius multiplier (default: {DEFAULTS['max_radius_scale']})")
    ap.add_argument("--combine-paths", action="store_true",
                    default=DEFAULTS["combine_paths"],
                    help="Merge dots into one compound path (faster in Illustrator)")
    ap.add_argument("--line-mode", action="store_true",
                    default=DEFAULTS["line_mode"],
                    help="Short lines instead of dots (grass/vector-field effect)")
    ap.add_argument("--line-length-factor", type=float,
                    default=DEFAULTS["line_length_factor"],
                    help=f"Max line length = factor × dot_radius (default: {DEFAULTS['line_length_factor']})")

    args = ap.parse_args()

    if args.gui:
        launch_gui()
        return

    p = Params(
        in_path=args.input,
        out_path=args.output,
        canvas_size_in=args.canvas_size_in,
        dot_radius_pt=args.dot_radius_pt,
        min_density=args.min_density,
        max_density=args.max_density,
        gamma=args.gamma,
        invert=args.invert,
        threshold=args.threshold,
        seed=args.seed,
        max_attempts_factor=args.max_attempts_factor,
        min_sep_factor=args.min_sep_factor,
        scale_with_darkness=args.scale_with_darkness,
        max_radius_scale=args.max_radius_scale,
        combine_paths=args.combine_paths,
        line_mode=args.line_mode,
        line_length_factor=args.line_length_factor,
    )
    generate(p)


# ─── Minimal GUI (tkinter) ───────────────────────────────────────────

def launch_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox
    import threading
    import os

    root = tk.Tk()
    root.title("Stipple SVG")
    root.resizable(False, False)

    row = [0]  # mutable counter

    def next_row():
        r = row[0]
        row[0] += 1
        return r

    def add_file_row(label_text, default="", save=False):
        r = next_row()
        tk.Label(root, text=label_text, anchor="e").grid(
            row=r, column=0, sticky="e", padx=(8, 4), pady=2)
        var = tk.StringVar(value=default)
        tk.Entry(root, textvariable=var, width=40).grid(
            row=r, column=1, padx=2, pady=2)

        def browse():
            if save:
                p = filedialog.asksaveasfilename(
                    defaultextension=".svg",
                    filetypes=[("SVG", "*.svg"), ("All", "*.*")])
            else:
                p = filedialog.askopenfilename(filetypes=[
                    ("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                    ("All", "*.*")])
            if p:
                var.set(p)

        tk.Button(root, text="Browse…", command=browse, width=8).grid(
            row=r, column=2, padx=(2, 8), pady=2)
        return var

    def add_entry_row(label_text, default):
        r = next_row()
        tk.Label(root, text=label_text, anchor="e").grid(
            row=r, column=0, sticky="e", padx=(8, 4), pady=2)
        var = tk.StringVar(value=str(default))
        tk.Entry(root, textvariable=var, width=12).grid(
            row=r, column=1, sticky="w", padx=2, pady=2)
        return var

    def add_check_row(label_text, default=False):
        r = next_row()
        var = tk.BooleanVar(value=default)
        tk.Checkbutton(root, text=label_text, variable=var).grid(
            row=r, column=0, columnspan=2, sticky="w", padx=(8, 4), pady=2)
        return var

    def add_separator():
        r = next_row()
        tk.Frame(root, height=1, bg="#cccccc").grid(
            row=r, column=0, columnspan=3, sticky="ew", padx=8, pady=6)

    # ── Layout ──

    input_var = add_file_row("Input image:", "")
    output_var = add_file_row("Output SVG:", "output.svg", save=True)
    add_separator()

    canvas_var = add_entry_row("Canvas size (in):", DEFAULTS["canvas_size_in"])
    radius_var = add_entry_row("Dot radius (pt):", DEFAULTS["dot_radius_pt"])
    min_d_var = add_entry_row("Min density:", DEFAULTS["min_density"])
    max_d_var = add_entry_row("Max density:", DEFAULTS["max_density"])
    gamma_var = add_entry_row("Gamma:", DEFAULTS["gamma"])
    thresh_var = add_entry_row("Threshold:", DEFAULTS["threshold"])
    seed_var = add_entry_row("Seed:", DEFAULTS["seed"])
    attempts_var = add_entry_row("Attempts factor:", DEFAULTS["max_attempts_factor"])
    sep_var = add_entry_row("Separation factor:", DEFAULTS["min_sep_factor"])
    add_separator()

    invert_var = add_check_row("Invert image", DEFAULTS["invert"])
    scale_var = add_check_row("Scale dots with darkness", DEFAULTS["scale_with_darkness"])
    max_scale_var = add_entry_row("  Max radius scale:", DEFAULTS["max_radius_scale"])
    combine_var = add_check_row("Combine paths (faster in AI)", DEFAULTS["combine_paths"])
    line_mode_var = add_check_row("Line mode (grass/field effect)", DEFAULTS["line_mode"])
    line_len_var = add_entry_row("  Line length factor:", DEFAULTS["line_length_factor"])
    add_separator()

    # ── Status + generate button ──

    r = next_row()
    status_var = tk.StringVar(value="Ready")
    tk.Label(root, textvariable=status_var, anchor="w", fg="#666666").grid(
        row=r, column=0, columnspan=2, sticky="w", padx=8, pady=4)

    def on_generate():
        in_path = input_var.get().strip()
        out_path = output_var.get().strip()
        if not in_path:
            messagebox.showerror("Error", "Select an input image.")
            return
        if not os.path.isfile(in_path):
            messagebox.showerror("Error", f"Input file not found:\n{in_path}")
            return
        if not out_path:
            messagebox.showerror("Error", "Specify an output path.")
            return

        try:
            p = Params(
                in_path=in_path,
                out_path=out_path,
                canvas_size_in=float(canvas_var.get()),
                dot_radius_pt=float(radius_var.get()),
                min_density=float(min_d_var.get()),
                max_density=float(max_d_var.get()),
                gamma=float(gamma_var.get()),
                invert=invert_var.get(),
                threshold=float(thresh_var.get()),
                seed=int(seed_var.get()),
                max_attempts_factor=float(attempts_var.get()),
                min_sep_factor=float(sep_var.get()),
                scale_with_darkness=scale_var.get(),
                max_radius_scale=float(max_scale_var.get()),
                combine_paths=combine_var.get(),
                line_mode=line_mode_var.get(),
                line_length_factor=float(line_len_var.get()),
            )
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter value:\n{e}")
            return

        gen_btn.config(state="disabled")
        status_var.set("Generating…")
        root.update()

        def run():
            try:
                stats = generate(p)
                root.after(0, lambda: status_var.set(
                    f"Done — {stats['dots']:,} dots, "
                    f"{stats['width_in']:.1f} x {stats['height_in']:.1f} in"))
            except Exception as e:
                root.after(0, lambda: messagebox.showerror("Error", str(e)))
                root.after(0, lambda: status_var.set("Error"))
            finally:
                root.after(0, lambda: gen_btn.config(state="normal"))

        threading.Thread(target=run, daemon=True).start()

    gen_btn = tk.Button(root, text="Generate", command=on_generate, width=12)
    gen_btn.grid(row=r, column=2, padx=(2, 8), pady=4)

    root.mainloop()


if __name__ == "__main__":
    main()
