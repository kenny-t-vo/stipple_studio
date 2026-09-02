"""Command line interface."""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

from .core import generate
from .params import COLOR_MODES, Params, SAMPLERS, SVG_STRUCTURES

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}


def _add_params(ap: argparse.ArgumentParser) -> None:
    d = Params()
    g = ap.add_argument_group("canvas")
    g.add_argument("--width-in", type=float, default=d.canvas_w_in, dest="canvas_w_in")
    g.add_argument("--height-in", type=float, default=d.canvas_h_in, dest="canvas_h_in")
    # BooleanOptionalAction gives --lock-aspect / --no-lock-aspect. The old
    # script paired action="store_true" with a default read from a config
    # dict, so any flag whose default was True could never be switched off.
    g.add_argument("--lock-aspect", action=argparse.BooleanOptionalAction,
                   default=d.lock_aspect,
                   help="derive height from the source image (default: on)")

    g = ap.add_argument_group("tone")
    g.add_argument("--invert", action=argparse.BooleanOptionalAction, default=d.invert)
    g.add_argument("--pre-blur", type=float, default=d.pre_blur,
                   help="gaussian sigma in source pixels; suppresses sensor noise")
    g.add_argument("--black-point", type=float, default=d.black_point)
    g.add_argument("--white-point", type=float, default=d.white_point)
    g.add_argument("--contrast", type=float, default=d.contrast, help="-1 to 1")
    g.add_argument("--gamma", type=float, default=d.gamma)
    g.add_argument("--threshold", type=float, default=d.threshold)

    g = ap.add_argument_group("density and sampling")
    g.add_argument("--min-density", type=float, default=d.min_density)
    g.add_argument("--max-density", type=float, default=d.max_density)
    g.add_argument("--sampler", choices=SAMPLERS, default=d.sampler)
    g.add_argument("--seed", type=int, default=d.seed)
    g.add_argument("--relax-iterations", type=int, default=d.relax_iterations)
    g.add_argument("--min-sep-factor", type=float, default=d.min_sep_factor)
    g.add_argument("--max-attempts-factor", type=float, default=d.max_attempts_factor)

    g = ap.add_argument_group("dots")
    g.add_argument("--dot-radius-pt", type=float, default=d.dot_radius_pt)
    g.add_argument("--scale-with-darkness", action=argparse.BooleanOptionalAction,
                   default=d.scale_with_darkness)
    g.add_argument("--max-radius-scale", type=float, default=d.max_radius_scale)

    g = ap.add_argument_group("lines and flow field")
    g.add_argument("--line-mode", action=argparse.BooleanOptionalAction, default=d.line_mode,
                   help="short flow-following strokes instead of dots")
    g.add_argument("--line-length-factor", type=float, default=d.line_length_factor,
                   help="stroke length in multiples of local spacing")
    g.add_argument("--line-taper", action=argparse.BooleanOptionalAction,
                   default=d.line_taper,
                   help="taper stroke ends (off gives plottable centrelines)")
    g.add_argument("--flow-smoothing", type=float, default=d.flow_smoothing)
    g.add_argument("--flow-diffusion", type=int, default=d.flow_diffusion)
    g.add_argument("--flow-bias-angle", type=float, default=d.flow_bias_angle)
    g.add_argument("--flow-bias-strength", type=float, default=d.flow_bias_strength)
    g.add_argument("--flow-perpendicular", action=argparse.BooleanOptionalAction,
                   default=d.flow_perpendicular)
    g.add_argument("--flow-jitter", type=float, default=d.flow_jitter)

    g = ap.add_argument_group("colour and output")
    g.add_argument("--color-mode", choices=COLOR_MODES, default=d.color_mode)
    g.add_argument("--ink", default=d.ink)
    g.add_argument("--paper", default=d.paper)
    g.add_argument("--svg-structure", choices=SVG_STRUCTURES, default=d.svg_structure)
    g.add_argument("--paper-background", action=argparse.BooleanOptionalAction,
                   default=d.paper_background)
    g.add_argument("--png-dpi", type=int, default=d.png_dpi,
                   help="also write a raster proof at this DPI (0 disables)")


def _params_from(args, **over) -> Params:
    names = {f.name for f in fields(Params)}
    kw = {k: v for k, v in vars(args).items() if k in names}
    kw.update(over)
    p = Params(**kw)
    if getattr(args, "preset", None):
        base = Params.load_preset(args.preset)
        for f in fields(Params):
            if f.name not in Params._JOB_KEYS and f"--{f.name.replace('_','-')}" not in sys.argv:
                setattr(p, f.name, getattr(base, f.name))
    return p


def _report(p: Params, res) -> None:
    g = res.geom
    print(f"  canvas   {g.canvas_w_in:.2f} x {g.canvas_h_in:.2f} in "
          f"({g.canvas_w_pt:.0f} x {g.canvas_h_pt:.0f} pt)")
    if g.has_margins:
        print(f"  image    inset {g.img_x_pt:.0f},{g.img_y_pt:.0f} pt (aspect unlocked)")
    print(f"  sampler  {p.sampler}  seed {p.seed}")
    print(f"  marks    {res.count:,} of {res.target:,} target")
    size = res.stats.get("bytes", 0) / 1048576
    print(f"  wrote    {p.out_path}  ({size:.2f} MB, {res.elapsed:.1f}s)")
    if "png_path" in res.stats:
        print(f"           {res.stats['png_path']}")


def cmd_render(args) -> int:
    p = _params_from(args, in_path=args.input, out_path=args.output)
    res = generate(p, png_dpi=p.png_dpi or None)
    _report(p, res)
    return 0


def cmd_batch(args) -> int:
    src = Path(args.folder)
    if not src.is_dir():
        print(f"not a directory: {src}", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir) if args.out_dir else src / "stippled"
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(f for f in src.iterdir()
                    if f.suffix.lower() in IMAGE_SUFFIXES and not f.name.startswith("."))
    if not images:
        print(f"no images found in {src}", file=sys.stderr)
        return 1

    print(f"{len(images)} image(s) -> {out_dir}")
    failed = 0
    for i, f in enumerate(images, 1):
        out = out_dir / (f.stem + args.suffix)
        print(f"[{i}/{len(images)}] {f.name}")
        try:
            p = _params_from(args, in_path=str(f), out_path=str(out))
            _report(p, generate(p, png_dpi=p.png_dpi or None))
        except Exception as exc:
            failed += 1
            print(f"  failed: {exc}", file=sys.stderr)
    if failed:
        print(f"\n{failed} of {len(images)} failed", file=sys.stderr)
    return 1 if failed else 0


def cmd_preset(args) -> int:
    if args.action == "save":
        p = _params_from(args, in_path="", out_path="")
        p.save_preset(args.path, name=Path(args.path).stem)
        print(f"saved preset -> {args.path}")
    else:
        p = Params.load_preset(args.path)
        for f in fields(Params):
            if f.name not in Params._JOB_KEYS:
                print(f"  {f.name:22s} {getattr(p, f.name)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="stipple", description="Raster image to stipple SVG")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", help="stipple a single image")
    r.add_argument("input")
    r.add_argument("output", nargs="?", default="output.svg",
                   help="output path; .svgz gzips it (Illustrator reads both)")
    r.add_argument("--preset", help="load a preset JSON first")
    _add_params(r)
    r.set_defaults(func=cmd_render)

    b = sub.add_parser("batch", help="stipple every image in a folder")
    b.add_argument("folder")
    b.add_argument("--out-dir")
    b.add_argument("--suffix", default=".svg")
    b.add_argument("--preset")
    _add_params(b)
    b.set_defaults(func=cmd_batch)

    s = sub.add_parser("preset", help="save or show a preset")
    s.add_argument("action", choices=("save", "show"))
    s.add_argument("path")
    _add_params(s)
    s.set_defaults(func=cmd_preset)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
