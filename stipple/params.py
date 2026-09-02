"""Parameter model and preset persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

PT_PER_INCH = 72.0

SAMPLERS = ("relaxed", "poisson", "classic")
COLOR_MODES = ("mono", "source")
SVG_STRUCTURES = ("compound", "circles")


@dataclass
class Params:
    # ── paths ────────────────────────────────────────────────────────
    in_path: str = ""
    out_path: str = "output.svg"

    # ── canvas ───────────────────────────────────────────────────────
    # Width is authoritative. When lock_aspect is True, height is derived
    # from the source image on load. When False, the image is fitted and
    # centred inside the canvas, leaving paper-coloured margins.
    canvas_w_in: float = 21.3333
    canvas_h_in: float = 28.0
    lock_aspect: bool = True

    # ── image preparation ────────────────────────────────────────────
    invert: bool = False
    pre_blur: float = 0.0        # gaussian sigma in source pixels
    black_point: float = 0.0     # levels: input black  (0-1)
    white_point: float = 1.0     # levels: input white  (0-1)
    contrast: float = 0.0        # -1 .. +1, pivots on mid grey
    gamma: float = 2.2           # darkness curve

    # ── density field ────────────────────────────────────────────────
    min_density: float = 0.0     # dots per pt^2 in lightest areas
    max_density: float = 0.225   # dots per pt^2 in darkest areas
    threshold: float = 0.0       # darkness below this produces no dots

    # ── sampling ─────────────────────────────────────────────────────
    sampler: str = "relaxed"
    seed: int = 39
    relax_iterations: int = 20   # spacing relaxation passes (relaxed sampler)
    min_sep_factor: float = 1.2  # spacing floor = 2 * radius * factor
    max_attempts_factor: float = 20.0  # classic sampler only

    # ── dot rendering ────────────────────────────────────────────────
    dot_radius_pt: float = 0.25
    scale_with_darkness: bool = False
    max_radius_scale: float = 1.5

    # ── line / flow mode ─────────────────────────────────────────────
    line_mode: bool = False
    line_length_factor: float = 6.0   # stroke length in multiples of local spacing
    line_taper: bool = True
    flow_smoothing: float = 4.0      # structure-tensor gaussian sigma (px)
    flow_diffusion: int = 6          # coherence-weighted diffusion passes
    flow_bias_angle: float = 90.0    # degrees; 90 = "grows upward"
    flow_bias_strength: float = 0.5  # blend toward bias where incoherent
    flow_perpendicular: bool = False # rotate field 90 degrees
    flow_jitter: float = 0.15        # per-stroke angular noise (radians)

    # ── colour ───────────────────────────────────────────────────────
    color_mode: str = "mono"
    ink: str = "#000000"
    paper: str = "#ffffff"

    # ── output ───────────────────────────────────────────────────────
    svg_structure: str = "compound"
    paper_background: bool = False   # emit a background rect

    # ─────────────────────────────────────────────────────────────────

    def validate(self) -> None:
        if self.sampler not in SAMPLERS:
            raise ValueError(f"sampler must be one of {SAMPLERS}, got {self.sampler!r}")
        if self.color_mode not in COLOR_MODES:
            raise ValueError(f"color_mode must be one of {COLOR_MODES}, got {self.color_mode!r}")
        if self.svg_structure not in SVG_STRUCTURES:
            raise ValueError(f"svg_structure must be one of {SVG_STRUCTURES}, got {self.svg_structure!r}")
        if self.canvas_w_in <= 0 or self.canvas_h_in <= 0:
            raise ValueError("canvas dimensions must be positive")
        if self.dot_radius_pt <= 0:
            raise ValueError("dot_radius_pt must be positive")
        if self.max_density <= 0:
            raise ValueError("max_density must be positive")
        if self.min_density < 0 or self.min_density > self.max_density:
            raise ValueError("min_density must be between 0 and max_density")
        if not 0.0 <= self.black_point < self.white_point <= 1.0:
            raise ValueError("require 0 <= black_point < white_point <= 1")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")
        if not -1.0 <= self.contrast <= 1.0:
            raise ValueError("contrast must be between -1 and 1")
        if self.relax_iterations < 0:
            raise ValueError("relax_iterations must be >= 0")

    @property
    def canvas_w_pt(self) -> float:
        return self.canvas_w_in * PT_PER_INCH

    @property
    def canvas_h_pt(self) -> float:
        return self.canvas_h_in * PT_PER_INCH

    # ── presets ──────────────────────────────────────────────────────

    #: keys that describe *look*, not a particular job
    _JOB_KEYS = ("in_path", "out_path", "canvas_w_in", "canvas_h_in")

    def to_preset(self) -> dict[str, Any]:
        d = asdict(self)
        for k in self._JOB_KEYS:
            d.pop(k, None)
        return d

    def apply_preset(self, data: dict[str, Any]) -> "Params":
        known = {f.name for f in fields(self)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown preset keys: {sorted(unknown)}")
        for k, v in data.items():
            if k in self._JOB_KEYS:
                continue
            setattr(self, k, v)
        return self

    def save_preset(self, path: str | Path, name: str = "") -> None:
        payload = {"name": name or Path(path).stem, "params": self.to_preset()}
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load_preset(cls, path: str | Path, **overrides: Any) -> "Params":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        data = raw.get("params", raw)
        p = cls()
        p.apply_preset(data)
        for k, v in overrides.items():
            setattr(p, k, v)
        return p
