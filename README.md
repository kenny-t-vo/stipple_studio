# stipple

Turns a raster image into stipple dots or flow-following strokes, written as
SVG for Illustrator.

## Running it

Double-click **Stipple.app** in `~/Applications` (Finder sidebar → Go →
Applications, or Spotlight). Drag it to the Dock if you use it often.

It is installed there rather than in this folder for a reason: **macOS will
not launch an app bundle from this exFAT volume.** The volume is mounted
`noowners`, so the executable has no verifiable ownership and LaunchServices
silently declines — `open` reports success and nothing happens. The launcher
lives on the internal drive and points back here. If you move this project,
rerun `python3 make_launcher.py` to repoint it.

From a terminal:

```bash
~/.local/share/stipple/venv/bin/python web/server.py
```

## First-time setup, or on another Mac

```bash
./bootstrap.sh
python3 make_launcher.py
```

`bootstrap.sh` builds the virtualenv at `~/.local/share/stipple/venv` using
Python 3.12 from Homebrew. It lives on the internal drive on purpose: this
project sits on an exFAT volume, where virtualenvs are unreliable and which
may be unmounted.

`make_launcher.py` installs `Stipple.app` into `~/Applications`, a thin
bundle pointing at this source tree — edits take effect on the next launch, nothing to rebuild. For a
bundle that runs on a Mac with no Python at all, `python3 build_app.py`
produces a self-contained one in `dist/` (Apple Silicon, unsigned, so its
first launch on another machine needs right-click → Open). Copy it off this
volume before launching it, for the reason above.

## The interface

Controls on the left, preview on the right, with a detail window showing
marks at the size they will actually print — a 0.25pt dot on a 28in canvas is
invisible in any whole-canvas view, so that window is where you judge mark
size. Click anywhere in the preview to move it.

The preview runs the same code as the export, on a scaled-down canvas. Marks
keep their true size, so ink coverage — and therefore tone — matches the
export to within a couple of percent. While a slider moves you get a coarse
pass (~50ms); on release it refines.

## Samplers

| | spacing uniformity | character |
|---|---|---|
| `relaxed` (default) | 0.21 | evenest |
| `poisson` | 0.29 | organic |
| `classic` | 0.59 | the original — random scatter |

Measured as nearest-neighbour distance normalised by local target spacing;
lower is more even. `classic` is preserved because existing artwork was made
with it, but its spread is indistinguishable from a Poisson process, which is
to say it is not really stippling. `relax_iterations` is the dial between
organic and even.

## Line mode

Strokes follow the image's own structure — Sobel gradients into a structure
tensor, whose minor eigenvector points along local grain and whose eigenvalue
gap says how far to trust it. Flat regions have no direction of their own, so
confident directions are diffused outward into them and what remains blends
toward the bias angle.

- **length** is in multiples of local spacing, not stroke width. Above 1,
  neighbouring strokes interleave, which is what makes a woven texture rather
  than isolated marks.
- **taper** fills each stroke as an outline that narrows to a point at both
  ends. Turn it off for plottable centrelines.
- **run across structure** rotates the field 90°, turning grain-following
  strokes into contour hatching.

## Presets

`presets/` holds six starting points: `fine-dots`, `dense-dots`,
`organic-dots`, `grass`, `contour-hatch`, `plotter-lines`. Load and save them
from the buttons under the export button, or from the CLI.

## Command line

```bash
V=~/.local/share/stipple/venv/bin/python

$V -m stipple.cli render Input.png out.svg
$V -m stipple.cli render Input.png out.svgz --preset presets/grass.json
$V -m stipple.cli batch ./renders --preset presets/fine-dots.json
$V -m stipple.cli render --help
```

`.svgz` output is gzipped SVG, which Illustrator opens directly and which is
roughly fifteen times smaller — 0.63 MB against 9.5 MB for a typical run.

## Output

`compound` emits every mark as one path: a single object rather than the
216,000 separate `<circle>` elements the original produced, which is the
difference between Illustrator being usable and not. `circles` keeps them
separate when you actually need to select individual dots.

Colour is one ink by default; `from image` samples each mark's colour from
the source and groups marks into one path per quantised colour.

## Tests

```bash
~/.local/share/stipple/venv/bin/python -m pytest tests/ -q
```

## Layout

```
stipple/     image · density · sample · flow · strokes · render · params · core · cli
web/         server · engine · index.html · app.css · app.js
presets/     tests/
```
