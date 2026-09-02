# stipple

Converts a raster image into stipple dots or flow-following strokes, written
as SVG for Illustrator.

## Running it

Open **Stipple.app** from `~/Applications` (Finder sidebar, Go > Applications,
or Spotlight). Drag it to the Dock if you use it often.

It installs there rather than beside the source because macOS will not launch
an app bundle from this project's exFAT volume. The volume is mounted
`noowners`, so the executable has no verifiable ownership and LaunchServices
declines: `open` reports success and no process starts. The launcher sits on
the internal drive and points back here. After moving the project, rerun
`python3 make_launcher.py` to repoint it.

From a terminal:

```bash
~/.local/share/stipple/venv/bin/python web/server.py
```

## Setup

```bash
./bootstrap.sh
python3 make_launcher.py
```

`bootstrap.sh` builds a virtualenv at `~/.local/share/stipple/venv` from
Homebrew Python 3.12. It lives on the internal drive because virtualenvs are
unreliable on exFAT and the volume can be unmounted.

`make_launcher.py` installs a thin bundle that runs this source tree, so
edits take effect on the next launch. For a bundle that runs on a Mac with no
Python, `python3 build_app.py` writes a self-contained one to `dist/`. It is
Apple Silicon only and unsigned, so its first launch on another machine needs
right-click then Open. Copy it off this volume before running it.

## The interface

Controls on the left, preview on the right. The detail window shows marks at
printed size; a 0.25pt dot on a 28in canvas is invisible in a whole-canvas
view, so judge mark size there. Click the preview to move the window.

Preview runs the same code as the export on a scaled-down canvas. Marks keep
their true size, so ink coverage, and with it tone, matches the export to
within two percent. While a slider moves you get a coarse pass at about 50ms;
on release it refines. Click any number to type an exact value.

## Samplers

| | spacing uniformity | character |
|---|---|---|
| `relaxed` (default) | 0.21 | evenest |
| `poisson` | 0.29 | organic |
| `classic` | 0.59 | the original |

Nearest-neighbour distance normalised by local target spacing. Lower is more
even; a Poisson process measures about 0.52. `classic` is kept because
existing artwork was made with it. `relax_iterations` moves between organic
and even.

## Line mode

Stroke direction comes from the image. Sobel gradients build a structure
tensor whose minor eigenvector points along local grain and whose eigenvalue
gap gives a confidence. Flat regions have no direction of their own, so
confident directions diffuse outward into them, and what remains blends
toward the bias angle.

**length** is measured in multiples of local spacing, not stroke width. Above
1, neighbouring strokes interleave and the texture reads as woven rather than
as separate marks.

**taper** fills each stroke as an outline narrowing to a point at both ends.
Turn it off for plottable centrelines.

**run across structure** rotates the field 90 degrees, turning
grain-following strokes into contour hatching.

## Presets

`presets/` holds six starting points: `fine-dots`, `dense-dots`,
`organic-dots`, `grass`, `contour-hatch`, `plotter-lines`. Load and save them
from the buttons below export, or from the CLI.

## Command line

```bash
V=~/.local/share/stipple/venv/bin/python

$V -m stipple.cli render Input.png out.svg
$V -m stipple.cli render Input.png out.svgz --preset presets/grass.json
$V -m stipple.cli batch ./renders --preset presets/fine-dots.json
$V -m stipple.cli render --help
```

Naming the output `.svgz` gzips it. Illustrator opens both. A typical run is
9.5 MB uncompressed and 0.63 MB gzipped.

## Output

`compound` writes every mark into one path, so Illustrator handles a single
object instead of 216,000 separate `<circle>` elements. `circles` keeps them
separate when you need to select individual dots.

Colour is one ink by default. `from image` samples each mark's colour from
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

## Licence

MIT. See `LICENSE`.
