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

## In a browser

<https://kenny-t-vo.github.io/stipple_studio/>

The same pipeline, running in the page. Pyodide fetches Python, numpy and
pillow on the first visit, about 10MB, and caches them. Pick an image, tweak,
export; nothing is uploaded and nothing is installed.

It is slower than the desktop app. WASM numpy has no threads, and the browser
build leaves scipy out because it alone is 13MB against numpy's 2.8MB. A
refined preview takes about 3 seconds against the desktop's 0.3, and the
refined preview caps relaxation at 8 passes, where evenness stops improving.
Exports are not capped.

Large TIFFs are the reason to keep the desktop app. Pyodide's heap tops out at
4GB, so a full-resolution 16-bit scan belongs there.

```bash
python3 browser/build.py        # writes dist/
```

A push to `main` builds and publishes it.

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

## Tone

**auto tone** sets the black and white points from a percentile clip of the
histogram, then re-solves gamma so mean darkness is unchanged. Mean darkness
is the mark count -- density is linear in it -- so the button never makes the
drawing heavier or lighter. It only redistributes tone. On the reference image
it moved the count by 0.05% and widened the range in use by 10%.

## The preview

The fit view holds ink coverage exactly and magnifies the grain by 1/scale, so
on a large canvas it reads coarser than the print. The stats line says by how
much. The detail window is always actual size.

`preview marks` trades time for a fit view closer to true scale: at 21in and
0.225 density, 26,000 marks gives 2.9x magnification and 100,000 gives 1.5x.
It is a view setting and never reaches the file, so it is not saved in presets.

Raising `max density` to make the preview look denser works against this -- it
puts more ink on the preview but magnifies the grain further, 5.3x at 0.75 --
and it does change the export.

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
             filters · spatial   (numpy stand-ins for the scipy calls)
web/         server · engine · shell · index.html · app.css · app.js
browser/     worker · bridge · shell · build
presets/     tests/
```

`web/app.js` is the interface for both builds. It reaches everything outside
the page through `STIPPLE_SHELL`, which is an HTTP server on the desktop and a
Pyodide worker in the browser.

## Licence

MIT. See `LICENSE`.
