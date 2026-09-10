#!/usr/bin/env python3
"""Assemble the static browser build into dist/.

Everything the page needs is copied or generated here, so dist/ can be served
by anything that serves files. The frontend is the desktop's, unmodified: only
the shell differs, and index.html is rewritten to load the browser one and to
use relative paths, since GitHub Pages serves the site from a subdirectory.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "browser"
DIST = ROOT / "dist"


def build() -> Path:
    if DIST.exists():
        # Sidecars first: deleting a file on exFAT takes its ._ partner with
        # it, and rmtree then trips over the entry it already listed.
        for junk in DIST.rglob("._*"):
            junk.unlink(missing_ok=True)
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)

    for src in (ROOT / "web" / "app.css", ROOT / "web" / "type.css",
                ROOT / "web" / "app.js", HERE / "shell.js", HERE / "worker.js"):
        shutil.copyfile(src, DIST / src.name)

    # type.css reaches the display face at fonts/, relative to itself.
    fonts = DIST / "fonts"
    fonts.mkdir()
    for src in sorted((ROOT / "web" / "fonts").iterdir()):
        if not src.name.startswith("._"):
            shutil.copyfile(src, fonts / src.name)

    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    html = html.replace('href="/type.css"', 'href="type.css"')
    html = html.replace('href="/app.css"', 'href="app.css"')
    html = html.replace('src="/shell.js"', 'src="shell.js"')
    html = html.replace('src="/app.js"', 'src="app.js"')
    assert 'src="/' not in html and 'href="/' not in html, "absolute path left in index.html"
    (DIST / "index.html").write_text(html, encoding="utf-8")

    # The python the worker unpacks. bridge.py and engine.py sit at the root of
    # the archive because bridge imports engine by name.
    zip_path = DIST / "stipple-src.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for py in sorted((ROOT / "stipple").glob("*.py")):
            z.write(py, f"stipple/{py.name}")
        z.write(ROOT / "web" / "engine.py", "engine.py")
        z.write(HERE / "bridge.py", "bridge.py")

    # exFAT: macOS materialises extended attributes as ._ sidecars.
    for junk in DIST.rglob("._*"):
        junk.unlink()

    total = sum(f.stat().st_size for f in DIST.rglob("*") if f.is_file())
    print(f"dist/  {len(list(DIST.rglob('*')))} files, {total/1024:.0f} KB")
    for f in sorted(DIST.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(DIST)!s:22s} {f.stat().st_size/1024:7.1f} KB")
    return DIST


if __name__ == "__main__":
    build()
