#!/usr/bin/env python3
"""Install Stipple.app -- a double-clickable launcher for the local install.

The bundle is thin: it runs the project venv against the source in this
folder, so edits take effect on the next launch with nothing to rebuild. For
a self-contained bundle that runs on a machine with no Python, use
build_app.py instead.

It installs to ~/Applications, NOT next to the source, and that is not a
matter of taste. macOS will not launch an app bundle from this project's
exFAT volume: the volume is mounted noowners, so the executable has no
verifiable ownership and LaunchServices silently declines. `open` returns
success and nothing happens. Verified by launching identical bundles from
both filesystems -- the APFS copy ran, the exFAT copy did not. The launcher
therefore lives on the internal drive and points back here.
"""

from __future__ import annotations

import plistlib
import shutil
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = Path.home() / ".local" / "share" / "stipple" / "venv"
DEST = Path.home() / "Applications"        # must be on the internal drive


def build(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    app = dest / "Stipple.app"
    if app.exists():
        shutil.rmtree(app)
    macos = app / "Contents" / "MacOS"
    macos.mkdir(parents=True)

    (app / "Contents" / "Info.plist").write_bytes(plistlib.dumps({
        "CFBundleName": "Stipple",
        "CFBundleDisplayName": "Stipple",
        "CFBundleExecutable": "stipple",
        "CFBundleIdentifier": "local.stipple",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "2.0",
        "LSMinimumSystemVersion": "12.0",
        "NSHighResolutionCapable": True,
        # No Dock icon for the helper process; pywebview owns the window.
        "LSUIElement": False,
    }))

    launcher = macos / "stipple"
    launcher.write_text(f"""#!/bin/bash
# Launcher for the local Stipple install.
VENV="{VENV}"
ROOT="{ROOT}"

fail() {{
  /usr/bin/osascript -e "display alert \\"Stipple\\" message \\"$1\\" as critical"
  exit 1
}}

# The project lives on an external volume; say so plainly rather than
# failing with a stack trace when it is unplugged.
[ -d "$ROOT" ] || fail "Cannot find the project folder:
$ROOT

If it lives on an external drive, plug the drive in and try again."

[ -x "$VENV/bin/python" ] || fail "The Python environment is missing.
Run bootstrap.sh in the project folder to create it."

exec "$VENV/bin/python" "$ROOT/web/server.py"
""")
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return app


if __name__ == "__main__":
    dest = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEST
    if dest.resolve().is_relative_to(Path("/Volumes")):
        print(f"warning: {dest} is on an external volume.", file=sys.stderr)
        print("macOS will not launch an app bundle from there; it will do "
              "nothing when opened.", file=sys.stderr)
    app = build(dest)
    print(f"installed {app}")
    print(f"points at  {ROOT}")
    print("\nOpen it from the Finder sidebar (Go > Applications), Spotlight, "
          "or drag it to the Dock.")
