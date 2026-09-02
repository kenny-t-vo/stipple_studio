#!/usr/bin/env python3
"""Build a self-contained Stipple.app with PyInstaller.

For a bundle that runs on another Mac with no Python installed. Takes a
minute or two and must be rerun after code changes; for everyday work use
make_launcher.py, which points at this source tree.

Apple Silicon only. The bundle is unsigned, so macOS refuses the first
launch on another machine until it is opened once via right-click -> Open.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = Path.home() / ".local" / "share" / "stipple" / "venv"
PY_BIN = VENV / "bin" / "python"


def main() -> int:
    if not PY_BIN.exists():
        print("venv missing -- run bootstrap.sh first", file=sys.stderr)
        return 1

    if subprocess.run([str(PY_BIN), "-c", "import PyInstaller"],
                      capture_output=True).returncode != 0:
        print("installing pyinstaller into the project venv…")
        subprocess.run([str(VENV / "bin" / "pip"), "install", "-q", "pyinstaller"],
                       check=True)

    for d in ("build", "dist"):
        shutil.rmtree(ROOT / d, ignore_errors=True)

    cmd = [
        str(PY_BIN), "-m", "PyInstaller",
        "--name", "Stipple",
        "--windowed",                       # .app bundle, no terminal
        "--noconfirm", "--clean",
        "--target-architecture", "arm64",
        # The UI is read from disk at request time, so it has to ship.
        "--add-data", f"{ROOT / 'web'}:web",
        "--add-data", f"{ROOT / 'presets'}:presets",
        "--paths", str(ROOT),
        "--paths", str(ROOT / "web"),
        "--hidden-import", "scipy.spatial",
        "--hidden-import", "scipy.ndimage",
        "--osx-bundle-identifier", "local.stipple.bundled",
        str(ROOT / "web" / "server.py"),
    ]
    print(" ".join(cmd[:6]), "…")
    if subprocess.run(cmd, cwd=ROOT).returncode != 0:
        return 1

    app = ROOT / "dist" / "Stipple.app"
    print(f"\nbuilt {app}")
    print("Unsigned, so on another Mac the first launch needs right-click -> Open.")
    print("\nNote: this bundle cannot be launched from where it was just built.")
    print("macOS will not run an app bundle from this exFAT volume -- it does")
    print("nothing when opened. Copy it to the internal drive, or to the other")
    print("Mac, before launching:")
    print(f"  cp -R '{app}' ~/Applications/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
