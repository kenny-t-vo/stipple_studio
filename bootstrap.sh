#!/bin/bash
# Create the project virtualenv. Run once per machine.
# The venv deliberately lives on the internal drive: this project sits on an
# exFAT volume, where venvs are unreliable and which may be unmounted.
set -euo pipefail

VENV="$HOME/.local/share/stipple/venv"
PY="${PYTHON:-/opt/homebrew/bin/python3.12}"

if [ ! -x "$PY" ]; then
  echo "Python 3.12 not found at $PY" >&2
  echo "Install it with:  brew install python@3.12" >&2
  exit 1
fi

mkdir -p "$(dirname "$VENV")"
[ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q numpy scipy pillow pytest

echo "venv ready: $VENV"
"$VENV/bin/python" -c "import numpy,scipy,PIL,sys;print('python',sys.version.split()[0])"
