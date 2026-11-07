#!/bin/bash
# Set up on first run, then start the translation UI. Extra args go to app.py.
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v uv >/dev/null; then
  [ -x /opt/homebrew/bin/uv ] && export PATH="/opt/homebrew/bin:$PATH"
fi
if ! command -v uv >/dev/null; then
  echo "uv が必要です: brew install uv" >&2
  exit 1
fi
uv sync --quiet --no-dev
exec .venv/bin/python app.py "$@"
