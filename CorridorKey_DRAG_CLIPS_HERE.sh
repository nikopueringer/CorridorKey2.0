#!/usr/bin/env bash
# CorridorKey 2.0 launcher. Drag a clip or folder onto this file, or run it
# and drop the clip into the terminal when asked.
cd "$(dirname "$0")" || exit 1

if command -v uv >/dev/null 2>&1; then
    uv run run_interactive.py "$@"
elif [ -x ".venv/bin/python" ]; then
    .venv/bin/python run_interactive.py "$@"
elif [ -x "venv/bin/python" ]; then
    venv/bin/python run_interactive.py "$@"
else
    python3 run_interactive.py "$@"
fi

echo
read -r -p "Press Enter to close."
