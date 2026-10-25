#!/usr/bin/env bash
# CorridorKey 2.0 launcher. Drag a clip or folder onto this file, or run it
# and drop the clip into the terminal when asked.
cd "$(dirname "$0")" || exit 1

if [ -x "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
elif [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

"$PYTHON" run_interactive.py "$@"

echo
read -r -p "Press Enter to close."
