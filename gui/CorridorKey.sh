#!/usr/bin/env bash
# CorridorKey 2.0 desktop UI. Run it, or drag clips or folders onto it.
cd "$(dirname "$0")/.." || exit 1

if [ -x "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
else
    PYTHON="python3"
fi

if ! "$PYTHON" -c "import PySide6" 2>/dev/null; then
    echo "Run scripts/setup_linux.sh first to install the GUI in an isolated environment."
    exit 1
fi

exec "$PYTHON" -m gui "$@"
