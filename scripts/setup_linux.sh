#!/usr/bin/env bash
# Installs only into the repository venv. Requires uv and an NVIDIA driver.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v uv >/dev/null; then
    echo "Install uv first (python3 -m pip install --user uv), then re-run this script."
    exit 1
fi
command -v ffmpeg >/dev/null || { echo "Install ffmpeg with your package manager first."; exit 1; }
uv venv --python 3.12 --allow-existing venv
uv pip install --python venv/bin/python torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python venv/bin/python -r requirements.txt -r requirements-gui.txt
venv/bin/python scripts/download_weights.py
echo "Ready. Launch with ./gui/CorridorKey.sh"
