#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec venv/bin/python -u -m ofx.worker "$@"
