#!/usr/bin/env bash
# Use the per-user OFX install without requiring a system-wide installation.
set -euo pipefail
export OFX_PLUGIN_PATH="${CORRIDORKEY_OFX_DIR:-$HOME/.local/lib/OFX/Plugins}${OFX_PLUGIN_PATH:+:$OFX_PLUGIN_PATH}"
RESOLVE_EXECUTABLE="${CORRIDORKEY_RESOLVE_BIN:-/opt/resolve/bin/resolve}"
if [[ ! -x "$RESOLVE_EXECUTABLE" ]]; then
    echo "Resolve was not found at $RESOLVE_EXECUTABLE. Set CORRIDORKEY_RESOLVE_BIN to its executable." >&2
    exit 1
fi
exec "$RESOLVE_EXECUTABLE" "$@"
