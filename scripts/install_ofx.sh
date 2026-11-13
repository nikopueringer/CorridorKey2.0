#!/usr/bin/env bash
# Resolve on Linux scans /usr/OFX/Plugins. Run with sudo for that destination.
set -euo pipefail
cd "$(dirname "$0")/.."
destination="${1:-/usr/OFX/Plugins}"
bundle="build/ofx/CorridorKeyLinux.ofx.bundle"
test -f "$bundle/Contents/Linux-x86-64/CorridorKeyLinux.ofx" || { echo "Run scripts/build_ofx.sh first."; exit 1; }
if [ -e "$destination/CorridorKeyLinux.ofx.bundle" ]; then
    echo "Already installed. Move the previous CorridorKeyLinux bundle aside before replacing it."
    exit 1
fi
mkdir -p "$destination"
cp -R "$bundle" "$destination/"
echo "Installed to $destination. Restart Resolve to scan the plugin."
