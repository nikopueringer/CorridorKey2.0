#!/usr/bin/env bash
# Install the platform-specific CorridorKey bundle without replacing another OFX plugin.
set -euo pipefail
cd "$(dirname "$0")/.."
case "$(uname -s)" in
    Linux)
        destination="${1:-/usr/OFX/Plugins}"
        bundle="build/ofx/CorridorKeyLinux.ofx.bundle"
        bundle_name="CorridorKeyLinux.ofx.bundle"
        test -f "$bundle/Contents/Linux-x86-64/CorridorKeyLinux.ofx" || { echo "Run scripts/build_ofx.sh first."; exit 1; }
        ;;
    Darwin)
        destination="${1:-/Library/OFX/Plugins}"
        bundle="build/ofx-macos/CorridorKeyMacOS.ofx.bundle"
        bundle_name="CorridorKeyMacOS.ofx.bundle"
        test -f "$bundle/Contents/MacOS/CorridorKeyMacOS.ofx" || { echo "Run scripts/build_ofx.sh on Apple Silicon first."; exit 1; }
        ;;
    *) echo "OFX installation is supported on Linux and macOS." >&2; exit 1 ;;
esac
if [ -e "$destination/$bundle_name" ] || [ -L "$destination/$bundle_name" ]; then
    echo "Already installed. Move the existing $bundle_name aside before replacing it."
    exit 1
fi
mkdir -p "$destination"
cp -R "$bundle" "$destination/$bundle_name"
echo "Installed to $destination. Restart Resolve to scan the plugin."
