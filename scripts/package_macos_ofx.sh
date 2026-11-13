#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$ROOT/dist"
ZIP="$DIST/CorridorKeyMacOS-OFX-arm64.zip"
FORCE=0

if [[ "${1:-}" == "--force" ]]; then FORCE=1; shift; fi
if [[ $# -ne 0 ]]; then echo "Usage: $0 [--force]" >&2; exit 2; fi
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
    echo "Package the OFX plugin on an Apple Silicon Mac." >&2
    exit 1
fi
if ! command -v xcrun >/dev/null 2>&1 || ! command -v codesign >/dev/null 2>&1; then
    echo "Install Xcode Command Line Tools first: xcode-select --install" >&2
    exit 1
fi
OFX_INCLUDE_DIR="${OFX_INCLUDE_DIR:-/opt/resolve/Developer/OpenFX/OpenFX-1.4/include}"
if [[ ! -f "$OFX_INCLUDE_DIR/ofxImageEffect.h" ]]; then
    echo "Set OFX_INCLUDE_DIR to a directory containing OpenFX SDK headers." >&2
    exit 1
fi
if [[ -e "$ZIP" || -L "$ZIP" ]]; then
    if [[ "$FORCE" != 1 ]]; then echo "Archive already exists; pass --force to replace it: $ZIP" >&2; exit 1; fi
    rm -f "$ZIP"
fi

mkdir -p "$DIST"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/corridorkey-ofx-macos.XXXXXX")"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT
BUILD="$WORK/build"
SOURCE="$WORK/source"
BUNDLE="CorridorKeyMacOS.ofx.bundle"
mkdir -p "$SOURCE"
echo "Exporting committed source tree..."
git -C "$ROOT" archive --format=tar HEAD | tar -xf - -C "$SOURCE"

echo "Building and testing the macOS OFX bridge..."
cmake -S "$SOURCE/ofx" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DOFX_INCLUDE_DIR="$OFX_INCLUDE_DIR"
cmake --build "$BUILD" --parallel
ctest --test-dir "$BUILD" --output-on-failure
test -f "$BUILD/$BUNDLE/Contents/MacOS/CorridorKeyMacOS.ofx" || {
    echo "The built plugin executable does not match the OFX bundle basename." >&2
    exit 1
}

PACKAGE="$WORK/CorridorKeyMacOS-OFX"
mkdir -p "$PACKAGE"
ditto "$BUILD/$BUNDLE" "$PACKAGE/$BUNDLE"
codesign --force --deep --sign - "$PACKAGE/$BUNDLE"
codesign --verify --deep --strict "$PACKAGE/$BUNDLE"
cp "$SOURCE/LICENSE" "$SOURCE/THIRD_PARTY_NOTICES.md" "$SOURCE/ofx/OPENFX_LICENSE" "$PACKAGE/"
cat > "$PACKAGE/README.txt" <<'EOF'
CorridorKey 2.0 Community OFX for macOS 15+ (Apple Silicon)

Run Install OFX.command from Finder or Terminal. It verifies the ad-hoc
signature, then uses sudo to install CorridorKeyMacOS.ofx.bundle into
/Library/OFX/Plugins. Restart DaVinci Resolve after installation.

This plugin requires the matching CorridorKey.app. Install and open the app,
download its model through the app, install FFmpeg with Homebrew, and start the
OFX engine in the app before rendering. The plugin keys one frame at a time;
use the app's render queue for temporal processing and hints. It does not
contain or start the inference engine by itself.

Apple Silicon packaging is provided, but full model inference inside DaVinci
Resolve on a Mac has not yet been validated. This plugin is ad-hoc signed;
signature verification does not guarantee Resolve will load it. If macOS
blocks the app or plugin, use Apple's per-app approval steps:
https://support.apple.com/en-us/102445
EOF
cat > "$PACKAGE/Install OFX.command" <<'EOF'
#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="/Library/OFX/Plugins"
BUNDLE="CorridorKeyMacOS.ofx.bundle"
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
    echo "This OFX bundle requires macOS on Apple Silicon." >&2
    read -r -p "Press Return to close."
    exit 1
fi
if [[ ! -f "$HERE/$BUNDLE/Contents/MacOS/CorridorKeyMacOS.ofx" ]]; then
    echo "The CorridorKey OFX bundle is missing next to this installer." >&2
    read -r -p "Press Return to close."
    exit 1
fi
if ! /usr/bin/codesign --verify --deep --strict "$HERE/$BUNDLE"; then
    echo "The plugin signature check failed; refusing to install it." >&2
    read -r -p "Press Return to close."
    exit 1
fi
if [[ -e "$DEST/$BUNDLE" || -L "$DEST/$BUNDLE" ]]; then
    echo "CorridorKey OFX is already installed at: $DEST/$BUNDLE"
    echo "Move that bundle aside manually if you want to replace it."
    read -r -p "Press Return to close."
    exit 1
fi
sudo -v
sudo mkdir -p "$DEST"
sudo ditto "$HERE/$BUNDLE" "$DEST/$BUNDLE"
echo "Installed CorridorKey OFX to $DEST. Restart DaVinci Resolve."
echo "Also open CorridorKey.app and start its OFX engine before rendering."
read -r -p "Press Return to close."
EOF
chmod +x "$PACKAGE/Install OFX.command"
ditto -c -k --sequesterRsrc --keepParent "$PACKAGE" "$ZIP"
echo "Built $ZIP"
