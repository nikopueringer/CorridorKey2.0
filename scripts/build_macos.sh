#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$ROOT/dist"
APP="$DIST/CorridorKey.app"
ZIP="$DIST/CorridorKey-macOS-arm64.zip"
FORCE=0

if [[ "${1:-}" == "--force" ]]; then
    FORCE=1
    shift
fi
if [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--force]" >&2
    exit 2
fi
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "Build this bundle on an Apple Silicon Mac." >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "Install uv first: brew install uv" >&2
    exit 1
fi
if ! command -v xcrun >/dev/null 2>&1; then
    echo "Install Xcode Command Line Tools first: xcode-select --install" >&2
    exit 1
fi
if [[ ! -f "$ROOT/requirements.txt" || ! -f "$ROOT/requirements-gui.txt" ]]; then
    echo "Run this script from a complete CorridorKey checkout." >&2
    exit 1
fi
if [[ ( -e "$APP" || -L "$APP" || -e "$ZIP" || -L "$ZIP" ) && "$FORCE" != 1 ]]; then
    echo "Build output already exists. Move it aside or pass --force: $APP" >&2
    exit 1
fi

mkdir -p "$DIST"
if [[ "$FORCE" == 1 ]]; then
    rm -rf "$APP" "$ZIP"
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/corridorkey-macos-build.XXXXXX")"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

CONTENTS="$APP/Contents"
RESOURCES="$CONTENTS/Resources"
SOURCE="$RESOURCES/src"
PYTHON_DIR="$RESOURCES/Python"
mkdir -p "$CONTENTS/MacOS" "$RESOURCES" "$SOURCE"

echo "Exporting committed source tree..."
git -C "$ROOT" archive --format=tar HEAD | tar -xf - -C "$SOURCE"

echo "Installing managed Python 3.12..."
export UV_PYTHON_INSTALL_DIR="$WORK/uv-python"
uv python install 3.12
PYTHON_BIN="$(uv python find --managed-python 3.12)"
PYTHON_PREFIX="$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)"
if [[ ! -x "$PYTHON_PREFIX/bin/python3" ]]; then
    echo "uv's Python install does not contain bin/python3: $PYTHON_PREFIX" >&2
    exit 1
fi
ditto "$PYTHON_PREFIX" "$PYTHON_DIR"
BUNDLED_PYTHON="$PYTHON_DIR/bin/python3"

echo "Installing PyTorch and GUI runtime dependencies..."
uv pip install --python "$BUNDLED_PYTHON" --system --break-system-packages \
    "torch==2.9.1" "torchvision==0.24.1" \
    -r "$SOURCE/requirements.txt" -r "$SOURCE/requirements-gui.txt"

echo "Compiling native macOS launcher..."
xcrun clang -fobjc-arc -mmacosx-version-min=15.0 -framework Cocoa \
    "$ROOT/packaging/macos/launcher.m" -o "$CONTENTS/MacOS/CorridorKey"
cp "$ROOT/packaging/macos/Info.plist" "$CONTENTS/Info.plist"

echo "Applying ad-hoc bundle signature..."
codesign --force --deep --sign - "$APP"

echo "Creating archive..."
ditto -c -k --sequesterRsrc --keepParent "$APP" "$ZIP"
echo "Built $APP"
echo "Built $ZIP"
