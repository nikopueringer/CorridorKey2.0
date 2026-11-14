#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
case "$(uname -s)" in
    Linux)
        build_dir="build/ofx"
        bundle="CorridorKeyLinux.ofx.bundle"
        ;;
    Darwin)
        [[ "$(uname -m)" == arm64 ]] || { echo "The macOS OFX build requires Apple Silicon." >&2; exit 1; }
        build_dir="build/ofx-macos"
        bundle="CorridorKeyMacOS.ofx.bundle"
        ;;
    *) echo "OFX builds are supported on Linux x86-64 and macOS arm64." >&2; exit 1 ;;
esac
cmake -S ofx -B "$build_dir" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$build_dir" --parallel
ctest --test-dir "$build_dir" --output-on-failure
echo "Built: $build_dir/$bundle"
