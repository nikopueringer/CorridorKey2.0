#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cmake -S ofx -B build/ofx -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build build/ofx --parallel
ctest --test-dir build/ofx --output-on-failure
echo "Built: build/ofx/CorridorKeyLinux.ofx.bundle"
