#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMAKE_BIN="${CMAKE_BIN:-/Applications/CMake.app/Contents/bin/cmake}"
JUCE_SOURCE_DIR="${JUCE_SOURCE_DIR:-/Users/user/Desktop/wavsynth/JUCE}"
RUBBERBAND_ROOT="${RUBBERBAND_ROOT:-$ROOT_DIR/local/rubberband}"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build/engine}"

if [[ ! -x "$CMAKE_BIN" ]]; then
  echo "CMake was not found at: $CMAKE_BIN"
  echo "Set CMAKE_BIN to your cmake binary path and rerun."
  exit 1
fi

"$CMAKE_BIN" -S "$ROOT_DIR" -B "$BUILD_DIR" -DPSBSL_BUILD_PLUGIN=OFF -DJUCE_SOURCE_DIR="$JUCE_SOURCE_DIR" -DRUBBERBAND_ROOT="$RUBBERBAND_ROOT"
"$CMAKE_BIN" --build "$BUILD_DIR"
"$BUILD_DIR/psbsl_engine_smoke_test"
