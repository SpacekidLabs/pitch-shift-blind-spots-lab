#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMAKE_BIN="${CMAKE_BIN:-/Applications/CMake.app/Contents/bin/cmake}"
JUCE_SOURCE_DIR="${JUCE_SOURCE_DIR:-/Users/user/Desktop/wavsynth/JUCE}"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build/plugin}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/Library/Audio/Plug-Ins/VST3}"
CONFIG="${CONFIG:-Release}"

if [[ ! -x "$CMAKE_BIN" ]]; then
  echo "CMake was not found at: $CMAKE_BIN"
  echo "Set CMAKE_BIN to your cmake binary path and rerun."
  exit 1
fi

if [[ ! -f "$JUCE_SOURCE_DIR/CMakeLists.txt" ]]; then
  echo "JUCE source was not found at: $JUCE_SOURCE_DIR"
  echo "Set JUCE_SOURCE_DIR to your JUCE checkout and rerun."
  exit 1
fi

"$CMAKE_BIN" -S "$ROOT_DIR" -B "$BUILD_DIR" \
  -DPSBSL_BUILD_PLUGIN=ON \
  -DPSBSL_BUILD_ENGINE_SMOKE_TEST=ON \
  -DJUCE_SOURCE_DIR="$JUCE_SOURCE_DIR"

"$CMAKE_BIN" --build "$BUILD_DIR" --config "$CONFIG"

PLUGIN_PATH="$(find "$BUILD_DIR" -type d -name 'Pitch Shift Blind Spots.vst3' | head -n 1)"
if [[ -z "$PLUGIN_PATH" ]]; then
  echo "Build finished, but no VST3 bundle was found under: $BUILD_DIR"
  exit 1
fi

mkdir -p "$INSTALL_DIR"
rm -rf "$INSTALL_DIR/Pitch Shift Blind Spots.vst3"
cp -R "$PLUGIN_PATH" "$INSTALL_DIR/"

echo "Installed:"
echo "$INSTALL_DIR/Pitch Shift Blind Spots.vst3"
echo
echo "If your DAW is open, restart it or rescan plugins."
