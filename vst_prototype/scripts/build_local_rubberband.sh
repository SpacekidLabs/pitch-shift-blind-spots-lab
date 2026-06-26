#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${SRC_DIR:-/private/tmp/psbsl_rubberband_src}"
INSTALL_ROOT="${RUBBERBAND_ROOT:-$ROOT_DIR/local/rubberband}"
REPO_URL="${RUBBERBAND_REPO_URL:-https://github.com/breakfastquay/rubberband.git}"

rm -rf "$SRC_DIR"
git clone --depth 1 "$REPO_URL" "$SRC_DIR"
make -C "$SRC_DIR" -f otherbuilds/Makefile.macos static

mkdir -p "$INSTALL_ROOT/include/rubberband" "$INSTALL_ROOT/lib"
cp "$SRC_DIR/rubberband/RubberBandStretcher.h" "$SRC_DIR/rubberband/rubberband-c.h" "$INSTALL_ROOT/include/rubberband/"
cp "$SRC_DIR/lib/librubberband.a" "$INSTALL_ROOT/lib/"

echo "Installed local Rubber Band SDK:"
echo "$INSTALL_ROOT"
echo
echo "Rebuild the VST with:"
echo "vst_prototype/scripts/build_and_install_vst3.sh"
