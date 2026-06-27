# Pitch Shift Blind Spots VST Prototype

Adaptive pitch shifter prototype for the Pitch Shift Blind Spots Lab.

The product idea is simple:

```text
do not trust one pitch-shifting strategy everywhere
classify the signal first
choose the safest behavior
fall back when the render looks unhealthy
```

This folder starts the real-time plugin path. It is not a finished pitch shifter yet. v0.3.2 contains the plugin shell, parameters, live signal-state analysis, adaptive strategy selection, manual observer/backend selection, a built-in Phase Vocoder backend, WSOLA-lite/PSOLA-lite prototype paths, a real Rubber Band backend when the SDK/library is available, block-level render-health rescue, and adaptive-router smoothing to prevent rapid mode chatter.

## Current Scope

Implemented:

- JUCE VST3 scaffold
- adaptive selector engine in pure C++
- signal-state vocabulary from the research experiments
- observer-disagreement proxy
- safe-mode routing rules
- adaptive router with separate intended strategy and active backend readouts
- adaptive-router hysteresis so Adaptive mode holds a backend before trusting a new decision
- manual backend mode: Adaptive, Phase Vocoder, WSOLA-lite, PSOLA-lite, Rubber Band slot, Bypass
- built-in real-time Phase Vocoder fallback backend
- WSOLA-lite and PSOLA-lite prototype paths based on the current overlap engine
- real Rubber Band backend when `RUBBERBAND_ROOT` points at a local install
- honest Rubber Band fallback status when the SDK/library is missing
- block-level health rescue if the Phase Vocoder fallback collapses
- basic plugin UI showing current state and chosen strategy
- standalone smoke test for the adaptive engine

Not implemented yet:

- PSOLA/WSOLA real-time backends
- post-render health gate for streaming audio
- artifact repair or transient reconstruction

## Architecture

```text
audio input
  |
  v
block feature analysis
  |
  v
signal-state classifier
  |
  v
adaptive strategy selector
  |
  v
available backend fallback
  |
  v
safe-mode dry/wet and shift guards
  |
  v
audio output
```

## Signal States

The first selector recognizes the same states used by the research lab:

- `stable_periodic`
- `octave_ambiguous`
- `subharmonic_ambiguous`
- `detuned_competing`
- `noise_like`
- `transient_like`
- `modulated_pitch`
- `untracked`

## Strategy Rules v0

- stable periodic signals prefer Rubber Band as a practical high-quality path
- octave and subharmonic ambiguity avoid PSOLA and choose Phase Vocoder
- detuned competing signals reduce shift strength and prefer Rubber Band
- noise-like signals use spectral fallback with reduced wet level
- transient-like signals prefer WSOLA, unless safe mode redirects them
- high observer disagreement enters safe mode

The UI shows both:

- `Strategy`: what the adaptive observer wanted
- `Active backend`: what the installed prototype actually used

If the Phase Vocoder fallback produces a collapsed block, the plugin temporarily rescues that block with WSOLA-lite and lowers the wet blend. This is the real-time version of the post-render health gate discovered in Experiments 24 and 25.

Adaptive mode also has a small anti-chatter guard. The observer can still change its mind, but the audio renderer waits for a new backend choice to remain stable for a few blocks, holds each selected backend briefly, and softens the dry/wet blend during backend transitions. Manual backend modes remain immediate for debugging.

On this machine, Rubber Band was built locally from source into:

```text
vst_prototype/local/rubberband
```

That folder is intentionally ignored by git. The build scripts pass it to CMake as `RUBBERBAND_ROOT`, and the installed plugin statically links the Rubber Band backend. If that local folder is missing on another machine, the plugin still builds and clearly reports that Rubber Band is unavailable.

Safe mode currently:

- avoids PSOLA and WSOLA for high-disagreement blocks
- reduces wet level
- reduces shift strength

## Build: Engine Smoke Test

The adaptive engine can be compiled without JUCE:

```sh
vst_prototype/scripts/test_engine.sh
```

## Build: VST3 Plugin

This repo can build against a local JUCE checkout. On this machine the default script uses:

```text
/Applications/CMake.app/Contents/bin/cmake
/Users/user/Desktop/wavsynth/JUCE
```

Build and install into the user VST3 folder:

```sh
vst_prototype/scripts/build_local_rubberband.sh
vst_prototype/scripts/build_and_install_vst3.sh
```

The installed plugin path is:

```text
~/Library/Audio/Plug-Ins/VST3/Pitch Shift Blind Spots.vst3
```

If your DAW is open, restart it or rescan plugins.

You can override local paths:

```sh
CMAKE_BIN=/path/to/cmake \
JUCE_SOURCE_DIR=/path/to/JUCE \
vst_prototype/scripts/build_and_install_vst3.sh
```

The plugin target is:

```text
Pitch Shift Blind Spots.vst3
```

## Next Backend Step

The next useful milestone is not fancy UI. It is a real backend slot:

1. Replace WSOLA-lite and PSOLA-lite with stronger real implementations.
2. Add per-backend health telemetry to the UI.
3. Add a small DAW listening test using the same drum loop.
4. Compare the installed plugin against the Python blind listening artifacts.
5. Decide whether Rubber Band should remain a local SDK dependency or become a documented external install step.

The point is not to make the loudest pitch shifter. The point is to make one that knows when not to trust itself.
