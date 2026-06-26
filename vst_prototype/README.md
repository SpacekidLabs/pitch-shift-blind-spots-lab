# Pitch Shift Blind Spots VST Prototype

Adaptive pitch shifter prototype for the Pitch Shift Blind Spots Lab.

The product idea is simple:

```text
do not trust one pitch-shifting strategy everywhere
classify the signal first
choose the safest behavior
fall back when the render looks unhealthy
```

This folder starts the real-time plugin path. It is not a finished pitch shifter yet. v0.2.0 contains the plugin shell, parameters, live signal-state analysis, adaptive strategy selection, manual observer/backend selection, a built-in Phase Vocoder backend, WSOLA-lite/PSOLA-lite prototype paths, a Rubber Band adapter slot, and block-level render-health rescue.

## Current Scope

Implemented:

- JUCE VST3 scaffold
- adaptive selector engine in pure C++
- signal-state vocabulary from the research experiments
- observer-disagreement proxy
- safe-mode routing rules
- adaptive router with separate intended strategy and active backend readouts
- manual backend mode: Adaptive, Phase Vocoder, WSOLA-lite, PSOLA-lite, Rubber Band slot, Bypass
- built-in real-time Phase Vocoder fallback backend
- WSOLA-lite and PSOLA-lite prototype paths based on the current overlap engine
- Rubber Band slot that reports unavailable until the SDK/library is embedded
- block-level health rescue if the Phase Vocoder fallback collapses
- basic plugin UI showing current state and chosen strategy
- standalone smoke test for the adaptive engine

Not implemented yet:

- Rubber Band SDK integration
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

Rubber Band is represented as a real routing slot, but it is not a real Rubber Band backend until the Rubber Band SDK/library is embedded. In manual `Rubber Band slot` mode, the plugin reports that status and uses the available Phase Vocoder fallback.

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
2. Embed the Rubber Band SDK/library and connect the Rubber Band slot to the real backend.
3. Add per-backend health telemetry to the UI.
4. Add a small DAW listening test using the same drum loop.
5. Compare the installed plugin against the Python blind listening artifacts.

The point is not to make the loudest pitch shifter. The point is to make one that knows when not to trust itself.
