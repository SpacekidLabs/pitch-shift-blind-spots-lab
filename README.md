# Pitch Shift Blind Spots Lab

Mapping the regions of signal space where pitch-shifting algorithms fail.

## Research Questions

- Which signals are difficult for pitch shifters?
- Do different algorithms fail in different ways?
- Which structures survive every representation?
- Can failure be predicted before processing?

## What This Repo Is For

This repository is a research lab for studying pitch-shift failure modes, not a benchmark suite and not a pitch shifter implementation.

The goal is to map where representations break:

- harmonic signals
- inharmonic resonances
- noise-like textures
- transients
- modulated tones

Experiment 01 creates a synthetic stress atlas and runs it through a single pitch-shifting path based on a phase vocoder.

Experiment 02 runs the same atlas through four algorithm families:

- Phase Vocoder
- WSOLA
- Rubber Band
- PSOLA

It asks a more interesting question than "which one wins?":

- Do these algorithms fail on the same structures?

## Repository Layout

- `signals/` synthetic signal generators
- `algorithms/` pitch-shift backends
- `metrics/` distortion and change metrics
- `visualization/` plotting helpers
- `experiments/` runnable studies
- `artifacts/` generated results and figures
- `data/` future source data and notes

## Experiment 01

`experiments/01_stress_signal_atlas.py` generates:

- harmonic: sine, saw, square
- inharmonic: bell resonator, random modal resonator
- noise: white noise, pink noise
- transient: impulse train, click sequence
- modulated: vibrato tone, FM tone

Each signal is shifted by:

- `+3` semitones
- `+7` semitones
- `+12` semitones
- `-12` semitones

The experiment computes:

- RMS error
- spectral distance
- spectral centroid difference

Outputs:

- `artifacts/01_results.csv`
- `artifacts/01_heatmap.png`

## Experiment 02

`experiments/02_algorithm_comparison.py` reuses the exact same synthetic atlas and metrics, then compares:

- Phase Vocoder
- WSOLA
- Rubber Band
- PSOLA

Outputs:

- `artifacts/02_results.csv`
- `artifacts/02_heatmap.png`

The comparison figure is arranged by algorithm and metric so the blind spots are easier to compare side by side.

## Roadmap

See [`ROADMAP.md`](ROADMAP.md) for the next steps, including comparisons against WSOLA, PSOLA, Rubber Band, and neural pitch shifters.
