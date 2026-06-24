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

Experiment 02 runs the same atlas through four algorithm families and measures where they disagree:

- Phase Vocoder
- WSOLA
- Rubber Band
- PSOLA

It asks:

- Which structures maximize disagreement between pitch-shifting algorithms?

Experiment 03 turns that disagreement score into a search objective and asks what generated signal maximizes it.

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

`experiments/02_algorithm_disagreement_landscape.py` reuses the exact same synthetic atlas and metrics, then computes:

```text
algorithm_disagreement = variance(composite_stress across algorithms)
```

The point is not to crown the best pitch shifter. The point is to find the signals that expose incompatible assumptions across representations.

Algorithms:

- Phase Vocoder
- WSOLA
- Rubber Band
- PSOLA

Outputs:

- `artifacts/02_results.csv`
- `artifacts/02_disagreement_landscape.csv`
- `artifacts/02_disagreement_summary.csv`
- `artifacts/02_heatmap.png`

The signals with the highest disagreement are the blind spot candidates for deeper analysis.

## Experiment 03

`experiments/03_blind_spot_discovery.py` moves from hand-designed atlas entries to generated candidate signals.

Candidate families:

- detuned oscillators
- beating oscillators
- chirps
- glissandi
- chaotic oscillators
- quasi-periodic signals
- modal banks
- noise mixtures

Objective:

```text
maximize variance(composite_stress across algorithms)
```

Outputs:

- `artifacts/03_candidates.csv`
- `artifacts/03_results.csv`
- `artifacts/03_disagreement_landscape.csv`
- `artifacts/03_discovery_summary.csv`
- `artifacts/03_best_signal.csv`
- `artifacts/03_heatmap.png`

This experiment asks what signal maximizes disagreement between pitch shifters, leaving room for structures that were not anticipated by the hand-designed atlas.

## Roadmap

See [`ROADMAP.md`](ROADMAP.md) for the next steps, including deeper blind spot taxonomy and neural pitch shifter comparisons.
