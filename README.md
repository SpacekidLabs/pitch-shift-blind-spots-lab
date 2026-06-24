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

Experiment 04 tests the Ambiguity Hypothesis directly:

- Does algorithm disagreement increase as pitch ambiguity increases?

Experiment 05 zooms in around the observed ambiguity peaks and asks whether disagreement behaves like a critical boundary between one pitch and obviously multiple pitches.

Experiment 06 asks what pitch trajectory each observer appears to infer over time.

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

## Experiment 04

`experiments/04_ambiguity_sweep.py` tests whether the discovery result generalizes to controlled ambiguity sweeps.

Sweep families:

- two oscillators: `440 + 440` through `440 + 480`
- three oscillators: close, medium, and wide competing pitch sets
- vibrato depth: `0`, `1`, `2`, `4`, `8`, `12` semitones
- beating rate: `0.5`, `1`, `2`, `5`, `10` Hz

Question:

```text
Does algorithm disagreement increase as pitch ambiguity increases?
```

Outputs:

- `artifacts/04_results.csv`
- `artifacts/04_disagreement_landscape.csv`
- `artifacts/04_ambiguity_summary.csv`
- `artifacts/04_family_trends.csv`
- `artifacts/04_ambiguity_plot.png`

This experiment treats pitch ambiguity as the independent variable and algorithm disagreement as the measured response.

## Experiment 05

`experiments/05_critical_ambiguity_zoom.py` focuses tightly around the Experiment 04 peaks.

Fine sweeps:

- two oscillators: `440 + 440.0` through `440 + 443`
- vibrato depth: `0` through `2` semitones
- beating rate: `0.1` through `5` Hz

Additional measurements:

- `peak_disagreement_location`
- `peak_width` above 90% of peak disagreement
- `half_max_width` above 50% of peak disagreement

Outputs:

- `artifacts/05_results.csv`
- `artifacts/05_disagreement_landscape.csv`
- `artifacts/05_ambiguity_summary.csv`
- `artifacts/05_peak_metrics.csv`
- `artifacts/05_critical_ambiguity_plot.png`

This experiment tests whether the most interesting signals live near the boundary between a single pitch interpretation and clearly competing pitch interpretations.

## Experiment 06

`experiments/06_observer_state_space.py` moves from scalar disagreement to observer trajectories.

For the critical ambiguity peak cases, it estimates framewise pitch from each algorithm output:

- Phase Vocoder: `f(t)`
- PSOLA: `f(t)`
- WSOLA: `f(t)`
- Rubber Band: `f(t)`

Because these local algorithm implementations do not expose private internal state, Experiment 06 uses an explicit proxy: framewise autocorrelation `f0(t)` estimated from each observer's output and mapped back through the requested pitch shift.

Questions:

- Do observers lock to different trajectories?
- Do they jump between attractors?
- Do they oscillate?
- Do they collapse onto the same solution?

Outputs:

- `artifacts/06_pitch_trajectories.csv`
- `artifacts/06_trajectory_summary.csv`
- `artifacts/06_case_state_summary.csv`
- `artifacts/06_observer_state_space.png`

This experiment starts treating pitch shifters as observers with inferred state trajectories, not just processors with error scores.

## Roadmap

See [`ROADMAP.md`](ROADMAP.md) for the next steps, including deeper blind spot taxonomy and neural pitch shifter comparisons.
