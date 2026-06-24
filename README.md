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

Experiment 07 maps attractor branches across a fine ambiguity sweep and treats state disagreement as the phenomenon itself.

Experiment 08 tests the Observer Bias Hypothesis directly: every pitch shifter carries an implicit pitch prior.

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

## Experiment 07

`experiments/07_attractor_basin_mapping.py` maps observer attractor branches for:

```text
440 + delta
delta = 0.0, 0.1, 0.2, ..., 5.0 Hz
```

For every algorithm, it records the dominant inferred pitch and classifies the trajectory:

- `lower_branch`
- `middle_branch`
- `upper_branch`
- `mixed_branch`

It also computes:

```text
variance_inferred_state = variance(dominant inferred pitch across algorithms)
```

Outputs:

- `artifacts/07_pitch_trajectories.csv`
- `artifacts/07_branch_map.csv`
- `artifacts/07_state_disagreement.csv`
- `artifacts/07_branch_switches.csv`
- `artifacts/07_hysteresis_note.csv`
- `artifacts/07_attractor_basin_map.png`

This experiment focuses on where observer branches switch, split, or collapse. True hysteresis is not claimed yet because the current algorithm wrappers are stateless batch processors.

## Experiment 08

`experiments/08_octave_preference_map.py` investigates octave ambiguity directly.

Signals:

- pure tones: `110`, `220`, `440`, `880`
- octave mixtures: `220 + 440`, `440 + 880`, `220 + 440 + 880`
- missing fundamental: `880 + 1320 + 1760`
- subharmonics: `440 + 220`, `440 + 110`
- detuned octaves: `440 + 875`, `440 + 885`

For each algorithm, it asks:

```text
what_pitch_does_this_observer_believe()
```

It classifies inferred state as:

- `subharmonic_seeking`
- `fundamental_seeking`
- `octave_seeking`
- `upper_partial_seeking`
- `mixed`
- `untracked`

Outputs:

- `artifacts/08_pitch_trajectories.csv`
- `artifacts/08_octave_preference_map.csv`
- `artifacts/08_algorithm_bias_summary.csv`
- `artifacts/08_case_belief_summary.csv`
- `artifacts/08_octave_preference_map.png`

This experiment treats state disagreement as the phenomenon itself, and stress as only one symptom of observer bias.

## Experiment 09

`experiments/09_adaptive_selector_prototype.py` is the first adaptive pitch-shifter prototype.

Instead of using one strategy everywhere, it follows this v0 architecture:

```text
input audio
observer analysis
state classifier
adaptive algorithm selector
pitch shift
artifact guard
output
```

Signal states:

- `stable_periodic`
- `octave_ambiguous`
- `subharmonic_ambiguous`
- `detuned_competing`
- `noise_like`
- `transient_like`
- `modulated_pitch`
- `untracked`

The selector includes an observer disagreement detector:

```text
state_disagreement = variance(inferred pitch across observers)
```

When disagreement is high, it enters safe mode:

- avoid PSOLA
- prefer a conservative Phase Vocoder or Rubber Band fallback
- optionally blend a small amount of dry signal as an artifact guard

Experiment 09 compares:

- fixed Phase Vocoder
- fixed PSOLA
- fixed WSOLA
- fixed Rubber Band
- `Adaptive Selector v0`

Outputs:

- `artifacts/09_results.csv`
- `artifacts/09_selector_decisions.csv`
- `artifacts/09_algorithm_summary.csv`
- `artifacts/09_signal_summary.csv`
- `artifacts/09_adaptive_selector_map.png`

The success criterion is not winning every row. The goal is avoiding catastrophic failures. In the current atlas run, `Adaptive Selector v0` has zero catastrophic cases under the fixed-algorithm stress threshold, while still not being the lowest-mean-stress method overall.

## Experiment 10

`experiments/10_preflight_failure_prediction.py` asks whether catastrophic pitch-shift risk can be predicted before processing.

Experiment 09 used observer probes from all pitch shifters before selecting an algorithm. That is useful for research, but expensive for a real adaptive shifter. Experiment 10 uses only source-audio features plus shift size:

- pitch tracking validity
- pitch stability
- spectral flatness
- spectral centroid
- spectral bandwidth
- transient score
- crest factor
- zero-crossing rate
- shift amount

It predicts whether any fixed algorithm is likely to cross the catastrophic stress threshold.

Outputs:

- `artifacts/10_preflight_features.csv`
- `artifacts/10_preflight_risk_predictions.csv`
- `artifacts/10_preflight_evaluation.csv`
- `artifacts/10_preflight_signal_summary.csv`
- `artifacts/10_preflight_risk_map.png`

Current result:

- recall: `0.947`
- precision: `0.429`
- false negatives: `1`

The predictor is intentionally conservative. It catches almost every catastrophic fixed-algorithm case, but over-flags many strange signals. The one miss is revealing: low-depth `vibrato_tone` at `+3 semitones` looks stable in source-only features, but still breaks a fixed observer. That suggests subtle modulation needs a better preflight descriptor.

## Experiment 11

`experiments/11_subtle_modulation_trap.py` zooms in on the false negative from Experiment 10.

It sweeps vibrato tones across:

- depths: `0`, `0.025`, `0.05`, `0.1`, `0.2`, `0.35`, `0.5`, `0.75`, `1.0` semitones
- rates: `2`, `4`, `5.5`, `8` Hz
- shifts: `+3`, `+7`, `+12`, `-12` semitones

It compares fixed Phase Vocoder, PSOLA, WSOLA, and Rubber Band, then asks:

```text
Where does subtle modulation masquerade as stable periodicity?
```

Outputs:

- `artifacts/11_results.csv`
- `artifacts/11_modulation_cases.csv`
- `artifacts/11_modulation_trap_summary.csv`
- `artifacts/11_subtle_modulation_trap_map.png`

Current result:

- recall: `0.790`
- missed cases: `17`
- missed depths: `0.025` to `0.75` semitones
- missed rates: `2`, `4`, `5.5`, `8` Hz

The missed cases cluster at `+3 semitones`, and the worst fixed observer is consistently WSOLA. This suggests a new blind spot class: small modulation that is too subtle for coarse preflight risk but still hostile to period-based time-domain shifting.

## Experiment 12

`experiments/12_micro_modulation_preflight.py` tests a repair for Experiment 11.

It adds three micro-modulation descriptors to the source feature set:

- `pitch_modulation_rms_cents`
- `pitch_modulation_peak_rate_hz`
- `pitch_modulation_peak_strength`

The new modulation-aware preflight rule asks whether a signal looks stable and periodic overall, but still has a coherent low-amplitude pitch modulation peak.

Outputs:

- `artifacts/12_micro_modulation_predictions.csv`
- `artifacts/12_micro_modulation_summary.csv`
- `artifacts/12_micro_modulation_preflight_map.png`

Current result on the Experiment 11 grid:

- baseline recall: `0.790`
- modulation-aware recall: `1.000`
- baseline false negatives: `17`
- modulation-aware false negatives: `0`
- modulation-aware precision: `0.579`

This suggests that the subtle modulation trap is detectable before pitch shifting, but only with a descriptor designed for coherent micro-motion. Broad pitch stability metrics alone miss it.

## Experiment 13

`experiments/13_preflight_adaptive_selector_v1.py` turns the micro-modulation detector into an adaptive routing rule.

`Preflight Adaptive v1` is source-only:

- compute modulation-aware preflight risk
- if `micro_modulation_trap` is detected, avoid period-based routing
- route to Phase Vocoder with a small dry guard
- otherwise use a simple stable/noisy/transient risk policy

It evaluates the selector on the Experiment 11 subtle-modulation grid.

Outputs:

- `artifacts/13_results.csv`
- `artifacts/13_preflight_adaptive_cases.csv`
- `artifacts/13_preflight_adaptive_summary.csv`
- `artifacts/13_preflight_adaptive_v1_map.png`

Current result:

- fixed danger cases: `81`
- adaptive catastrophic cases: `0`
- avoided fixed-danger cases: `81`
- missed fixed-danger cases: `0`
- introduced catastrophic cases: `0`
- avoidance rate: `1.000`

This is the first prototype that uses a discovered blind spot to change routing behavior. It is deliberately conservative: on this grid, v1 selects Phase Vocoder for `140` of `144` cases. The next challenge is recovering selectivity without giving up safety.

## Experiment 14

`experiments/14_selectivity_recovery_frontier.py` searches source-only selector policies over the Experiment 11 subtle-modulation grid.

It asks:

```text
Can adaptive selection recover algorithm diversity without catastrophic failures?
```

The policy simulator varies:

- preflight high-risk threshold
- micro-modulation risk bonus
- micro-modulation RMS threshold
- micro-modulation peak-strength threshold

For each policy, it simulates routing to:

- Phase Vocoder for high-risk cases
- PSOLA for low-risk light shifts
- Rubber Band for low-risk large shifts

Outputs:

- `artifacts/14_policy_frontier.csv`
- `artifacts/14_best_policy_cases.csv`
- `artifacts/14_selectivity_summary.csv`
- `artifacts/14_selectivity_frontier.png`

Current result:

- policies searched: `1176`
- zero-catastrophic policies: `990`
- most selective safe policy: `thr0.50_bonus0.06_rms5_str0.30`
- Phase Vocoder selections in most selective safe policy: `49 / 144`
- PSOLA selections: `67 / 144`
- Rubber Band selections: `28 / 144`
- catastrophic failures: `0`

This shows that Experiment 13's safety does not require routing everything to Phase Vocoder. There is a safety/selectivity frontier: more Phase Vocoder lowers average stress, but a much more diverse policy can still avoid catastrophic failures.

## Roadmap

See [`ROADMAP.md`](ROADMAP.md) for the next steps, including deeper blind spot taxonomy and neural pitch shifter comparisons.
