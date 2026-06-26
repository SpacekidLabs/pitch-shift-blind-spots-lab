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

## Experiment 15

`experiments/15_selective_adaptive_v2.py` instantiates the most selective safe policy from Experiment 14 as `Preflight Adaptive v2`.

The v2 policy uses:

- high-risk threshold: `0.50`
- micro-modulation bonus: `0.06`
- micro-modulation RMS threshold: `5` cents
- micro-modulation peak-strength threshold: `0.30`

Routing:

- high risk: Phase Vocoder
- low-risk light shifts: PSOLA
- low-risk large shifts: Rubber Band

Outputs:

- `artifacts/15_results.csv`
- `artifacts/15_selective_adaptive_v2_cases.csv`
- `artifacts/15_selective_adaptive_v2_summary.csv`
- `artifacts/15_selective_adaptive_v2_comparison.png`

Current result:

- fixed danger cases: `81`
- v2 catastrophic cases: `0`
- v2 avoided fixed-danger cases: `81`
- v2 missed fixed-danger cases: `0`
- v2 introduced catastrophic cases: `0`
- Phase Vocoder selections: `49 / 144`
- PSOLA selections: `67 / 144`
- Rubber Band selections: `28 / 144`

This confirms the frontier result in an instantiated selector. v2 recovers algorithm diversity while preserving catastrophic-failure avoidance on the subtle-modulation grid.

## Experiment 16

`experiments/16_selective_adaptive_v2_generalization.py` tests whether `Preflight Adaptive v2` generalizes from the subtle-modulation grid back to the original stress atlas.

It compares:

- fixed Phase Vocoder
- fixed PSOLA
- fixed WSOLA
- fixed Rubber Band
- `Preflight Adaptive v2`

Outputs:

- `artifacts/16_results.csv`
- `artifacts/16_selector_decisions.csv`
- `artifacts/16_algorithm_summary.csv`
- `artifacts/16_signal_summary.csv`
- `artifacts/16_selective_adaptive_v2_map.png`

Current result:

- `Preflight Adaptive v2` catastrophic cases: `1`
- Phase Vocoder catastrophic cases: `1`
- PSOLA catastrophic cases: `2`
- Rubber Band catastrophic cases: `7`
- WSOLA catastrophic cases: `17`

The remaining v2 catastrophic case is inherited from the Phase Vocoder fallback on `white_noise` at `-12` semitones. This is an important limitation: adaptive routing can avoid choosing the wrong observer, but it cannot eliminate blind spots in the fallback observer itself.

## Experiment 17

`experiments/17_noise_fallback_guard.py` tests a fallback-specific guard for the remaining broad-atlas failure from Experiment 16.

Question:

```text
Can a dry/wet guard remove the Phase Vocoder fallback failure on noise?
```

It sweeps Phase Vocoder dry/wet blending on noise-like atlas signals:

- dry mix: `0.00`, `0.05`, `0.10`, `0.15`, `0.20`, `0.30`, `0.40`
- signals: `white_noise`, `pink_noise`
- shifts: `+3`, `+7`, `+12`, `-12`

Outputs:

- `artifacts/17_noise_guard_results.csv`
- `artifacts/17_noise_guard_dry_mix_summary.csv`
- `artifacts/17_noise_guard_summary.csv`
- `artifacts/17_noise_guard_map.png`

Current result:

- best dry mix: `0.40`
- catastrophic cases at `0.00` dry: `5`
- catastrophic cases at `0.40` dry: `0`
- mean stress at `0.40` dry: `0.109`

This suggests that fallback failure needs its own guard layer. Routing to the safest observer is not enough when the fallback observer has a blind spot on unpitched/noise-like material.

## Experiment 18

`experiments/18_adaptive_v3_fallback_guards.py` combines the selective `Preflight Adaptive v2` router with the noise fallback guard from Experiment 17.

Question:

```text
Can fallback guards remove safe-mode fallback failures?
```

It compares:

- fixed Phase Vocoder
- fixed PSOLA
- fixed WSOLA
- fixed Rubber Band
- `Preflight Adaptive v2`
- `Preflight Adaptive v3`

Outputs:

- `artifacts/18_results.csv`
- `artifacts/18_selector_decisions.csv`
- `artifacts/18_algorithm_summary.csv`
- `artifacts/18_signal_summary.csv`
- `artifacts/18_adaptive_v3_map.png`

Current result:

- `Preflight Adaptive v3` catastrophic cases: `0`
- `Preflight Adaptive v2` catastrophic cases: `1`
- Phase Vocoder catastrophic cases: `1`
- PSOLA catastrophic cases: `2`
- Rubber Band catastrophic cases: `7`
- WSOLA catastrophic cases: `17`

The v3 selector keeps the v2 routing policy, then applies a dry/wet fallback guard for large-shift noise-like cases routed to Phase Vocoder, excluding sparse transients. This removes the remaining broad-atlas adaptive catastrophe without turning the selector into a single fixed algorithm.

## Experiment 19

`experiments/19_guard_audit.py` audits whether the v3 fallback guard is useful or just over-conservative.

Question:

```text
Does the fallback guard avoid catastrophes without suppressing the shift?
```

It reuses the v3 guard-triggered cases from Experiment 18 and sweeps:

- dry mix: `0.00`, `0.10`, `0.20`, `0.30`, `0.40`, `0.50`, `0.60`, `0.70`, `0.85`, `1.00`

It measures:

- composite stress
- catastrophic count
- shift retention
- dry/source similarity
- peak preservation
- guard audit score

Outputs:

- `artifacts/19_guard_audit_results.csv`
- `artifacts/19_guard_audit_summary.csv`
- `artifacts/19_guard_audit_case_summary.csv`
- `artifacts/19_guard_audit_plot.png`

Current result:

- best compromise dry mix: `0.10`
- catastrophic cases at `0.00` dry: `1`
- catastrophic cases at `0.10` dry: `0`
- mean stress at `0.10` dry: `0.222`
- mean shift retention at `0.10` dry: `0.957`
- guard audit score at `0.10` dry: `0.734`

This refines the guard from Experiment 17. A heavy guard can reduce metrics by drifting toward the dry signal, but the more interesting behavior is the smallest guard that removes catastrophe while preserving the pitch-shift action. For the current atlas, that boundary is around `0.10` dry mix.

## Experiment 20

`experiments/20_guard_specificity_test.py` tests whether the v3 guard fires only where it should.

Question:

```text
Does the guard help only where it should?
```

It compares guard policies on the same atlas:

- no guard
- current v3
- noise only
- noise plus transient
- forced everywhere

It measures:

- catastrophic count
- rescued catastrophe count
- missed guard count
- unnecessary guard count
- mean stress
- shift retention
- guard specificity score

Outputs:

- `artifacts/20_guard_specificity_results.csv`
- `artifacts/20_guard_policy_summary.csv`
- `artifacts/20_guard_signal_summary.csv`
- `artifacts/20_guard_specificity_plot.png`

Current result:

- current v3 guarded cases: `2 / 44`
- current v3 catastrophic cases: `0`
- current v3 rescued cases: `1`
- current v3 unnecessary guards: `1`
- current v3 mean shift retention: `0.998`
- forced-everywhere unnecessary guards: `43`
- no-guard catastrophic cases: `1`

This changed the v3 rule. The earlier broad guard treated sparse transients as noise-like and guarded `6 / 44` cases. Experiment 20 showed that a noise-only guard rescues the same failure while avoiding unnecessary transient guards. The current v3 rule is now: guard large-shift Phase Vocoder fallback when the source is noise-like but not sparse-transient.

## Experiment 21

`experiments/21_guard_type_comparison.py` compares different fallback actions on the same v3 guard-triggered cases.

Question:

```text
What should the fallback guard do?
```

It compares:

- no guard
- dry/wet blend at `0.10`
- dry/wet blend at `0.40`
- reduced shift strength at `80%`
- spectral smoothing
- Rubber Band crossfade
- Rubber Band fallback

It measures:

- catastrophic count
- composite stress
- shift retention
- source similarity
- peak preservation
- guard type score

Outputs:

- `artifacts/21_guard_type_results.csv`
- `artifacts/21_guard_type_summary.csv`
- `artifacts/21_guard_type_case_summary.csv`
- `artifacts/21_guard_type_plot.png`

Current result:

- best guard type: reduced shift strength at `80%`
- reduced-shift catastrophic cases: `0`
- reduced-shift mean stress: `0.226`
- reduced-shift mean shift retention: `0.980`
- dry/wet `0.10` catastrophic cases: `0`
- dry/wet `0.10` mean stress: `0.222`
- dry/wet `0.10` mean shift retention: `0.957`
- Rubber Band fallback catastrophic cases: `1`
- Rubber Band crossfade catastrophic cases: `1`

This suggests that "use a different algorithm" is not automatically a safer guard. On the current guarded noise cases, reduced shift strength slightly outperforms dry/wet blending, while Rubber Band fallback and crossfade still miss the catastrophe. The case-level summary also shows that `white_noise +12` does not need a guard, while `white_noise -12` does. That points toward a future direction-specific guard trigger.

## Experiment 22

`experiments/22_drum_listening_test.py` creates a blinded listening test from a real drum loop sample.

Question:

```text
Which pitch-shift render feels best on a real drum loop?
```

The current session uses:

- source: `2023-01-02 - 002 - 121 bpm - crash-y tom-y.wav`
- excerpt length: `8` seconds
- sample rate: `48000`
- channels: stereo
- shifts: `+3`, `+7`, `-12` semitones
- strategies: Phase Vocoder, WSOLA, Rubber Band, PSOLA, Adaptive v3

Outputs:

- `artifacts/22_drum_listening_test/README.md`
- `artifacts/22_drum_listening_test/blind_listening_session.html`
- `artifacts/22_drum_listening_test/listening_sheet.csv`
- `artifacts/22_drum_listening_test/blind_manifest.csv`
- `artifacts/22_drum_listening_test/objective_metrics.csv`
- `artifacts/22_drum_listening_test/objective_metrics_plot.png`
- local-only audio files in `artifacts/22_drum_listening_test/audio/`
- local-only answer key at `artifacts/22_drum_listening_test/private_answer_key.csv`

Current result:

- listening stimuli: `15`
- reference file: `audio/REFERENCE_original_excerpt.wav`
- level matched: `True`
- blind scoring page: `blind_listening_session.html`
- Adaptive v3 collapses to Phase Vocoder on this drum loop
- the adaptive/Phase Vocoder duplicate pairs act as blind consistency checks

This is the first bridge from synthetic blind-spot mapping into actual listening. The blind scoring page lets a listener play each file, rate it, save progress in the browser, and export scores before opening the private answer key. The objective metrics are included only as diagnostics; the real point is to score transient crispness, groove preservation, artifact severity, and overall usefulness by ear.

## Experiment 23

`experiments/23_listening_score_analysis.py` analyzes the completed Experiment 22 blind listening scores.

Question:

```text
Do objective stress metrics agree with human drum-loop preference?
```

It combines:

- blind listener scores
- local private answer key
- objective metrics from Experiment 22

Outputs:

- `artifacts/23_listening_score_analysis/23_decoded_listener_scores.csv`
- `artifacts/23_listening_score_analysis/23_strategy_summary.csv`
- `artifacts/23_listening_score_analysis/23_shift_summary.csv`
- `artifacts/23_listening_score_analysis/23_metric_correlation_summary.csv`
- `artifacts/23_listening_score_analysis/23_duplicate_consistency.csv`
- `artifacts/23_listening_score_analysis/23_note_flags.csv`
- `artifacts/23_listening_score_analysis/23_listening_score_plot.png`

Current result:

- best overall mean score: Phase Vocoder and Adaptive v3 tied at `2.67`
- Adaptive v3 tied because it selected Phase Vocoder for this drum loop
- WSOLA mean overall score: `1.00`
- Rubber Band mean overall score: `1.00`
- WSOLA silence flags: `3 / 3`
- Rubber Band silence flags: `3 / 3`
- composite-stress correlation with overall rating: `-0.01`
- spectral-distance correlation with overall rating: `-0.74`

This is an important real-audio correction to the synthetic lab. The listener did not simply prefer the lowest composite stress. Several renders that looked plausible by objective summaries were perceived as silence or unusable. The listening test turns "algorithm stress" into a more product-relevant question: does the render preserve audible groove and transient identity?

## Experiment 24

`experiments/24_render_health_gate.py` tests whether simple post-render health metrics can detect the perceived silence failures from Experiment 23.

Question:

```text
Can render health detect perceived silence failures?
```

It measures each rendered listening-test WAV against the reference:

- relative RMS in dB
- active frame fraction
- median frame RMS
- peak ratio
- crest factor

Outputs:

- `artifacts/24_render_health_gate/24_render_health_results.csv`
- `artifacts/24_render_health_gate/24_render_health_detection_summary.csv`
- `artifacts/24_render_health_gate/24_render_health_strategy_summary.csv`
- `artifacts/24_render_health_gate/24_render_health_plot.png`

Current result:

- listener silence flags: `6`
- render-health flags: `6`
- true positives: `6`
- false positives: `0`
- false negatives: `0`
- silence precision: `1.00`
- silence recall: `1.00`
- WSOLA mean relative RMS: `-33.1 dB`
- Rubber Band mean relative RMS: `-32.9 dB`

The silence failures were real waveform-health failures, not just subjective dislike. WSOLA and Rubber Band renders had normal peak spikes but extremely low RMS and almost no active frames. This suggests the adaptive system needs a post-render health gate before listening or delivery: if a render collapses, automatically reject it and fall back.

## Experiment 25

`experiments/25_post_render_fallback_selector.py` prototypes `Adaptive v4`: a post-render health-gated selector.

Question:

```text
Can a post-render health gate rescue collapsed candidate renders?
```

Architecture:

```text
preflight candidate
render
health gate
if collapsed:
    try fallback chain
```

It compares:

- fixed Phase Vocoder
- fixed WSOLA
- fixed Rubber Band
- fixed PSOLA
- Adaptive v3
- Adaptive v4
- Adaptive v4 starting from a WSOLA candidate
- Adaptive v4 starting from a Rubber Band candidate

Outputs:

- `artifacts/25_post_render_fallback_selector/25_post_render_fallback_results.csv`
- `artifacts/25_post_render_fallback_selector/25_post_render_fallback_summary.csv`
- `artifacts/25_post_render_fallback_selector/25_post_render_fallback_plot.png`

Current result:

- fixed WSOLA health flags: `3 / 3`
- fixed Rubber Band health flags: `3 / 3`
- Adaptive v4 health flags: `0 / 3`
- Adaptive v4 from WSOLA fallback count: `3 / 3`
- Adaptive v4 from Rubber Band fallback count: `3 / 3`
- all v4 fallback probes resolved to Phase Vocoder

This is the first prototype of a pitch shifter that checks whether its own output is alive. Normal Adaptive v4 chooses Phase Vocoder for this drum loop, so it does not need fallback. The forced-candidate probes show the safety mechanism: if a future selector chooses WSOLA or Rubber Band and the render collapses, post-render health rejects it and falls back before delivery.

## Experiment 26

`experiments/26_adaptive_v4_blind_listening_test.py` turns the Adaptive v4 observation into a blind listening test.

Question:

```text
Does Adaptive v4 Normal remain preferred when compared blindly against rescue-path renders?
```

It compares:

- Phase Vocoder
- Adaptive v3
- Adaptive v4 Normal
- Adaptive v4 Rescue From WSOLA
- Adaptive v4 Rescue From Rubber Band

Outputs:

- `artifacts/26_adaptive_v4_blind_listening_test/README.md`
- `artifacts/26_adaptive_v4_blind_listening_test/blind_listening_session.html`
- `artifacts/26_adaptive_v4_blind_listening_test/listening_sheet.csv`
- `artifacts/26_adaptive_v4_blind_listening_test/blind_manifest.csv`
- `artifacts/26_adaptive_v4_blind_listening_test/objective_metrics.csv`
- `artifacts/26_adaptive_v4_blind_listening_test/objective_metrics_plot.png`
- `artifacts/26_adaptive_v4_blind_listening_test/session_info.csv`

Local-only outputs:

- `artifacts/26_adaptive_v4_blind_listening_test/audio/`
- `artifacts/26_adaptive_v4_blind_listening_test/private_answer_key.csv`

Current result:

- stimuli: `15`
- shifts: `+3`, `+7`, `-12`
- Adaptive v4 Normal selected Phase Vocoder directly
- WSOLA and Rubber Band rescue probes were rejected by the render-health gate
- all rescue probes fell back to Phase Vocoder

This test is deliberately not a new "which algorithm is best" benchmark. It asks whether the adaptive safety path preserves the listening preference under blind scoring. Because multiple blind IDs collapse to the same final Phase Vocoder render, it also acts as a listener-stability check: if equivalent renders receive different scores, the blind test is exposing context effects and rating noise rather than algorithm differences.

## Experiment 27

`experiments/27_adaptive_v4_score_analysis.py` decodes the Experiment 26 blind scores.

Question:

```text
Did the unblinded Adaptive v4 Normal preference survive blind scoring?
```

Outputs:

- `artifacts/27_adaptive_v4_score_analysis/README.md`
- `artifacts/27_adaptive_v4_score_analysis/27_listener_scores.csv`
- `artifacts/27_adaptive_v4_score_analysis/27_decoded_listener_scores.csv`
- `artifacts/27_adaptive_v4_score_analysis/27_policy_summary.csv`
- `artifacts/27_adaptive_v4_score_analysis/27_shift_summary.csv`
- `artifacts/27_adaptive_v4_score_analysis/27_fallback_summary.csv`
- `artifacts/27_adaptive_v4_score_analysis/27_equivalent_render_summary.csv`
- `artifacts/27_adaptive_v4_score_analysis/27_metric_correlation_summary.csv`
- `artifacts/27_adaptive_v4_score_analysis/27_adaptive_v4_score_plot.png`

Current result:

- Adaptive v4 Normal mean overall score: `3.33`
- Adaptive v3 mean overall score: `3.33`
- Adaptive v4 Rescue From WSOLA mean overall score: `3.00`
- Adaptive v4 Rescue From Rubber Band mean overall score: `2.67`
- Phase Vocoder mean overall score: `2.33`
- all final renders resolved to Phase Vocoder
- no blind file was reported as silence
- composite-stress correlation with overall rating: `0.12`
- widest equivalent-render overall-score range: `2.0`

The most important conclusion is not that Adaptive v4 "beat" Phase Vocoder. Since all Experiment 26 final renders resolved to Phase Vocoder, the decoded policy differences are partly listener/context effects among equivalent render paths. The practical result still supports Adaptive v4 as a safe behavior: the health gate rescued risky paths and avoided the silence failures from Experiment 22. The research result is that future listening tests need explicit duplicate controls and preference-stability analysis before treating small score gaps as algorithm differences.

## VST Prototype

`vst_prototype/` starts the real-time plugin path for the adaptive pitch shifter.

Current scope:

- JUCE VST3 scaffold
- pure C++ adaptive selector engine
- live block-level signal-state analysis
- observer-disagreement proxy
- safe-mode routing rules
- plugin UI for shift, dry/wet, safe mode, state, strategy, and disagreement
- standalone smoke test that builds without JUCE

The first prototype is intentionally a control-plane plugin, not a production pitch shifter. It passes audio through while reporting the selected state and strategy. The next milestone is to add backend adapters for pass-through, Phase Vocoder, and Rubber Band so the selector can control real pitch-shift engines.

Local install:

```sh
vst_prototype/scripts/build_and_install_vst3.sh
```

Installed path:

```text
~/Library/Audio/Plug-Ins/VST3/Pitch Shift Blind Spots.vst3
```

## Roadmap

See [`ROADMAP.md`](ROADMAP.md) for the next steps, including deeper blind spot taxonomy and neural pitch shifter comparisons.
