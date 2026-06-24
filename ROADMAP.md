# Roadmap

## Phase 1: Stress Atlas

- Build a small synthetic signal zoo
- Map distortion under a single phase-vocoder pipeline
- Identify which structures are fragile under pitch shift

## Phase 2: Algorithm Disagreement Landscape

- Reuse the stress atlas across multiple algorithm families
- Add WSOLA
- Add PSOLA
- Add Rubber Band
- Add a neural pitch shifter
- Compute disagreement as variance of composite stress across algorithms
- Rank signals by disagreement instead of algorithm performance
- Use high-disagreement signals as blind spot candidates

## Phase 3: Blind Spot Taxonomy

- Add Blind Spot Discovery with generated candidate signals
- Search directly for signals that maximize algorithm disagreement
- Test the Ambiguity Hypothesis with controlled oscillator and modulation sweeps
- Zoom in on critical ambiguity zones and measure peak width
- Map observer-state pitch trajectories over time
- Map attractor basins and branch switches across ambiguity
- Build octave preference maps to identify observer priors
- Group failure modes by signal structure
- Separate transient loss, timbral smear, and harmonic drift
- Track which structures survive across representations

## Phase 4: Prediction

- Search for pre-flight indicators of failure
- Test whether simple signal descriptors predict distortion
- Build a compact blind-spot classifier for future experiments
- Prototype adaptive pitch-shifting rules that avoid known blind spots
- Track catastrophic-failure avoidance separately from average quality
- Replace expensive observer probes with source-only preflight risk features
- Add better subtle-modulation descriptors for low-depth vibrato failures
- Detect small modulation traps that masquerade as stable periodicity
- Add WSOLA-specific preflight warnings for period-tracking fragility
- Add micro-modulation-aware preflight scoring to adaptive selection
- Separate coherent pitch micro-motion from broad unstable pitch motion
- Measure adaptive-selector conservatism after adding preflight guards
- Recover algorithm selectivity without reintroducing catastrophic failures
- Map safety/selectivity frontiers for adaptive routing policies
- Choose adaptive policies from Pareto tradeoffs instead of single-score rankings
- Instantiate selective adaptive policies and verify them with rendered outputs
- Compare conservative and selective adaptive modes as separate product behaviors
- Test adaptive selectors on the full stress atlas after narrow-grid tuning
- Add fallback-specific guards for cases where the safe algorithm has its own blind spot
- Add noise-specific dry/wet fallback guards for unpitched material
- Treat fallback guards as separate from algorithm selection
