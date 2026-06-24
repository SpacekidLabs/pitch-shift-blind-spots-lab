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
- Group failure modes by signal structure
- Separate transient loss, timbral smear, and harmonic drift
- Track which structures survive across representations

## Phase 4: Prediction

- Search for pre-flight indicators of failure
- Test whether simple signal descriptors predict distortion
- Build a compact blind-spot classifier for future experiments
