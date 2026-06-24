# Roadmap

## Phase 1: Stress Atlas

- Build a small synthetic signal zoo
- Map distortion under a single phase-vocoder pipeline
- Identify which structures are fragile under pitch shift

## Phase 2: Algorithm Comparison

- Reuse the stress atlas across multiple algorithm families
- Add WSOLA
- Add PSOLA
- Add Rubber Band
- Add a neural pitch shifter
- Compare failure shapes across methods
- Ask whether the same signal structures break every family

## Phase 3: Blind Spot Taxonomy

- Group failure modes by signal structure
- Separate transient loss, timbral smear, and harmonic drift
- Track which structures survive across representations

## Phase 4: Prediction

- Search for pre-flight indicators of failure
- Test whether simple signal descriptors predict distortion
- Build a compact blind-spot classifier for future experiments
