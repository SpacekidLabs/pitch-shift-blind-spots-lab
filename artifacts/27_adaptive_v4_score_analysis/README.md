# Experiment 27 Adaptive v4 Score Analysis

Question:

```text
Did the unblinded Adaptive v4 Normal preference survive blind scoring?
```

Finding:

- highest mean overall policy: `Adaptive v4 Normal` at `3.33`
- final rendered algorithm set: `Phase Vocoder`
- composite-stress correlation with overall rating: `0.12`
- widest equivalent-render overall-score range: `2.0` at `-12` semitones

Interpretation:

All Experiment 26 final renders resolved to Phase Vocoder, either directly or through the post-render health fallback.
That means the decoded policy means should not be treated as proof that one waveform-producing strategy sounded better.
Instead, this test exposes listener/context variability among equivalent or near-equivalent renders.

The practical result still supports Adaptive v4 as a safe product behavior:

- Adaptive v4 Normal avoided collapse without needing fallback on this drum loop.
- Forced WSOLA and Rubber Band paths were caught by the health gate and rescued.
- No blind file was described as silence, unlike Experiment 22.

The deeper lab result is that blind listening needs duplicate controls.
Without them, a small score difference can look like an algorithm preference even when the final rendered audio path is the same.

Outputs:

- `27_listener_scores.csv`
- `27_decoded_listener_scores.csv`
- `27_policy_summary.csv`
- `27_shift_summary.csv`
- `27_fallback_summary.csv`
- `27_equivalent_render_summary.csv`
- `27_metric_correlation_summary.csv`
- `27_adaptive_v4_score_plot.png`
