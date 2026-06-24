from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.phase_vocoder import pitch_shift_phase_vocoder
from metrics.audio_metrics import rms_error, spectral_centroid_difference, spectral_distance
from signals.library import SIGNAL_SPECS, build_signal_atlas
from visualization.heatmap import save_stress_heatmap


SR = 22050
DURATION = 2.0
SHIFT_STEPS = [3, 7, 12, -12]


def run_experiment() -> pd.DataFrame:
    atlas = build_signal_atlas(sr=SR, duration=DURATION, seed=7)
    rows = []

    for spec in SIGNAL_SPECS:
        source = atlas[spec.name]
        for shift in SHIFT_STEPS:
            shifted = pitch_shift_phase_vocoder(source, sr=SR, n_steps=shift)
            rows.append(
                {
                    "signal_family": spec.family,
                    "signal_name": spec.name,
                    "signal_label": f"{spec.family} / {spec.name}",
                    "algorithm": "phase_vocoder",
                    "implementation": "librosa_or_local_fallback",
                    "shift_semitones": shift,
                    "rms_error": rms_error(source, shifted),
                    "spectral_distance": spectral_distance(source, shifted),
                    "spectral_centroid_difference": spectral_centroid_difference(source, shifted, sr=SR),
                }
            )

    results = pd.DataFrame(rows).sort_values(["signal_family", "signal_name", "shift_semitones"]).reset_index(drop=True)
    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(artifacts_dir / "01_results.csv", index=False)
    save_stress_heatmap(results, artifacts_dir / "01_heatmap.png")
    return results


def main() -> None:
    results = run_experiment()
    print(f"Wrote {len(results)} rows to artifacts/01_results.csv")
    print("Wrote artifacts/01_heatmap.png")


if __name__ == "__main__":
    main()

