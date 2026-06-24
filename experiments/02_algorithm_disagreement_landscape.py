from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.registry import ALGORITHMS
from metrics.audio_metrics import rms_error, spectral_centroid_difference, spectral_distance
from metrics.disagreement import add_composite_stress, add_disagreement_levels, build_disagreement_landscape
from signals.library import SIGNAL_SPECS, build_signal_atlas
from visualization.disagreement_heatmap import save_disagreement_heatmap


SR = 22050
DURATION = 2.0
SHIFT_STEPS = [3, 7, 12, -12]


def _build_signal_summary(landscape: pd.DataFrame) -> pd.DataFrame:
    summary = (
        landscape.groupby(["signal_family", "signal_name", "signal_label"], as_index=False)
        .agg(
            mean_algorithm_disagreement=("algorithm_disagreement", "mean"),
            max_algorithm_disagreement=("algorithm_disagreement", "max"),
            mean_algorithm_spread=("algorithm_spread", "mean"),
            mean_composite_stress=("mean_composite_stress", "mean"),
        )
        .sort_values("mean_algorithm_disagreement", ascending=False)
        .reset_index(drop=True)
    )

    most_disagreeing_shifts = []
    for _, row in summary.iterrows():
        signal_rows = landscape[landscape["signal_name"] == row["signal_name"]]
        strongest = signal_rows.sort_values("algorithm_disagreement", ascending=False).iloc[0]
        most_disagreeing_shifts.append(int(strongest["shift_semitones"]))

    summary["most_disagreeing_shift"] = most_disagreeing_shifts
    return add_disagreement_levels(summary)


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    atlas = build_signal_atlas(sr=SR, duration=DURATION, seed=7)
    rows = []

    for algorithm in ALGORITHMS:
        for spec in SIGNAL_SPECS:
            source = atlas[spec.name]
            for shift in SHIFT_STEPS:
                shifted = algorithm.pitch_shift(source, sr=SR, n_steps=shift)
                rows.append(
                    {
                        "signal_family": spec.family,
                        "signal_name": spec.name,
                        "signal_label": f"{spec.family} / {spec.name}",
                        "algorithm": algorithm.name,
                        "algorithm_label": algorithm.display_name,
                        "implementation": algorithm.implementation,
                        "shift_semitones": shift,
                        "rms_error": rms_error(source, shifted),
                        "spectral_distance": spectral_distance(source, shifted),
                        "spectral_centroid_difference": spectral_centroid_difference(source, shifted, sr=SR),
                    }
                )

    results = pd.DataFrame(rows).sort_values(
        ["algorithm", "signal_family", "signal_name", "shift_semitones"]
    ).reset_index(drop=True)
    results = add_composite_stress(results)
    landscape = build_disagreement_landscape(
        results,
        group_columns=["signal_family", "signal_name", "signal_label", "shift_semitones"],
    ).sort_values(["signal_family", "signal_name", "shift_semitones"]).reset_index(drop=True)
    summary = _build_signal_summary(landscape)

    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(artifacts_dir / "02_results.csv", index=False)
    landscape.to_csv(artifacts_dir / "02_disagreement_landscape.csv", index=False)
    summary.to_csv(artifacts_dir / "02_disagreement_summary.csv", index=False)
    save_disagreement_heatmap(landscape, summary, artifacts_dir / "02_heatmap.png")

    return results, landscape, summary


def main() -> None:
    results, landscape, summary = run_experiment()
    print(f"Wrote {len(results)} algorithm rows to artifacts/02_results.csv")
    print(f"Wrote {len(landscape)} disagreement rows to artifacts/02_disagreement_landscape.csv")
    print("Wrote artifacts/02_disagreement_summary.csv")
    print("Wrote artifacts/02_heatmap.png")
    print("Question: Which structures maximize disagreement between pitch-shifting algorithms?")
    print(summary[["disagreement_rank", "signal_name", "disagreement_level", "mean_algorithm_disagreement"]].to_string(index=False))


if __name__ == "__main__":
    main()
