from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.registry import ALGORITHMS
from metrics.audio_metrics import rms_error, spectral_centroid_difference, spectral_distance
from metrics.disagreement import add_composite_stress, add_disagreement_levels, build_disagreement_landscape
from signals.discovery import DiscoveryCandidate, build_discovery_candidates
from visualization.discovery_heatmap import save_discovery_heatmap


SR = 22050
DURATION = 2.0
SHIFT_STEPS = [3, 7, 12, -12]
DEFAULT_CANDIDATES_PER_FAMILY = 3
DEFAULT_SEED = 303


def _evaluate_candidates(candidates: list[DiscoveryCandidate]) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        source = candidate.audio
        for algorithm in ALGORITHMS:
            for shift in SHIFT_STEPS:
                shifted = algorithm.pitch_shift(source, sr=SR, n_steps=shift)
                rows.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "candidate_family": candidate.family,
                        "candidate_name": candidate.name,
                        "candidate_label": candidate.label,
                        "parameter_summary": candidate.parameter_summary,
                        "algorithm": algorithm.name,
                        "algorithm_label": algorithm.display_name,
                        "implementation": algorithm.implementation,
                        "shift_semitones": shift,
                        "rms_error": rms_error(source, shifted),
                        "spectral_distance": spectral_distance(source, shifted),
                        "spectral_centroid_difference": spectral_centroid_difference(source, shifted, sr=SR),
                    }
                )
    return pd.DataFrame(rows).sort_values(["candidate_id", "shift_semitones", "algorithm"]).reset_index(drop=True)


def _build_candidate_summary(landscape: pd.DataFrame) -> pd.DataFrame:
    summary = (
        landscape.groupby(
            ["candidate_id", "candidate_family", "candidate_name", "candidate_label", "parameter_summary"],
            as_index=False,
        )
        .agg(
            mean_algorithm_disagreement=("algorithm_disagreement", "mean"),
            max_algorithm_disagreement=("algorithm_disagreement", "max"),
            mean_algorithm_spread=("algorithm_spread", "mean"),
            mean_composite_stress=("mean_composite_stress", "mean"),
        )
        .sort_values("max_algorithm_disagreement", ascending=False)
        .reset_index(drop=True)
    )

    best_shifts = []
    for _, row in summary.iterrows():
        candidate_rows = landscape[landscape["candidate_id"] == row["candidate_id"]]
        strongest = candidate_rows.sort_values("algorithm_disagreement", ascending=False).iloc[0]
        best_shifts.append(int(strongest["shift_semitones"]))

    summary["best_shift_semitones"] = best_shifts
    summary = add_disagreement_levels(summary, score_column="max_algorithm_disagreement")
    summary = summary.rename(columns={"disagreement_rank": "discovery_rank"})
    return summary


def _save_candidate_metadata(candidates: list[DiscoveryCandidate], output_path: Path) -> None:
    rows = [
        {
            "candidate_id": candidate.candidate_id,
            "candidate_family": candidate.family,
            "candidate_name": candidate.name,
            "candidate_label": candidate.label,
            "parameter_summary": candidate.parameter_summary,
        }
        for candidate in candidates
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)


def _save_best_signal(candidates: list[DiscoveryCandidate], summary: pd.DataFrame, output_path: Path) -> None:
    best_id = str(summary.iloc[0]["candidate_id"])
    best = next(candidate for candidate in candidates if candidate.candidate_id == best_id)
    times = np.arange(best.audio.size, dtype=np.float64) / float(SR)
    pd.DataFrame({"time_seconds": times, "amplitude": best.audio}).to_csv(output_path, index=False)


def run_experiment(candidates_per_family: int = DEFAULT_CANDIDATES_PER_FAMILY, seed: int = DEFAULT_SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if candidates_per_family < 1:
        raise ValueError("candidates_per_family must be at least 1")

    candidates = build_discovery_candidates(
        sr=SR,
        duration=DURATION,
        candidates_per_family=candidates_per_family,
        seed=seed,
    )
    results = add_composite_stress(_evaluate_candidates(candidates))
    landscape = build_disagreement_landscape(
        results,
        group_columns=[
            "candidate_id",
            "candidate_family",
            "candidate_name",
            "candidate_label",
            "parameter_summary",
            "shift_semitones",
        ],
    ).sort_values(["candidate_id", "shift_semitones"]).reset_index(drop=True)
    summary = _build_candidate_summary(landscape)

    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    _save_candidate_metadata(candidates, artifacts_dir / "03_candidates.csv")
    results.to_csv(artifacts_dir / "03_results.csv", index=False)
    landscape.to_csv(artifacts_dir / "03_disagreement_landscape.csv", index=False)
    summary.to_csv(artifacts_dir / "03_discovery_summary.csv", index=False)
    _save_best_signal(candidates, summary, artifacts_dir / "03_best_signal.csv")
    save_discovery_heatmap(landscape, summary, artifacts_dir / "03_heatmap.png")

    return results, landscape, summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 03: Blind Spot Discovery")
    parser.add_argument("--candidates-per-family", type=int, default=DEFAULT_CANDIDATES_PER_FAMILY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results, landscape, summary = run_experiment(
        candidates_per_family=args.candidates_per_family,
        seed=args.seed,
    )
    best = summary.iloc[0]
    print(f"Wrote {len(results)} algorithm rows to artifacts/03_results.csv")
    print(f"Wrote {len(landscape)} disagreement rows to artifacts/03_disagreement_landscape.csv")
    print("Wrote artifacts/03_discovery_summary.csv")
    print("Wrote artifacts/03_best_signal.csv")
    print("Wrote artifacts/03_heatmap.png")
    print("Question: What signal maximizes disagreement between pitch shifters?")
    print(
        f"Best candidate: {best['candidate_id']} "
        f"({best['candidate_family']}) at {int(best['best_shift_semitones']):+d} semitones, "
        f"disagreement={best['max_algorithm_disagreement']:.6f}"
    )
    print(summary[["discovery_rank", "candidate_id", "candidate_family", "best_shift_semitones", "max_algorithm_disagreement"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()

