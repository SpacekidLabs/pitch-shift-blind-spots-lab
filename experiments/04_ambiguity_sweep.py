from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.registry import ALGORITHMS
from metrics.audio_metrics import rms_error, spectral_centroid_difference, spectral_distance
from metrics.disagreement import add_composite_stress, build_disagreement_landscape
from signals.ambiguity import AmbiguityCase, build_ambiguity_sweep
from visualization.ambiguity_plot import save_ambiguity_plot


SR = 22050
DURATION = 2.0
SHIFT_STEPS = [3, 7, 12, -12]


def _evaluate_cases(cases: list[AmbiguityCase]) -> pd.DataFrame:
    rows = []
    for case in cases:
        source = case.audio
        for algorithm in ALGORITHMS:
            for shift in SHIFT_STEPS:
                shifted = algorithm.pitch_shift(source, sr=SR, n_steps=shift)
                rows.append(
                    {
                        "case_id": case.case_id,
                        "sweep_family": case.family,
                        "case_label": case.label,
                        "ambiguity_amount": case.ambiguity_amount,
                        "ambiguity_units": case.ambiguity_units,
                        "parameter_summary": case.parameter_summary,
                        "algorithm": algorithm.name,
                        "algorithm_label": algorithm.display_name,
                        "implementation": algorithm.implementation,
                        "shift_semitones": shift,
                        "rms_error": rms_error(source, shifted),
                        "spectral_distance": spectral_distance(source, shifted),
                        "spectral_centroid_difference": spectral_centroid_difference(source, shifted, sr=SR),
                    }
                )
    return pd.DataFrame(rows).sort_values(["sweep_family", "ambiguity_amount", "shift_semitones", "algorithm"]).reset_index(drop=True)


def _build_summary(landscape: pd.DataFrame) -> pd.DataFrame:
    summary = (
        landscape.groupby(
            ["case_id", "sweep_family", "case_label", "ambiguity_amount", "ambiguity_units", "parameter_summary"],
            as_index=False,
        )
        .agg(
            mean_algorithm_disagreement=("algorithm_disagreement", "mean"),
            max_algorithm_disagreement=("algorithm_disagreement", "max"),
            mean_algorithm_spread=("algorithm_spread", "mean"),
            mean_composite_stress=("mean_composite_stress", "mean"),
        )
        .sort_values(["sweep_family", "ambiguity_amount"])
        .reset_index(drop=True)
    )

    peak_shifts = []
    for _, row in summary.iterrows():
        case_rows = landscape[landscape["case_id"] == row["case_id"]]
        peak = case_rows.sort_values("algorithm_disagreement", ascending=False).iloc[0]
        peak_shifts.append(int(peak["shift_semitones"]))
    summary["peak_shift_semitones"] = peak_shifts
    return summary


def _build_family_trends(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, frame in summary.groupby("sweep_family"):
        ordered = frame.sort_values("ambiguity_amount")
        x = ordered["ambiguity_amount"].to_numpy(dtype=float)
        y = ordered["max_algorithm_disagreement"].to_numpy(dtype=float)
        if len(ordered) > 1:
            slope = float(pd.Series(y).corr(pd.Series(x), method="pearson"))
            monotonic_steps = int((pd.Series(y).diff().fillna(0.0) >= 0.0).sum() - 1)
        else:
            slope = 0.0
            monotonic_steps = 0
        rows.append(
            {
                "sweep_family": family,
                "ambiguity_units": str(ordered["ambiguity_units"].iloc[0]),
                "pearson_correlation": slope,
                "nondecreasing_steps": monotonic_steps,
                "step_count": max(0, len(ordered) - 1),
                "peak_case_id": str(ordered.sort_values("max_algorithm_disagreement", ascending=False).iloc[0]["case_id"]),
                "peak_disagreement": float(ordered["max_algorithm_disagreement"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values("peak_disagreement", ascending=False).reset_index(drop=True)


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cases = build_ambiguity_sweep(sr=SR, duration=DURATION)
    results = add_composite_stress(_evaluate_cases(cases))
    landscape = build_disagreement_landscape(
        results,
        group_columns=[
            "case_id",
            "sweep_family",
            "case_label",
            "ambiguity_amount",
            "ambiguity_units",
            "parameter_summary",
            "shift_semitones",
        ],
    ).sort_values(["sweep_family", "ambiguity_amount", "shift_semitones"]).reset_index(drop=True)
    summary = _build_summary(landscape)
    trends = _build_family_trends(summary)

    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(artifacts_dir / "04_results.csv", index=False)
    landscape.to_csv(artifacts_dir / "04_disagreement_landscape.csv", index=False)
    summary.to_csv(artifacts_dir / "04_ambiguity_summary.csv", index=False)
    trends.to_csv(artifacts_dir / "04_family_trends.csv", index=False)
    save_ambiguity_plot(summary, artifacts_dir / "04_ambiguity_plot.png")

    return results, landscape, summary, trends


def main() -> None:
    results, landscape, summary, trends = run_experiment()
    print(f"Wrote {len(results)} algorithm rows to artifacts/04_results.csv")
    print(f"Wrote {len(landscape)} disagreement rows to artifacts/04_disagreement_landscape.csv")
    print("Wrote artifacts/04_ambiguity_summary.csv")
    print("Wrote artifacts/04_family_trends.csv")
    print("Wrote artifacts/04_ambiguity_plot.png")
    print("Question: Does algorithm disagreement increase as pitch ambiguity increases?")
    print(trends[["sweep_family", "pearson_correlation", "peak_case_id", "peak_disagreement"]].to_string(index=False))
    print(summary[["sweep_family", "case_label", "ambiguity_amount", "max_algorithm_disagreement", "peak_shift_semitones"]].to_string(index=False))


if __name__ == "__main__":
    main()

