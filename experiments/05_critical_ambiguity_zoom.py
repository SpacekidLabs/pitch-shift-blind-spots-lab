from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.registry import ALGORITHMS
from metrics.audio_metrics import rms_error, spectral_centroid_difference, spectral_distance
from metrics.disagreement import add_composite_stress, build_disagreement_landscape
from signals.ambiguity import AmbiguityCase, build_critical_ambiguity_sweep
from visualization.critical_ambiguity_plot import save_critical_ambiguity_plot


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


def _interpolate_crossing(x1: float, y1: float, x2: float, y2: float, threshold: float) -> float:
    if y1 == y2:
        return x1
    ratio = (threshold - y1) / (y2 - y1)
    return x1 + ratio * (x2 - x1)


def _width_around_peak(x: np.ndarray, y: np.ndarray, fraction: float) -> float:
    peak_index = int(np.argmax(y))
    threshold = float(np.max(y) * fraction)

    left = float(x[0])
    for index in range(peak_index, 0, -1):
        if y[index - 1] < threshold <= y[index]:
            left = _interpolate_crossing(float(x[index - 1]), float(y[index - 1]), float(x[index]), float(y[index]), threshold)
            break

    right = float(x[-1])
    for index in range(peak_index, len(x) - 1):
        if y[index] >= threshold > y[index + 1]:
            right = _interpolate_crossing(float(x[index]), float(y[index]), float(x[index + 1]), float(y[index + 1]), threshold)
            break

    return max(0.0, right - left)


def _build_peak_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, frame in summary.groupby("sweep_family"):
        ordered = frame.sort_values("ambiguity_amount").reset_index(drop=True)
        x = ordered["ambiguity_amount"].to_numpy(dtype=np.float64)
        y = ordered["max_algorithm_disagreement"].to_numpy(dtype=np.float64)
        peak_index = int(np.argmax(y))
        peak_row = ordered.iloc[peak_index]
        rows.append(
            {
                "sweep_family": family,
                "ambiguity_units": str(peak_row["ambiguity_units"]),
                "peak_case_id": str(peak_row["case_id"]),
                "peak_disagreement_location": float(peak_row["ambiguity_amount"]),
                "peak_disagreement": float(peak_row["max_algorithm_disagreement"]),
                "peak_shift_semitones": int(peak_row["peak_shift_semitones"]),
                "peak_width": _width_around_peak(x, y, fraction=0.90),
                "peak_width_threshold": float(np.max(y) * 0.90),
                "half_max_width": _width_around_peak(x, y, fraction=0.50),
                "half_max_threshold": float(np.max(y) * 0.50),
            }
        )
    return pd.DataFrame(rows).sort_values("peak_disagreement", ascending=False).reset_index(drop=True)


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cases = build_critical_ambiguity_sweep(sr=SR, duration=DURATION)
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
    peak_metrics = _build_peak_metrics(summary)

    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(artifacts_dir / "05_results.csv", index=False)
    landscape.to_csv(artifacts_dir / "05_disagreement_landscape.csv", index=False)
    summary.to_csv(artifacts_dir / "05_ambiguity_summary.csv", index=False)
    peak_metrics.to_csv(artifacts_dir / "05_peak_metrics.csv", index=False)
    save_critical_ambiguity_plot(summary, peak_metrics, artifacts_dir / "05_critical_ambiguity_plot.png")

    return results, landscape, summary, peak_metrics


def main() -> None:
    results, landscape, summary, peak_metrics = run_experiment()
    print(f"Wrote {len(results)} algorithm rows to artifacts/05_results.csv")
    print(f"Wrote {len(landscape)} disagreement rows to artifacts/05_disagreement_landscape.csv")
    print("Wrote artifacts/05_ambiguity_summary.csv")
    print("Wrote artifacts/05_peak_metrics.csv")
    print("Wrote artifacts/05_critical_ambiguity_plot.png")
    print("Question: Is there a critical ambiguity zone where pitch-shifter disagreement peaks?")
    print(peak_metrics.to_string(index=False))
    print(summary[["sweep_family", "case_label", "ambiguity_amount", "max_algorithm_disagreement", "peak_shift_semitones"]].to_string(index=False))


if __name__ == "__main__":
    main()

