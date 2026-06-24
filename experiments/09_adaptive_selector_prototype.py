from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.adaptive import pitch_shift_adaptive_v0
from algorithms.registry import ALGORITHMS
from metrics.audio_metrics import rms_error, spectral_centroid_difference, spectral_distance
from metrics.disagreement import add_composite_stress
from signals.library import SIGNAL_SPECS, build_signal_atlas
from visualization.adaptive_selector_plot import save_adaptive_selector_plot


SR = 22050
DURATION = 2.0
SHIFT_STEPS = [3, 7, 12, -12]
ADAPTIVE_NAME = "adaptive_selector_v0"
ADAPTIVE_LABEL = "Adaptive Selector v0"


def _metric_row(
    source: np.ndarray,
    shifted: np.ndarray,
    signal_family: str,
    signal_name: str,
    algorithm: str,
    algorithm_label: str,
    implementation: str,
    shift: int,
) -> dict[str, float | int | str]:
    return {
        "signal_family": signal_family,
        "signal_name": signal_name,
        "signal_label": f"{signal_family} / {signal_name}",
        "algorithm": algorithm,
        "algorithm_label": algorithm_label,
        "implementation": implementation,
        "shift_semitones": shift,
        "rms_error": rms_error(source, shifted),
        "spectral_distance": spectral_distance(source, shifted),
        "spectral_centroid_difference": spectral_centroid_difference(source, shifted, sr=SR),
    }


def _catastrophic_threshold(results: pd.DataFrame) -> float:
    fixed = results[results["algorithm"] != ADAPTIVE_NAME]
    return float(fixed["composite_stress"].quantile(0.85))


def _build_algorithm_summary(results: pd.DataFrame, catastrophic_threshold: float) -> pd.DataFrame:
    rows = []
    for algorithm_label, frame in results.groupby("algorithm_label"):
        stresses = frame["composite_stress"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "algorithm_label": algorithm_label,
                "case_count": int(len(frame)),
                "mean_composite_stress": float(np.mean(stresses)),
                "median_composite_stress": float(np.median(stresses)),
                "p90_composite_stress": float(np.percentile(stresses, 90)),
                "max_composite_stress": float(np.max(stresses)),
                "catastrophic_threshold": catastrophic_threshold,
                "catastrophic_count": int(np.sum(stresses >= catastrophic_threshold)),
            }
        )
    return pd.DataFrame(rows).sort_values("max_composite_stress").reset_index(drop=True)


def _build_signal_summary(results: pd.DataFrame, decisions: pd.DataFrame, catastrophic_threshold: float) -> pd.DataFrame:
    rows = []
    for signal_name, adaptive_rows in results[results["algorithm"] == ADAPTIVE_NAME].groupby("signal_name", sort=False):
        fixed_rows = results[(results["signal_name"] == signal_name) & (results["algorithm"] != ADAPTIVE_NAME)]
        by_algorithm = (
            fixed_rows.groupby("algorithm_label", as_index=False)
            .agg(fixed_mean_stress=("composite_stress", "mean"))
            .sort_values("fixed_mean_stress")
            .reset_index(drop=True)
        )
        decision_rows = decisions[decisions["signal_name"] == signal_name]
        states = "|".join(sorted(decision_rows["adaptive_state"].unique()))
        selected = "|".join(sorted(decision_rows["selected_algorithm_label"].unique()))
        stresses = adaptive_rows["composite_stress"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "signal_family": str(adaptive_rows["signal_family"].iloc[0]),
                "signal_name": signal_name,
                "signal_label": str(adaptive_rows["signal_label"].iloc[0]),
                "adaptive_states": states,
                "selected_algorithms": selected,
                "adaptive_mean_stress": float(np.mean(stresses)),
                "adaptive_max_stress": float(np.max(stresses)),
                "adaptive_catastrophic_count": int(np.sum(stresses >= catastrophic_threshold)),
                "best_fixed_algorithm_label": str(by_algorithm["algorithm_label"].iloc[0]),
                "best_fixed_mean_stress": float(by_algorithm["fixed_mean_stress"].iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values("adaptive_max_stress", ascending=False).reset_index(drop=True)


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    atlas = build_signal_atlas(sr=SR, duration=DURATION, seed=7)
    metric_rows = []
    decision_rows = []

    for spec in SIGNAL_SPECS:
        source = atlas[spec.name]
        for shift in SHIFT_STEPS:
            for algorithm in ALGORITHMS:
                shifted = algorithm.pitch_shift(source, sr=SR, n_steps=shift)
                metric_rows.append(
                    _metric_row(
                        source=source,
                        shifted=shifted,
                        signal_family=spec.family,
                        signal_name=spec.name,
                        algorithm=algorithm.name,
                        algorithm_label=algorithm.display_name,
                        implementation=algorithm.implementation,
                        shift=shift,
                    )
                )

            adaptive_shifted, analysis = pitch_shift_adaptive_v0(source, sr=SR, n_steps=shift)
            metric_rows.append(
                _metric_row(
                    source=source,
                    shifted=adaptive_shifted,
                    signal_family=spec.family,
                    signal_name=spec.name,
                    algorithm=ADAPTIVE_NAME,
                    algorithm_label=ADAPTIVE_LABEL,
                    implementation="observer-state adaptive selector prototype",
                    shift=shift,
                )
            )
            decision_rows.append(
                {
                    "signal_family": spec.family,
                    "signal_name": spec.name,
                    "signal_label": f"{spec.family} / {spec.name}",
                    "shift_semitones": shift,
                    "adaptive_state": analysis.state,
                    **asdict(analysis),
                }
            )

    results = pd.DataFrame(metric_rows)
    results = add_composite_stress(results)
    decisions = pd.DataFrame(decision_rows)
    adaptive_scores = results[results["algorithm"] == ADAPTIVE_NAME][
        ["signal_name", "shift_semitones", "composite_stress"]
    ]
    decisions = decisions.merge(adaptive_scores, on=["signal_name", "shift_semitones"], how="left")
    threshold = _catastrophic_threshold(results)
    algorithm_summary = _build_algorithm_summary(results, catastrophic_threshold=threshold)
    signal_summary = _build_signal_summary(results, decisions, catastrophic_threshold=threshold)

    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(artifacts_dir / "09_results.csv", index=False)
    decisions.to_csv(artifacts_dir / "09_selector_decisions.csv", index=False)
    algorithm_summary.to_csv(artifacts_dir / "09_algorithm_summary.csv", index=False)
    signal_summary.to_csv(artifacts_dir / "09_signal_summary.csv", index=False)
    save_adaptive_selector_plot(decisions, signal_summary, artifacts_dir / "09_adaptive_selector_map.png")
    return results, decisions, algorithm_summary, signal_summary


def main() -> None:
    results, decisions, algorithm_summary, signal_summary = run_experiment()
    print(f"Wrote {len(results)} rows to artifacts/09_results.csv")
    print(f"Wrote {len(decisions)} selector decisions to artifacts/09_selector_decisions.csv")
    print("Wrote artifacts/09_algorithm_summary.csv")
    print("Wrote artifacts/09_signal_summary.csv")
    print("Wrote artifacts/09_adaptive_selector_map.png")
    print("Question: Can an observer-aware pitch shifter avoid catastrophic failures?")
    print(algorithm_summary.to_string(index=False))
    print(signal_summary.to_string(index=False))


if __name__ == "__main__":
    main()
