from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.adaptive_preflight import pitch_shift_preflight_adaptive_v2, pitch_shift_preflight_adaptive_v3
from metrics.audio_metrics import rms_error, spectral_centroid_difference, spectral_distance
from metrics.disagreement import add_composite_stress
from signals.library import SIGNAL_SPECS, build_signal_atlas
from visualization.adaptive_selector_plot import save_adaptive_selector_plot


SR = 22050
DURATION = 2.0
SHIFT_STEPS = [3, 7, 12, -12]
FIXED_ALGORITHMS = {"phase_vocoder", "psola", "wsola", "rubber_band"}


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


def _load_fixed_results() -> pd.DataFrame:
    path = REPO_ROOT / "artifacts" / "09_results.csv"
    if not path.exists():
        raise FileNotFoundError("Experiment 18 expects artifacts/09_results.csv. Run Experiment 09 first.")
    results = pd.read_csv(path)
    fixed = results[results["algorithm"].isin(FIXED_ALGORITHMS)].copy()
    return fixed[
        [
            "signal_family",
            "signal_name",
            "signal_label",
            "algorithm",
            "algorithm_label",
            "implementation",
            "shift_semitones",
            "rms_error",
            "spectral_distance",
            "spectral_centroid_difference",
        ]
    ]


def _build_algorithm_summary(results: pd.DataFrame, threshold: float) -> pd.DataFrame:
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
                "catastrophic_threshold": threshold,
                "catastrophic_count": int(np.sum(stresses >= threshold)),
            }
        )
    return pd.DataFrame(rows).sort_values(["catastrophic_count", "mean_composite_stress"]).reset_index(drop=True)


def _build_signal_summary(results: pd.DataFrame, decisions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for signal_name, adaptive_rows in results[results["algorithm"] == "preflight_adaptive_v3"].groupby("signal_name", sort=False):
        decision_rows = decisions[decisions["signal_name"] == signal_name]
        stresses = adaptive_rows["composite_stress"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "signal_family": adaptive_rows["signal_family"].iloc[0],
                "signal_name": signal_name,
                "signal_label": adaptive_rows["signal_label"].iloc[0],
                "selected_algorithms": "|".join(sorted(decision_rows["selected_algorithm_label"].unique())),
                "guards": "|".join(sorted(decision_rows["guard"].unique())),
                "adaptive_mean_stress": float(np.mean(stresses)),
                "adaptive_max_stress": float(np.max(stresses)),
                "adaptive_catastrophic_count": int(np.sum(stresses >= threshold)),
                "best_fixed_algorithm_label": "n/a",
            }
        )
    return pd.DataFrame(rows).sort_values("adaptive_catastrophic_count", ascending=False).reset_index(drop=True)


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    atlas = build_signal_atlas(sr=SR, duration=DURATION, seed=7)
    fixed_results = _load_fixed_results()
    rows = []
    decisions = []

    for spec in SIGNAL_SPECS:
        source = atlas[spec.name]
        for shift in SHIFT_STEPS:
            shifted_v2, analysis_v2 = pitch_shift_preflight_adaptive_v2(source, sr=SR, n_steps=shift)
            rows.append(
                _metric_row(
                    source,
                    shifted_v2,
                    spec.family,
                    spec.name,
                    "preflight_adaptive_v2",
                    "Preflight Adaptive v2",
                    "selective source-only preflight selector",
                    shift,
                )
            )

            shifted_v3, analysis_v3 = pitch_shift_preflight_adaptive_v3(source, sr=SR, n_steps=shift)
            rows.append(
                _metric_row(
                    source,
                    shifted_v3,
                    spec.family,
                    spec.name,
                    "preflight_adaptive_v3",
                    "Preflight Adaptive v3",
                    "selective preflight selector with fallback guards",
                    shift,
                )
            )
            decisions.append(
                {
                    "signal_family": spec.family,
                    "signal_name": spec.name,
                    "signal_label": f"{spec.family} / {spec.name}",
                    "shift_semitones": shift,
                    "adaptive_state": analysis_v3.preflight_risk_level,
                    **asdict(analysis_v3),
                }
            )

    results = add_composite_stress(pd.concat([fixed_results, pd.DataFrame(rows)], ignore_index=True))
    fixed_scored = results[results["algorithm"].isin(FIXED_ALGORITHMS)]
    threshold = float(fixed_scored["composite_stress"].quantile(0.85))
    decisions_frame = pd.DataFrame(decisions)
    v3_scores = results[results["algorithm"] == "preflight_adaptive_v3"][["signal_name", "shift_semitones", "composite_stress"]]
    decisions_frame = decisions_frame.merge(v3_scores, on=["signal_name", "shift_semitones"], how="left")
    algorithm_summary = _build_algorithm_summary(results, threshold=threshold)
    signal_summary = _build_signal_summary(results, decisions_frame, threshold=threshold)

    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(artifacts_dir / "18_results.csv", index=False)
    decisions_frame.to_csv(artifacts_dir / "18_selector_decisions.csv", index=False)
    algorithm_summary.to_csv(artifacts_dir / "18_algorithm_summary.csv", index=False)
    signal_summary.to_csv(artifacts_dir / "18_signal_summary.csv", index=False)
    save_adaptive_selector_plot(
        decisions_frame,
        signal_summary,
        artifacts_dir / "18_adaptive_v3_map.png",
        title="Exp18 Adaptive v3 With Fallback Guards",
        subtitle="Broad-atlas risk, selected algorithm, guard, and stress.",
        legend_title="Risk Legend",
        legend_items=["low", "medium", "high"],
    )
    return results, decisions_frame, algorithm_summary, signal_summary


def main() -> None:
    results, decisions, algorithm_summary, signal_summary = run_experiment()
    print(f"Wrote {len(results)} rows to artifacts/18_results.csv")
    print(f"Wrote {len(decisions)} selector decisions to artifacts/18_selector_decisions.csv")
    print("Wrote artifacts/18_algorithm_summary.csv")
    print("Wrote artifacts/18_signal_summary.csv")
    print("Wrote artifacts/18_adaptive_v3_map.png")
    print("Question: Can fallback guards remove safe-mode fallback failures?")
    print(algorithm_summary.to_string(index=False))
    print(signal_summary.to_string(index=False))


if __name__ == "__main__":
    main()
