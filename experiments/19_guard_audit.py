from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.registry import get_algorithm
from metrics.audio_metrics import rms_error, spectral_centroid_difference, spectral_distance
from metrics.guard_metrics import blend_with_source, dry_similarity, peak_preservation, shift_retention
from signals.library import build_signal_atlas
from visualization.guard_audit_plot import save_guard_audit_plot


SR = 22050
DURATION = 2.0
DRY_MIXES = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.0]
REFERENCE_ALGORITHMS = {"phase_vocoder", "psola", "wsola", "rubber_band"}
METRIC_COLUMNS = ["rms_error", "spectral_distance", "spectral_centroid_difference"]


def _load_v3_guard_cases() -> pd.DataFrame:
    path = REPO_ROOT / "artifacts" / "18_selector_decisions.csv"
    if not path.exists():
        raise FileNotFoundError("Experiment 19 expects artifacts/18_selector_decisions.csv. Run Experiment 18 first.")
    decisions = pd.read_csv(path)
    return decisions[decisions["guard"] == "noise_fallback_dry_guard"].copy().reset_index(drop=True)


def _load_reference_results() -> pd.DataFrame:
    path = REPO_ROOT / "artifacts" / "18_results.csv"
    if not path.exists():
        raise FileNotFoundError("Experiment 19 expects artifacts/18_results.csv. Run Experiment 18 first.")
    return pd.read_csv(path)


def _add_reference_composite(rows: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    scored = rows.copy()
    normalized_columns = []
    for column in METRIC_COLUMNS:
        normalized = f"{column}_normalized"
        ref_values = reference[column].to_numpy(dtype=np.float64)
        min_value = float(np.min(ref_values))
        max_value = float(np.max(ref_values))
        values = scored[column].to_numpy(dtype=np.float64)
        if max_value > min_value:
            scored[normalized] = np.clip((values - min_value) / (max_value - min_value), 0.0, 1.5)
        else:
            scored[normalized] = 0.0
        normalized_columns.append(normalized)
    scored["composite_stress"] = scored[normalized_columns].mean(axis=1)
    return scored


def _metric_row(
    source: np.ndarray,
    shifted: np.ndarray,
    guarded: np.ndarray,
    signal_family: str,
    signal_name: str,
    shift: int,
    dry_mix: float,
) -> dict[str, float | int | str]:
    return {
        "signal_family": signal_family,
        "signal_name": signal_name,
        "signal_label": f"{signal_family} / {signal_name}",
        "case_label": f"{signal_family} / {signal_name} {shift:+d}",
        "shift_semitones": shift,
        "dry_mix": dry_mix,
        "algorithm": "preflight_adaptive_v3_guard_sweep",
        "algorithm_label": "Adaptive v3 Guard Sweep",
        "rms_error": rms_error(source, guarded),
        "spectral_distance": spectral_distance(source, guarded),
        "spectral_centroid_difference": spectral_centroid_difference(source, guarded, sr=SR),
        "shift_retention": shift_retention(source, shifted, guarded),
        "dry_similarity": dry_similarity(source, guarded),
        "peak_preservation": peak_preservation(source, guarded),
    }


def _build_summary(results: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        results.groupby("dry_mix", as_index=False)
        .agg(
            catastrophic_count=("catastrophic", "sum"),
            mean_stress=("composite_stress", "mean"),
            max_stress=("composite_stress", "max"),
            mean_shift_retention=("shift_retention", "mean"),
            mean_dry_similarity=("dry_similarity", "mean"),
            mean_peak_preservation=("peak_preservation", "mean"),
        )
        .reset_index(drop=True)
    )
    grouped["catastrophic_count"] = grouped["catastrophic_count"].astype(int)
    grouped["catastrophe_rate"] = grouped["catastrophic_count"] / results["case_label"].nunique()
    grouped["guard_audit_score"] = (
        grouped["mean_shift_retention"] - grouped["mean_stress"] - 0.35 * grouped["catastrophe_rate"]
    )
    return grouped.sort_values(["catastrophic_count", "guard_audit_score"], ascending=[True, False]).reset_index(drop=True)


def _build_case_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case_label, frame in results.groupby("case_label", sort=False):
        zero_cat = frame[~frame["catastrophic"]].copy()
        if zero_cat.empty:
            best = frame.sort_values("composite_stress").iloc[0]
        else:
            zero_cat["case_score"] = zero_cat["shift_retention"] - zero_cat["composite_stress"]
            best = zero_cat.sort_values("case_score", ascending=False).iloc[0]
        rows.append(
            {
                "case_label": case_label,
                "signal_family": best["signal_family"],
                "signal_name": best["signal_name"],
                "shift_semitones": int(best["shift_semitones"]),
                "best_dry_mix": float(best["dry_mix"]),
                "best_composite_stress": float(best["composite_stress"]),
                "best_shift_retention": float(best["shift_retention"]),
                "best_dry_similarity": float(best["dry_similarity"]),
                "best_peak_preservation": float(best["peak_preservation"]),
            }
        )
    return pd.DataFrame(rows)


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    guard_cases = _load_v3_guard_cases()
    reference = _load_reference_results()
    reference_fixed = reference[reference["algorithm"].isin(REFERENCE_ALGORITHMS)].copy()
    threshold = float(reference_fixed["composite_stress"].quantile(0.85))
    atlas = build_signal_atlas(sr=SR, duration=DURATION, seed=7)
    rows = []

    for case in guard_cases.to_dict("records"):
        source = atlas[str(case["signal_name"])]
        algorithm = get_algorithm(str(case["selected_algorithm"]))
        shift = int(case["shift_semitones"])
        shifted = algorithm.pitch_shift(source, sr=SR, n_steps=shift)
        for dry_mix in DRY_MIXES:
            guarded = blend_with_source(source, shifted, dry_mix=dry_mix)
            rows.append(
                _metric_row(
                    source,
                    shifted,
                    guarded,
                    str(case["signal_family"]),
                    str(case["signal_name"]),
                    shift,
                    dry_mix,
                )
            )

    results = _add_reference_composite(pd.DataFrame(rows), reference)
    results["catastrophic_threshold"] = threshold
    results["catastrophic"] = results["composite_stress"] >= threshold
    summary = _build_summary(results)
    case_summary = _build_case_summary(results)

    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(artifacts_dir / "19_guard_audit_results.csv", index=False)
    summary.to_csv(artifacts_dir / "19_guard_audit_summary.csv", index=False)
    case_summary.to_csv(artifacts_dir / "19_guard_audit_case_summary.csv", index=False)
    save_guard_audit_plot(results, summary, artifacts_dir / "19_guard_audit_plot.png")
    return results, summary, case_summary


def main() -> None:
    results, summary, case_summary = run_experiment()
    print(f"Wrote {len(results)} rows to artifacts/19_guard_audit_results.csv")
    print("Wrote artifacts/19_guard_audit_summary.csv")
    print("Wrote artifacts/19_guard_audit_case_summary.csv")
    print("Wrote artifacts/19_guard_audit_plot.png")
    print("Question: Does the fallback guard avoid catastrophes without suppressing the shift?")
    print(summary.to_string(index=False))
    print(case_summary.to_string(index=False))


if __name__ == "__main__":
    main()
