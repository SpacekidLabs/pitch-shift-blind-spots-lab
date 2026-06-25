from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.phase_vocoder import pitch_shift_phase_vocoder
from algorithms.rubber_band import pitch_shift_rubber_band
from metrics.audio_metrics import rms_error, spectral_centroid_difference, spectral_distance
from metrics.guard_metrics import blend_with_source, dry_similarity, peak_preservation, shift_retention
from signals.library import build_signal_atlas
from visualization.guard_type_plot import save_guard_type_plot


SR = 22050
DURATION = 2.0
REFERENCE_ALGORITHMS = {"phase_vocoder", "psola", "wsola", "rubber_band"}
METRIC_COLUMNS = ["rms_error", "spectral_distance", "spectral_centroid_difference"]

GUARD_TYPES = [
    ("no_guard", "No Guard"),
    ("dry_wet_010", "Dry/Wet 0.10"),
    ("dry_wet_040", "Dry/Wet 0.40"),
    ("reduced_shift_080", "Reduced Shift 80%"),
    ("spectral_smooth", "Spectral Smoothing"),
    ("rubberband_crossfade", "Rubber Band Crossfade"),
    ("rubberband_fallback", "Rubber Band Fallback"),
]


def _load_guard_cases() -> pd.DataFrame:
    path = REPO_ROOT / "artifacts" / "18_selector_decisions.csv"
    if not path.exists():
        raise FileNotFoundError("Experiment 21 expects artifacts/18_selector_decisions.csv. Run Experiment 18 first.")
    decisions = pd.read_csv(path)
    return decisions[decisions["guard"] == "noise_fallback_dry_guard"].copy().reset_index(drop=True)


def _load_reference_results() -> pd.DataFrame:
    path = REPO_ROOT / "artifacts" / "18_results.csv"
    if not path.exists():
        raise FileNotFoundError("Experiment 21 expects artifacts/18_results.csv. Run Experiment 18 first.")
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


def _moving_average(y: np.ndarray, width: int) -> np.ndarray:
    width = max(3, int(width))
    if width % 2 == 0:
        width += 1
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(np.asarray(y, dtype=np.float64), kernel, mode="same")


def _spectral_smooth_guard(shifted: np.ndarray, sr: int) -> np.ndarray:
    low = _moving_average(shifted, width=max(31, sr // 320))
    return 0.72 * np.asarray(shifted, dtype=np.float64) + 0.28 * low


def _apply_guard(source: np.ndarray, full_shifted: np.ndarray, shift: int, guard_type: str) -> tuple[np.ndarray, str]:
    if guard_type == "no_guard":
        return full_shifted, "baseline_phase_vocoder"
    if guard_type == "dry_wet_010":
        return blend_with_source(source, full_shifted, dry_mix=0.10), "small_dry_blend"
    if guard_type == "dry_wet_040":
        return blend_with_source(source, full_shifted, dry_mix=0.40), "legacy_heavy_dry_blend"
    if guard_type == "reduced_shift_080":
        return pitch_shift_phase_vocoder(source, sr=SR, n_steps=0.80 * shift), "reduced_shift_strength"
    if guard_type == "spectral_smooth":
        return _spectral_smooth_guard(full_shifted, sr=SR), "smoothed_shifted_output"
    if guard_type == "rubberband_crossfade":
        rubber = pitch_shift_rubber_band(source, sr=SR, n_steps=shift)
        return 0.50 * full_shifted[: len(rubber)] + 0.50 * rubber, "phase_vocoder_rubberband_blend"
    if guard_type == "rubberband_fallback":
        return pitch_shift_rubber_band(source, sr=SR, n_steps=shift), "alternate_algorithm_fallback"
    raise ValueError(f"Unknown guard type: {guard_type}")


def _metric_row(
    source: np.ndarray,
    full_shifted: np.ndarray,
    guarded: np.ndarray,
    signal_family: str,
    signal_name: str,
    shift: int,
    guard_type: str,
    guard_label: str,
    implementation: str,
) -> dict[str, float | int | str]:
    return {
        "signal_family": signal_family,
        "signal_name": signal_name,
        "signal_label": f"{signal_family} / {signal_name}",
        "case_label": f"{signal_family} / {signal_name} {shift:+d}",
        "shift_semitones": shift,
        "guard_type": guard_type,
        "guard_label": guard_label,
        "implementation": implementation,
        "rms_error": rms_error(source, guarded),
        "spectral_distance": spectral_distance(source, guarded),
        "spectral_centroid_difference": spectral_centroid_difference(source, guarded, sr=SR),
        "shift_retention": shift_retention(source, full_shifted, guarded),
        "dry_similarity": dry_similarity(source, guarded),
        "peak_preservation": peak_preservation(source, guarded),
    }


def _build_summary(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby(["guard_type", "guard_label"], as_index=False)
        .agg(
            case_count=("case_label", "nunique"),
            catastrophic_count=("catastrophic", "sum"),
            mean_stress=("composite_stress", "mean"),
            max_stress=("composite_stress", "max"),
            mean_shift_retention=("shift_retention", "mean"),
            mean_dry_similarity=("dry_similarity", "mean"),
            mean_peak_preservation=("peak_preservation", "mean"),
        )
        .reset_index(drop=True)
    )
    summary["catastrophic_count"] = summary["catastrophic_count"].astype(int)
    summary["catastrophe_rate"] = summary["catastrophic_count"] / results["case_label"].nunique()
    summary["retention_quality"] = np.minimum(summary["mean_shift_retention"], 1.0)
    summary["retention_overshoot"] = np.maximum(summary["mean_shift_retention"] - 1.0, 0.0)
    summary["guard_type_score"] = (
        summary["retention_quality"]
        - summary["mean_stress"]
        - 0.90 * summary["catastrophe_rate"]
        - 0.25 * summary["retention_overshoot"]
    )
    return summary.sort_values(["catastrophic_count", "guard_type_score"], ascending=[True, False]).reset_index(drop=True)


def _build_case_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case_label, frame in results.groupby("case_label", sort=False):
        safe = frame[~frame["catastrophic"]].copy()
        if safe.empty:
            best = frame.sort_values("composite_stress").iloc[0]
        else:
            safe["case_score"] = (
                np.minimum(safe["shift_retention"], 1.0)
                - safe["composite_stress"]
                - 0.25 * np.maximum(safe["shift_retention"] - 1.0, 0.0)
            )
            best = safe.sort_values("case_score", ascending=False).iloc[0]
        rows.append(
            {
                "case_label": case_label,
                "best_guard_label": best["guard_label"],
                "best_composite_stress": float(best["composite_stress"]),
                "best_shift_retention": float(best["shift_retention"]),
                "best_dry_similarity": float(best["dry_similarity"]),
            }
        )
    return pd.DataFrame(rows)


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    guard_cases = _load_guard_cases()
    reference = _load_reference_results()
    reference_fixed = reference[reference["algorithm"].isin(REFERENCE_ALGORITHMS)].copy()
    threshold = float(reference_fixed["composite_stress"].quantile(0.85))
    atlas = build_signal_atlas(sr=SR, duration=DURATION, seed=7)
    rows = []

    for case in guard_cases.to_dict("records"):
        signal_name = str(case["signal_name"])
        signal_family = str(case["signal_family"])
        shift = int(case["shift_semitones"])
        source = atlas[signal_name]
        full_shifted = pitch_shift_phase_vocoder(source, sr=SR, n_steps=shift)
        for guard_type, guard_label in GUARD_TYPES:
            guarded, implementation = _apply_guard(source, full_shifted, shift=shift, guard_type=guard_type)
            rows.append(
                _metric_row(
                    source,
                    full_shifted,
                    guarded,
                    signal_family,
                    signal_name,
                    shift,
                    guard_type,
                    guard_label,
                    implementation,
                )
            )

    results = _add_reference_composite(pd.DataFrame(rows), reference)
    results["catastrophic_threshold"] = threshold
    results["catastrophic"] = results["composite_stress"] >= threshold
    summary = _build_summary(results)
    case_summary = _build_case_summary(results)

    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(artifacts_dir / "21_guard_type_results.csv", index=False)
    summary.to_csv(artifacts_dir / "21_guard_type_summary.csv", index=False)
    case_summary.to_csv(artifacts_dir / "21_guard_type_case_summary.csv", index=False)
    save_guard_type_plot(summary, artifacts_dir / "21_guard_type_plot.png")
    return results, summary, case_summary


def main() -> None:
    results, summary, case_summary = run_experiment()
    print(f"Wrote {len(results)} rows to artifacts/21_guard_type_results.csv")
    print("Wrote artifacts/21_guard_type_summary.csv")
    print("Wrote artifacts/21_guard_type_case_summary.csv")
    print("Wrote artifacts/21_guard_type_plot.png")
    print("Question: What should the fallback guard do?")
    print(summary.to_string(index=False))
    print(case_summary.to_string(index=False))


if __name__ == "__main__":
    main()
