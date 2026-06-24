from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.adaptive_preflight import NOISE_FALLBACK_DRY_MIX, analyze_selective_preflight_adaptive_state
from algorithms.registry import get_algorithm
from metrics.audio_metrics import rms_error, spectral_centroid_difference, spectral_distance
from metrics.guard_metrics import blend_with_source, shift_retention
from signals.library import SIGNAL_SPECS, build_signal_atlas
from visualization.guard_specificity_plot import save_guard_specificity_plot


SR = 22050
DURATION = 2.0
SHIFT_STEPS = [3, 7, 12, -12]
REFERENCE_ALGORITHMS = {"phase_vocoder", "psola", "wsola", "rubber_band"}
METRIC_COLUMNS = ["rms_error", "spectral_distance", "spectral_centroid_difference"]

POLICIES = [
    ("no_guard", "No Guard"),
    ("current_v3", "Current v3"),
    ("noise_only", "Noise Only"),
    ("noise_plus_transient", "Noise + Transient"),
    ("forced_everywhere", "Forced Everywhere"),
]


def _load_reference_results() -> pd.DataFrame:
    path = REPO_ROOT / "artifacts" / "18_results.csv"
    if not path.exists():
        raise FileNotFoundError("Experiment 20 expects artifacts/18_results.csv. Run Experiment 18 first.")
    return pd.read_csv(path)


def _reason_parts(reasons: str) -> set[str]:
    if reasons == "stable_source":
        return set()
    return set(str(reasons).split("|"))


def _should_guard(policy: str, signal_family: str, shift: int, analysis_reasons: str, selected_algorithm: str) -> bool:
    large_shift = abs(shift) >= 12
    if selected_algorithm != "phase_vocoder":
        return policy == "forced_everywhere"
    reasons = _reason_parts(analysis_reasons)
    if policy == "no_guard":
        return False
    if policy == "forced_everywhere":
        return True
    if not large_shift:
        return False
    if policy == "noise_only":
        return signal_family == "noise" and "noise_like_spectrum" in reasons
    if policy == "noise_plus_transient":
        return (
            signal_family in {"noise", "transient"}
            and ("noise_like_spectrum" in reasons or "sparse_transients" in reasons or "untracked_pitch" in reasons)
        )
    if policy == "current_v3":
        return "noise_like_spectrum" in reasons and "sparse_transients" not in reasons
    raise ValueError(f"Unknown guard policy: {policy}")


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
    spec_name: str,
    spec_family: str,
    shift: int,
    policy: str,
    policy_label: str,
    guard_applied: bool,
    analysis: object,
) -> dict[str, float | int | str | bool]:
    return {
        **asdict(analysis),
        "signal_family": spec_family,
        "signal_name": spec_name,
        "signal_label": f"{spec_family} / {spec_name}",
        "case_id": f"{spec_name}:{shift}",
        "case_label": f"{spec_family} / {spec_name} {shift:+d}",
        "shift_semitones": shift,
        "policy": policy,
        "policy_label": policy_label,
        "selected_algorithm": analysis.selected_algorithm,
        "selected_algorithm_label": analysis.selected_algorithm_label,
        "guard_applied": guard_applied,
        "dry_mix": NOISE_FALLBACK_DRY_MIX if guard_applied else 0.0,
        "preflight_risk_score": analysis.preflight_risk_score,
        "preflight_reasons": analysis.preflight_reasons,
        "rms_error": rms_error(source, guarded),
        "spectral_distance": spectral_distance(source, guarded),
        "spectral_centroid_difference": spectral_centroid_difference(source, guarded, sr=SR),
        "shift_retention": shift_retention(source, shifted, guarded),
    }


def _build_policy_summary(results: pd.DataFrame) -> pd.DataFrame:
    baseline = results[results["policy"] == "no_guard"][["case_id", "catastrophic"]].rename(
        columns={"catastrophic": "baseline_catastrophic"}
    )
    merged = results.merge(baseline, on="case_id", how="left")
    merged["rescued"] = merged["baseline_catastrophic"] & ~merged["catastrophic"]
    merged["missed_guard"] = merged["baseline_catastrophic"] & merged["catastrophic"]
    merged["unnecessary_guard"] = merged["guard_applied"] & ~merged["baseline_catastrophic"]
    total_cases = merged["case_id"].nunique()
    summary = (
        merged.groupby(["policy", "policy_label"], as_index=False)
        .agg(
            case_count=("case_id", "nunique"),
            guarded_count=("guard_applied", "sum"),
            catastrophic_count=("catastrophic", "sum"),
            rescued_count=("rescued", "sum"),
            missed_guard_count=("missed_guard", "sum"),
            unnecessary_guard_count=("unnecessary_guard", "sum"),
            mean_stress=("composite_stress", "mean"),
            max_stress=("composite_stress", "max"),
            mean_shift_retention=("shift_retention", "mean"),
        )
        .reset_index(drop=True)
    )
    for column in ["guarded_count", "catastrophic_count", "rescued_count", "missed_guard_count", "unnecessary_guard_count"]:
        summary[column] = summary[column].astype(int)
    summary["catastrophe_rate"] = summary["catastrophic_count"] / total_cases
    summary["unnecessary_guard_rate"] = summary["unnecessary_guard_count"] / total_cases
    summary["guard_specificity_score"] = (
        summary["mean_shift_retention"]
        - summary["mean_stress"]
        - 0.35 * summary["catastrophe_rate"]
        - 0.05 * summary["unnecessary_guard_rate"]
    )
    return summary.sort_values(["catastrophic_count", "guard_specificity_score"], ascending=[True, False]).reset_index(drop=True)


def _build_signal_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (signal_family, signal_name, policy_label), frame in results.groupby(["signal_family", "signal_name", "policy_label"]):
        rows.append(
            {
                "signal_family": signal_family,
                "signal_name": signal_name,
                "signal_label": f"{signal_family} / {signal_name}",
                "policy_label": policy_label,
                "guarded_count": int(frame["guard_applied"].sum()),
                "catastrophic_count": int(frame["catastrophic"].sum()),
                "mean_stress": float(frame["composite_stress"].mean()),
                "mean_shift_retention": float(frame["shift_retention"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["signal_family", "signal_name", "policy_label"]).reset_index(drop=True)


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reference = _load_reference_results()
    reference_fixed = reference[reference["algorithm"].isin(REFERENCE_ALGORITHMS)].copy()
    threshold = float(reference_fixed["composite_stress"].quantile(0.85))
    atlas = build_signal_atlas(sr=SR, duration=DURATION, seed=7)
    rows = []

    for spec in SIGNAL_SPECS:
        source = atlas[spec.name]
        for shift in SHIFT_STEPS:
            analysis = analyze_selective_preflight_adaptive_state(source, sr=SR, n_steps=shift)
            shifted = get_algorithm(analysis.selected_algorithm).pitch_shift(source, sr=SR, n_steps=shift)
            for policy, policy_label in POLICIES:
                guard_applied = _should_guard(
                    policy,
                    spec.family,
                    shift,
                    analysis.preflight_reasons,
                    analysis.selected_algorithm,
                )
                guarded = blend_with_source(source, shifted, NOISE_FALLBACK_DRY_MIX) if guard_applied else shifted
                rows.append(_metric_row(source, shifted, guarded, spec.name, spec.family, shift, policy, policy_label, guard_applied, analysis))

    results = _add_reference_composite(pd.DataFrame(rows), reference)
    results["catastrophic_threshold"] = threshold
    results["catastrophic"] = results["composite_stress"] >= threshold
    policy_summary = _build_policy_summary(results)
    signal_summary = _build_signal_summary(results)

    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(artifacts_dir / "20_guard_specificity_results.csv", index=False)
    policy_summary.to_csv(artifacts_dir / "20_guard_policy_summary.csv", index=False)
    signal_summary.to_csv(artifacts_dir / "20_guard_signal_summary.csv", index=False)
    save_guard_specificity_plot(policy_summary, artifacts_dir / "20_guard_specificity_plot.png")
    return results, policy_summary, signal_summary


def main() -> None:
    results, policy_summary, signal_summary = run_experiment()
    print(f"Wrote {len(results)} rows to artifacts/20_guard_specificity_results.csv")
    print("Wrote artifacts/20_guard_policy_summary.csv")
    print("Wrote artifacts/20_guard_signal_summary.csv")
    print("Wrote artifacts/20_guard_specificity_plot.png")
    print("Question: Does the guard help only where it should?")
    print(policy_summary.to_string(index=False))
    print(signal_summary.to_string(index=False))


if __name__ == "__main__":
    main()
