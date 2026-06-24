from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visualization.selectivity_frontier_plot import save_selectivity_frontier_plot


THRESHOLDS = [0.28, 0.32, 0.36, 0.40, 0.45, 0.50, 0.55]
MICRO_BONUSES = [0.0, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
MICRO_RMS_THRESHOLDS = [0.5, 0.7, 1.0, 2.0, 5.0, 10.0, 20.0]
MICRO_STRENGTH_THRESHOLDS = [0.30, 0.35, 0.40]


def _micro_modulation_condition(row: pd.Series, rms_threshold: float, strength_threshold: float) -> bool:
    return (
        float(row["pitch_valid_fraction"]) > 0.95
        and float(row["spectral_flatness"]) < 0.02
        and float(row["pitch_modulation_rms_cents"]) > rms_threshold
        and float(row["pitch_modulation_peak_strength"]) > strength_threshold
        and 1.0 <= float(row["pitch_modulation_peak_rate_hz"]) <= 12.0
    )


def _select_algorithm(row: pd.Series, threshold: float, micro_bonus: float, rms_threshold: float, strength_threshold: float) -> tuple[str, float, bool]:
    score = float(row["baseline_risk_score"])
    if _micro_modulation_condition(row, rms_threshold=rms_threshold, strength_threshold=strength_threshold):
        score = min(1.0, score + micro_bonus)
    high_risk = score >= threshold
    if high_risk:
        return "phase_vocoder", score, True
    if abs(int(row["shift_semitones"])) <= 7:
        return "psola", score, False
    return "rubber_band", score, False


def _stress_for_selection(fixed_results: pd.DataFrame, row: pd.Series, algorithm: str) -> float:
    selected = fixed_results[
        (fixed_results["depth_semitones"] == row["depth_semitones"])
        & (fixed_results["rate_hz"] == row["rate_hz"])
        & (fixed_results["shift_semitones"] == row["shift_semitones"])
        & (fixed_results["algorithm"] == algorithm)
    ]
    return float(selected["composite_stress"].iloc[0])


def _simulate_policy(
    policy_id: str,
    fixed_results: pd.DataFrame,
    predictions: pd.DataFrame,
    catastrophic_threshold: float,
    threshold: float,
    micro_bonus: float,
    rms_threshold: float,
    strength_threshold: float,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str | bool]]]:
    rows = []
    for _, row in predictions.iterrows():
        algorithm, risk_score, high_risk = _select_algorithm(
            row,
            threshold=threshold,
            micro_bonus=micro_bonus,
            rms_threshold=rms_threshold,
            strength_threshold=strength_threshold,
        )
        selected_stress = _stress_for_selection(fixed_results, row, algorithm)
        fixed_danger = bool(row["catastrophic_any_fixed"])
        selected_catastrophic = selected_stress >= catastrophic_threshold
        rows.append(
            {
                "policy_id": policy_id,
                "depth_semitones": float(row["depth_semitones"]),
                "rate_hz": float(row["rate_hz"]),
                "shift_semitones": int(row["shift_semitones"]),
                "selected_algorithm": algorithm,
                "selected_stress": selected_stress,
                "selected_catastrophic": selected_catastrophic,
                "fixed_danger": fixed_danger,
                "fixed_catastrophic_count": int(row["catastrophic_fixed_count"]),
                "safe_mode": high_risk,
                "policy_risk_score": risk_score,
            }
        )

    selected = pd.DataFrame(rows)
    selected_values = selected["selected_stress"].to_numpy(dtype=np.float64)
    fixed_danger = selected["fixed_danger"].to_numpy(dtype=bool)
    selected_fail = selected["selected_catastrophic"].to_numpy(dtype=bool)
    summary = {
        "policy_id": policy_id,
        "threshold": threshold,
        "micro_bonus": micro_bonus,
        "micro_rms_threshold": rms_threshold,
        "micro_strength_threshold": strength_threshold,
        "case_count": int(len(selected)),
        "fixed_danger_cases": int(np.sum(fixed_danger)),
        "adaptive_catastrophic_cases": int(np.sum(selected_fail)),
        "avoided_fixed_danger_cases": int(np.sum(fixed_danger & ~selected_fail)),
        "missed_fixed_danger_cases": int(np.sum(fixed_danger & selected_fail)),
        "introduced_catastrophic_cases": int(np.sum(~fixed_danger & selected_fail)),
        "phase_vocoder_count": int(np.sum(selected["selected_algorithm"] == "phase_vocoder")),
        "psola_count": int(np.sum(selected["selected_algorithm"] == "psola")),
        "rubberband_count": int(np.sum(selected["selected_algorithm"] == "rubber_band")),
        "wsola_count": int(np.sum(selected["selected_algorithm"] == "wsola")),
        "mean_selected_stress": float(np.mean(selected_values)),
        "max_selected_stress": float(np.max(selected_values)),
        "p90_selected_stress": float(np.percentile(selected_values, 90)),
    }
    return summary, rows


def _build_experiment_summary(policies: pd.DataFrame) -> pd.DataFrame:
    safe = policies[policies["adaptive_catastrophic_cases"] == 0].copy()
    most_selective = safe.sort_values(["phase_vocoder_count", "mean_selected_stress"]).iloc[0]
    lowest_mean = safe.sort_values(["mean_selected_stress", "phase_vocoder_count"]).iloc[0]
    return pd.DataFrame(
        [
            {
                "policy_count": int(len(policies)),
                "zero_catastrophic_policy_count": int(len(safe)),
                "most_selective_safe_policy_id": most_selective["policy_id"],
                "most_selective_phase_vocoder_count": int(most_selective["phase_vocoder_count"]),
                "most_selective_psola_count": int(most_selective["psola_count"]),
                "most_selective_rubberband_count": int(most_selective["rubberband_count"]),
                "most_selective_mean_stress": float(most_selective["mean_selected_stress"]),
                "lowest_mean_safe_policy_id": lowest_mean["policy_id"],
                "lowest_mean_phase_vocoder_count": int(lowest_mean["phase_vocoder_count"]),
                "lowest_mean_stress": float(lowest_mean["mean_selected_stress"]),
            }
        ]
    )


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fixed_path = REPO_ROOT / "artifacts" / "11_results.csv"
    predictions_path = REPO_ROOT / "artifacts" / "12_micro_modulation_predictions.csv"
    if not fixed_path.exists() or not predictions_path.exists():
        raise FileNotFoundError("Experiment 14 expects Experiment 11 and 12 artifacts.")

    fixed_results = pd.read_csv(fixed_path)
    predictions = pd.read_csv(predictions_path)
    catastrophic_threshold = float(fixed_results["composite_stress"].quantile(0.85))
    policy_rows = []
    best_case_rows: list[dict[str, float | int | str | bool]] = []
    all_case_rows_by_policy: dict[str, list[dict[str, float | int | str | bool]]] = {}

    for threshold in THRESHOLDS:
        for micro_bonus in MICRO_BONUSES:
            for rms_threshold in MICRO_RMS_THRESHOLDS:
                for strength_threshold in MICRO_STRENGTH_THRESHOLDS:
                    policy_id = f"thr{threshold:.2f}_bonus{micro_bonus:.2f}_rms{rms_threshold:g}_str{strength_threshold:.2f}"
                    summary, case_rows = _simulate_policy(
                        policy_id=policy_id,
                        fixed_results=fixed_results,
                        predictions=predictions,
                        catastrophic_threshold=catastrophic_threshold,
                        threshold=threshold,
                        micro_bonus=micro_bonus,
                        rms_threshold=rms_threshold,
                        strength_threshold=strength_threshold,
                    )
                    policy_rows.append(summary)
                    all_case_rows_by_policy[policy_id] = case_rows

    policies = pd.DataFrame(policy_rows).sort_values(
        ["adaptive_catastrophic_cases", "phase_vocoder_count", "mean_selected_stress"]
    ).reset_index(drop=True)
    experiment_summary = _build_experiment_summary(policies)
    best_policy_id = str(experiment_summary.iloc[0]["most_selective_safe_policy_id"])
    best_case_rows = all_case_rows_by_policy[best_policy_id]
    best_cases = pd.DataFrame(best_case_rows)

    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    policies.to_csv(artifacts_dir / "14_policy_frontier.csv", index=False)
    best_cases.to_csv(artifacts_dir / "14_best_policy_cases.csv", index=False)
    experiment_summary.to_csv(artifacts_dir / "14_selectivity_summary.csv", index=False)
    save_selectivity_frontier_plot(policies, experiment_summary, artifacts_dir / "14_selectivity_frontier.png")
    return policies, best_cases, experiment_summary


def main() -> None:
    policies, best_cases, summary = run_experiment()
    print(f"Wrote {len(policies)} policy rows to artifacts/14_policy_frontier.csv")
    print(f"Wrote {len(best_cases)} best-policy cases to artifacts/14_best_policy_cases.csv")
    print("Wrote artifacts/14_selectivity_summary.csv")
    print("Wrote artifacts/14_selectivity_frontier.png")
    print("Question: Can adaptive selection recover algorithm diversity without catastrophic failures?")
    print(summary.to_string(index=False))
    print(policies.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
