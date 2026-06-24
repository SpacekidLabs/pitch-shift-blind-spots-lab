from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.adaptive_preflight import pitch_shift_preflight_adaptive_v2
from metrics.audio_metrics import rms_error, spectral_centroid_difference, spectral_distance
from metrics.disagreement import add_composite_stress
from visualization.selectivity_frontier_plot import save_selectivity_frontier_plot


SR = 22050
DURATION = 2.0
ADAPTIVE_NAME = "preflight_adaptive_v2"
ADAPTIVE_LABEL = "Preflight Adaptive v2"


def _normalize(y: np.ndarray, peak: float = 0.9) -> np.ndarray:
    max_abs = float(np.max(np.abs(y))) if y.size else 0.0
    if max_abs <= 0.0:
        return y.copy()
    return (peak / max_abs) * y


def _vibrato_tone(base_hz: float, depth_semitones: float, rate_hz: float, sr: int, duration: float) -> np.ndarray:
    t = np.arange(int(round(sr * duration)), dtype=np.float64) / float(sr)
    semitone_offset = depth_semitones * np.sin(2 * np.pi * rate_hz * t)
    inst_freq = base_hz * (2.0 ** (semitone_offset / 12.0))
    phase = 2 * np.pi * np.cumsum(inst_freq) / sr
    return _normalize(np.sin(phase))


def _metric_row(source: np.ndarray, shifted: np.ndarray, case: pd.Series) -> dict[str, float | int | str]:
    depth = float(case["depth_semitones"])
    rate = float(case["rate_hz"])
    shift = int(case["shift_semitones"])
    return {
        "signal_family": "subtle_modulation",
        "signal_name": f"vibrato_depth_{depth:g}_rate_{rate:g}",
        "signal_label": f"{depth:g} st / {rate:g} Hz",
        "depth_semitones": depth,
        "rate_hz": rate,
        "shift_semitones": shift,
        "algorithm": ADAPTIVE_NAME,
        "algorithm_label": ADAPTIVE_LABEL,
        "implementation": "selective source-only preflight selector",
        "rms_error": rms_error(source, shifted),
        "spectral_distance": spectral_distance(source, shifted),
        "spectral_centroid_difference": spectral_centroid_difference(source, shifted, sr=SR),
    }


def _build_cases(scored: pd.DataFrame, decisions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    fixed = scored[~scored["algorithm"].isin([ADAPTIVE_NAME, "preflight_adaptive_v1"])]
    v1 = scored[scored["algorithm"] == "preflight_adaptive_v1"]
    v2 = scored[scored["algorithm"] == ADAPTIVE_NAME]
    rows = []
    for _, decision in decisions.iterrows():
        key = (
            (fixed["depth_semitones"] == decision["depth_semitones"])
            & (fixed["rate_hz"] == decision["rate_hz"])
            & (fixed["shift_semitones"] == decision["shift_semitones"])
        )
        fixed_case = fixed[key]
        v2_case = v2[
            (v2["depth_semitones"] == decision["depth_semitones"])
            & (v2["rate_hz"] == decision["rate_hz"])
            & (v2["shift_semitones"] == decision["shift_semitones"])
        ].iloc[0]
        v1_case = v1[
            (v1["depth_semitones"] == decision["depth_semitones"])
            & (v1["rate_hz"] == decision["rate_hz"])
            & (v1["shift_semitones"] == decision["shift_semitones"])
        ]
        v1_stress = float(v1_case["composite_stress"].iloc[0]) if not v1_case.empty else np.nan
        worst = fixed_case.sort_values("composite_stress", ascending=False).iloc[0]
        fixed_danger_count = int(np.sum(fixed_case["composite_stress"].to_numpy(dtype=np.float64) >= threshold))
        v2_catastrophic = float(v2_case["composite_stress"]) >= threshold
        rows.append(
            {
                **decision.to_dict(),
                "catastrophic_threshold": threshold,
                "fixed_danger_count": fixed_danger_count,
                "fixed_danger_any": fixed_danger_count > 0,
                "worst_fixed_algorithm_label": worst["algorithm_label"],
                "worst_fixed_composite_stress": float(worst["composite_stress"]),
                "v1_composite_stress": v1_stress,
                "v2_composite_stress": float(v2_case["composite_stress"]),
                "v2_catastrophic": v2_catastrophic,
                "fixed_danger_avoided": fixed_danger_count > 0 and not v2_catastrophic,
            }
        )
    return pd.DataFrame(rows).sort_values(["rate_hz", "depth_semitones", "shift_semitones"]).reset_index(drop=True)


def _build_summary(cases: pd.DataFrame) -> pd.DataFrame:
    fixed_danger = cases["fixed_danger_any"].to_numpy(dtype=bool)
    v2_fail = cases["v2_catastrophic"].to_numpy(dtype=bool)
    return pd.DataFrame(
        [
            {
                "cases": int(len(cases)),
                "fixed_danger_cases": int(np.sum(fixed_danger)),
                "v2_catastrophic_cases": int(np.sum(v2_fail)),
                "v2_avoided_fixed_danger_cases": int(np.sum(fixed_danger & ~v2_fail)),
                "v2_missed_fixed_danger_cases": int(np.sum(fixed_danger & v2_fail)),
                "v2_introduced_catastrophic_cases": int(np.sum(~fixed_danger & v2_fail)),
                "v2_avoidance_rate": float(np.sum(fixed_danger & ~v2_fail) / max(np.sum(fixed_danger), 1)),
                "v2_phase_vocoder_count": int(np.sum(cases["selected_algorithm"] == "phase_vocoder")),
                "v2_psola_count": int(np.sum(cases["selected_algorithm"] == "psola")),
                "v2_rubberband_count": int(np.sum(cases["selected_algorithm"] == "rubber_band")),
                "v2_mean_stress": float(cases["v2_composite_stress"].mean()),
                "v1_mean_stress": float(cases["v1_composite_stress"].mean()),
            }
        ]
    )


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fixed_path = REPO_ROOT / "artifacts" / "11_results.csv"
    v1_path = REPO_ROOT / "artifacts" / "13_results.csv"
    source_cases_path = REPO_ROOT / "artifacts" / "11_modulation_cases.csv"
    if not fixed_path.exists() or not v1_path.exists() or not source_cases_path.exists():
        raise FileNotFoundError("Experiment 15 expects Experiment 11 and 13 artifacts.")

    fixed_results = pd.read_csv(fixed_path)
    v1_results = pd.read_csv(v1_path)
    v1_results = v1_results[v1_results["algorithm"] == "preflight_adaptive_v1"]
    source_cases = pd.read_csv(source_cases_path)[["depth_semitones", "rate_hz", "shift_semitones"]].drop_duplicates()
    v2_rows = []
    decision_rows = []
    for _, case in source_cases.iterrows():
        source = _vibrato_tone(220.0, float(case["depth_semitones"]), float(case["rate_hz"]), sr=SR, duration=DURATION)
        shifted, analysis = pitch_shift_preflight_adaptive_v2(source, sr=SR, n_steps=int(case["shift_semitones"]))
        v2_rows.append(_metric_row(source, shifted, case))
        decision_rows.append(
            {
                "depth_semitones": float(case["depth_semitones"]),
                "rate_hz": float(case["rate_hz"]),
                "shift_semitones": int(case["shift_semitones"]),
                **asdict(analysis),
            }
        )

    scored = add_composite_stress(pd.concat([fixed_results, v1_results, pd.DataFrame(v2_rows)], ignore_index=True))
    fixed_scored = scored[~scored["algorithm"].isin([ADAPTIVE_NAME, "preflight_adaptive_v1"])]
    threshold = float(fixed_scored["composite_stress"].quantile(0.85))
    decisions = pd.DataFrame(decision_rows)
    cases = _build_cases(scored, decisions, threshold=threshold)
    summary = _build_summary(cases)

    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(artifacts_dir / "15_results.csv", index=False)
    cases.to_csv(artifacts_dir / "15_selective_adaptive_v2_cases.csv", index=False)
    summary.to_csv(artifacts_dir / "15_selective_adaptive_v2_summary.csv", index=False)

    frontier_like = pd.DataFrame(
        [
            {
                "policy_id": "preflight_adaptive_v1",
                "phase_vocoder_count": 140,
                "mean_selected_stress": float(summary.iloc[0]["v1_mean_stress"]),
                "adaptive_catastrophic_cases": 0,
            },
            {
                "policy_id": "preflight_adaptive_v2",
                "phase_vocoder_count": int(summary.iloc[0]["v2_phase_vocoder_count"]),
                "mean_selected_stress": float(summary.iloc[0]["v2_mean_stress"]),
                "adaptive_catastrophic_cases": int(summary.iloc[0]["v2_catastrophic_cases"]),
            },
        ]
    )
    frontier_summary = pd.DataFrame(
        [
            {
                "policy_count": 2,
                "zero_catastrophic_policy_count": int(np.sum(frontier_like["adaptive_catastrophic_cases"] == 0)),
                "most_selective_safe_policy_id": "preflight_adaptive_v2",
                "most_selective_phase_vocoder_count": int(summary.iloc[0]["v2_phase_vocoder_count"]),
                "most_selective_mean_stress": float(summary.iloc[0]["v2_mean_stress"]),
                "lowest_mean_safe_policy_id": "preflight_adaptive_v1",
                "lowest_mean_stress": float(summary.iloc[0]["v1_mean_stress"]),
            }
        ]
    )
    save_selectivity_frontier_plot(
        frontier_like,
        frontier_summary,
        artifacts_dir / "15_selective_adaptive_v2_comparison.png",
        title="Exp15 Selective Adaptive v2",
        subtitle="Conservative v1 vs. selective v2 on the subtle-modulation grid.",
    )
    return scored, cases, summary


def main() -> None:
    results, cases, summary = run_experiment()
    print(f"Wrote {len(results)} rows to artifacts/15_results.csv")
    print(f"Wrote {len(cases)} case rows to artifacts/15_selective_adaptive_v2_cases.csv")
    print("Wrote artifacts/15_selective_adaptive_v2_summary.csv")
    print("Wrote artifacts/15_selective_adaptive_v2_comparison.png")
    print("Question: Does the most selective safe frontier policy still work when instantiated?")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
