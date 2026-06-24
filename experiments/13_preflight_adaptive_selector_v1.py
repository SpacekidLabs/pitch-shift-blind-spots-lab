from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.adaptive_preflight import pitch_shift_preflight_adaptive_v1
from metrics.audio_metrics import rms_error, spectral_centroid_difference, spectral_distance
from metrics.disagreement import add_composite_stress
from visualization.modulation_trap_plot import save_modulation_trap_plot


SR = 22050
DURATION = 2.0
ADAPTIVE_NAME = "preflight_adaptive_v1"
ADAPTIVE_LABEL = "Preflight Adaptive v1"


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


def _adaptive_metric_row(
    source: np.ndarray,
    shifted: np.ndarray,
    case: pd.Series,
) -> dict[str, float | int | str]:
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
        "implementation": "source-only modulation-aware preflight selector",
        "rms_error": rms_error(source, shifted),
        "spectral_distance": spectral_distance(source, shifted),
        "spectral_centroid_difference": spectral_centroid_difference(source, shifted, sr=SR),
    }


def _outcome(fixed_catastrophic: bool, adaptive_catastrophic: bool, safe_mode: bool) -> str:
    if fixed_catastrophic and not adaptive_catastrophic:
        return "caught"
    if fixed_catastrophic and adaptive_catastrophic:
        return "missed"
    if not fixed_catastrophic and safe_mode:
        return "false_alarm"
    if adaptive_catastrophic:
        return "missed"
    return "safe"


def _build_case_table(results: pd.DataFrame, decisions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    fixed = results[results["algorithm"] != ADAPTIVE_NAME]
    adaptive = results[results["algorithm"] == ADAPTIVE_NAME]
    rows = []
    for _, decision in decisions.iterrows():
        keys = (
            (fixed["depth_semitones"] == decision["depth_semitones"])
            & (fixed["rate_hz"] == decision["rate_hz"])
            & (fixed["shift_semitones"] == decision["shift_semitones"])
        )
        fixed_case = fixed[keys]
        adaptive_case = adaptive[
            (adaptive["depth_semitones"] == decision["depth_semitones"])
            & (adaptive["rate_hz"] == decision["rate_hz"])
            & (adaptive["shift_semitones"] == decision["shift_semitones"])
        ].iloc[0]
        worst = fixed_case.sort_values("composite_stress", ascending=False).iloc[0]
        fixed_catastrophic_count = int(np.sum(fixed_case["composite_stress"].to_numpy(dtype=np.float64) >= threshold))
        fixed_catastrophic = fixed_catastrophic_count > 0
        adaptive_catastrophic = float(adaptive_case["composite_stress"]) >= threshold
        rows.append(
            {
                **decision.to_dict(),
                "catastrophic_threshold": threshold,
                "fixed_catastrophic_count": fixed_catastrophic_count,
                "fixed_catastrophic_any": fixed_catastrophic,
                "adaptive_catastrophic": adaptive_catastrophic,
                "adaptive_composite_stress": float(adaptive_case["composite_stress"]),
                "worst_fixed_algorithm_label": worst["algorithm_label"],
                "worst_fixed_composite_stress": float(worst["composite_stress"]),
                "preflight_outcome": _outcome(fixed_catastrophic, adaptive_catastrophic, bool(decision["safe_mode"])),
            }
        )
    return pd.DataFrame(rows).sort_values(["rate_hz", "depth_semitones", "shift_semitones"]).reset_index(drop=True)


def _build_summary(cases: pd.DataFrame) -> pd.DataFrame:
    fixed_danger = cases["fixed_catastrophic_any"].to_numpy(dtype=bool)
    adaptive_fail = cases["adaptive_catastrophic"].to_numpy(dtype=bool)
    avoided = int(np.sum(fixed_danger & ~adaptive_fail))
    missed = int(np.sum(fixed_danger & adaptive_fail))
    introduced = int(np.sum(~fixed_danger & adaptive_fail))
    safe_mode_count = int(np.sum(cases["safe_mode"].to_numpy(dtype=bool)))
    return pd.DataFrame(
        [
            {
                "cases": int(len(cases)),
                "fixed_danger_cases": int(np.sum(fixed_danger)),
                "adaptive_catastrophic_cases": int(np.sum(adaptive_fail)),
                "avoided_fixed_danger_cases": avoided,
                "missed_fixed_danger_cases": missed,
                "introduced_catastrophic_cases": introduced,
                "safe_mode_count": safe_mode_count,
                "avoidance_rate": avoided / max(int(np.sum(fixed_danger)), 1),
                "selected_phase_vocoder_count": int(np.sum(cases["selected_algorithm"] == "phase_vocoder")),
                "selected_psola_count": int(np.sum(cases["selected_algorithm"] == "psola")),
                "selected_rubberband_count": int(np.sum(cases["selected_algorithm"] == "rubber_band")),
                "selected_wsola_count": int(np.sum(cases["selected_algorithm"] == "wsola")),
            }
        ]
    )


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fixed_path = REPO_ROOT / "artifacts" / "11_results.csv"
    cases_path = REPO_ROOT / "artifacts" / "11_modulation_cases.csv"
    if not fixed_path.exists() or not cases_path.exists():
        raise FileNotFoundError("Experiment 13 expects Experiment 11 artifacts. Run Experiment 11 first.")

    fixed_results = pd.read_csv(fixed_path)
    source_cases = pd.read_csv(cases_path)[["depth_semitones", "rate_hz", "shift_semitones"]].drop_duplicates()
    adaptive_rows = []
    decision_rows = []
    for _, case in source_cases.iterrows():
        source = _vibrato_tone(220.0, float(case["depth_semitones"]), float(case["rate_hz"]), sr=SR, duration=DURATION)
        shifted, analysis = pitch_shift_preflight_adaptive_v1(source, sr=SR, n_steps=int(case["shift_semitones"]))
        adaptive_rows.append(_adaptive_metric_row(source, shifted, case))
        decision_rows.append(
            {
                "depth_semitones": float(case["depth_semitones"]),
                "rate_hz": float(case["rate_hz"]),
                "shift_semitones": int(case["shift_semitones"]),
                **asdict(analysis),
            }
        )

    results = add_composite_stress(pd.concat([fixed_results, pd.DataFrame(adaptive_rows)], ignore_index=True))
    threshold = float(results[results["algorithm"] != ADAPTIVE_NAME]["composite_stress"].quantile(0.85))
    decisions = pd.DataFrame(decision_rows)
    cases = _build_case_table(results, decisions, threshold=threshold)
    summary = _build_summary(cases)

    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(artifacts_dir / "13_results.csv", index=False)
    cases.to_csv(artifacts_dir / "13_preflight_adaptive_cases.csv", index=False)
    summary.to_csv(artifacts_dir / "13_preflight_adaptive_summary.csv", index=False)

    plot_frame = cases.rename(
        columns={
            "preflight_risk_score": "preflight_risk_score",
            "preflight_risk_level": "preflight_risk_level",
            "fixed_catastrophic_any": "catastrophic_any_fixed",
            "fixed_catastrophic_count": "catastrophic_fixed_count",
        }
    )
    plot_frame["worst_fixed_composite_stress"] = plot_frame["adaptive_composite_stress"]
    save_modulation_trap_plot(
        plot_frame,
        artifacts_dir / "13_preflight_adaptive_v1_map.png",
        shift=3,
        title="Exp13 Preflight Adaptive v1",
        subtitle="Can modulation-aware routing avoid the +3 semitone trap?",
        miss_note="caught = fixed danger avoided by adaptive v1; missed = adaptive v1 still catastrophic",
    )
    return results, cases, summary


def main() -> None:
    results, cases, summary = run_experiment()
    print(f"Wrote {len(results)} rows to artifacts/13_results.csv")
    print(f"Wrote {len(cases)} case rows to artifacts/13_preflight_adaptive_cases.csv")
    print("Wrote artifacts/13_preflight_adaptive_summary.csv")
    print("Wrote artifacts/13_preflight_adaptive_v1_map.png")
    print("Question: Can modulation-aware preflight routing avoid the subtle modulation trap?")
    print(summary.to_string(index=False))
    print(cases[cases["preflight_outcome"] == "missed"][["depth_semitones", "rate_hz", "shift_semitones", "adaptive_composite_stress", "selected_algorithm_label"]].to_string(index=False))


if __name__ == "__main__":
    main()
