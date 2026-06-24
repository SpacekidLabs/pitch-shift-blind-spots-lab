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
from metrics.disagreement import add_composite_stress
from metrics.preflight_risk import HIGH_RISK_THRESHOLD, preflight_risk_score, risk_level
from metrics.signal_features import compute_signal_features
from visualization.modulation_trap_plot import save_modulation_trap_plot


SR = 22050
DURATION = 2.0
SHIFT_STEPS = [3, 7, 12, -12]
DEPTHS_SEMITONES = [0.0, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
RATES_HZ = [2.0, 4.0, 5.5, 8.0]


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


def _metric_row(
    source: np.ndarray,
    shifted: np.ndarray,
    depth: float,
    rate: float,
    shift: int,
    algorithm_name: str,
    algorithm_label: str,
    implementation: str,
) -> dict[str, float | int | str]:
    signal_name = f"vibrato_depth_{depth:g}_rate_{rate:g}"
    return {
        "signal_family": "subtle_modulation",
        "signal_name": signal_name,
        "signal_label": f"{depth:g} st / {rate:g} Hz",
        "depth_semitones": depth,
        "rate_hz": rate,
        "shift_semitones": shift,
        "algorithm": algorithm_name,
        "algorithm_label": algorithm_label,
        "implementation": implementation,
        "rms_error": rms_error(source, shifted),
        "spectral_distance": spectral_distance(source, shifted),
        "spectral_centroid_difference": spectral_centroid_difference(source, shifted, sr=SR),
    }


def _outcome(catastrophic: bool, high_risk: bool) -> str:
    if catastrophic and high_risk:
        return "caught"
    if catastrophic and not high_risk:
        return "missed"
    if not catastrophic and high_risk:
        return "false_alarm"
    return "safe"


def _build_case_table(results: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for (depth, rate, shift), frame in results.groupby(["depth_semitones", "rate_hz", "shift_semitones"], sort=False):
        source = _vibrato_tone(220.0, depth, rate, sr=SR, duration=DURATION)
        features = compute_signal_features(source, sr=SR).as_dict()
        risk_score, reasons = preflight_risk_score(features, int(shift))
        worst = frame.sort_values("composite_stress", ascending=False).iloc[0]
        safest = frame.sort_values("composite_stress", ascending=True).iloc[0]
        catastrophic_count = int(np.sum(frame["composite_stress"].to_numpy(dtype=np.float64) >= threshold))
        catastrophic = catastrophic_count > 0
        high_risk = risk_score >= HIGH_RISK_THRESHOLD
        rows.append(
            {
                "depth_semitones": depth,
                "rate_hz": rate,
                "shift_semitones": int(shift),
                "preflight_risk_score": risk_score,
                "preflight_risk_level": risk_level(risk_score),
                "preflight_high_risk": high_risk,
                "preflight_reasons": reasons,
                "catastrophic_threshold": threshold,
                "catastrophic_fixed_count": catastrophic_count,
                "catastrophic_any_fixed": catastrophic,
                "preflight_outcome": _outcome(catastrophic, high_risk),
                "worst_fixed_algorithm": worst["algorithm"],
                "worst_fixed_algorithm_label": worst["algorithm_label"],
                "worst_fixed_composite_stress": float(worst["composite_stress"]),
                "safest_fixed_algorithm": safest["algorithm"],
                "safest_fixed_algorithm_label": safest["algorithm_label"],
                "safest_fixed_composite_stress": float(safest["composite_stress"]),
                **features,
            }
        )
    return pd.DataFrame(rows).sort_values(["rate_hz", "depth_semitones", "shift_semitones"]).reset_index(drop=True)


def _build_summary(cases: pd.DataFrame) -> pd.DataFrame:
    actual = cases["catastrophic_any_fixed"].to_numpy(dtype=bool)
    predicted = cases["preflight_high_risk"].to_numpy(dtype=bool)
    tp = int(np.sum(predicted & actual))
    fp = int(np.sum(predicted & ~actual))
    tn = int(np.sum(~predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    missed = cases[cases["preflight_outcome"] == "missed"]
    return pd.DataFrame(
        [
            {
                "cases": int(len(cases)),
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn,
                "precision": precision,
                "recall": recall,
                "missed_case_count": int(len(missed)),
                "lowest_missed_depth_semitones": float(missed["depth_semitones"].min()) if not missed.empty else np.nan,
                "highest_missed_depth_semitones": float(missed["depth_semitones"].max()) if not missed.empty else np.nan,
                "missed_rates_hz": "|".join(f"{rate:g}" for rate in sorted(missed["rate_hz"].unique())),
            }
        ]
    )


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for rate in RATES_HZ:
        for depth in DEPTHS_SEMITONES:
            source = _vibrato_tone(220.0, depth, rate, sr=SR, duration=DURATION)
            for shift in SHIFT_STEPS:
                for algorithm in ALGORITHMS:
                    shifted = algorithm.pitch_shift(source, sr=SR, n_steps=shift)
                    rows.append(
                        _metric_row(
                            source=source,
                            shifted=shifted,
                            depth=depth,
                            rate=rate,
                            shift=shift,
                            algorithm_name=algorithm.name,
                            algorithm_label=algorithm.display_name,
                            implementation=algorithm.implementation,
                        )
                    )

    results = add_composite_stress(pd.DataFrame(rows))
    catastrophic_threshold = float(results["composite_stress"].quantile(0.85))
    cases = _build_case_table(results, threshold=catastrophic_threshold)
    summary = _build_summary(cases)

    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(artifacts_dir / "11_results.csv", index=False)
    cases.to_csv(artifacts_dir / "11_modulation_cases.csv", index=False)
    summary.to_csv(artifacts_dir / "11_modulation_trap_summary.csv", index=False)
    save_modulation_trap_plot(cases, artifacts_dir / "11_subtle_modulation_trap_map.png", shift=3)
    return results, cases, summary


def main() -> None:
    results, cases, summary = run_experiment()
    print(f"Wrote {len(results)} rows to artifacts/11_results.csv")
    print(f"Wrote {len(cases)} case rows to artifacts/11_modulation_cases.csv")
    print("Wrote artifacts/11_modulation_trap_summary.csv")
    print("Wrote artifacts/11_subtle_modulation_trap_map.png")
    print("Question: Where does subtle modulation masquerade as stable periodicity?")
    print(summary.to_string(index=False))
    missed = cases[cases["preflight_outcome"] == "missed"]
    print(missed[["depth_semitones", "rate_hz", "shift_semitones", "preflight_risk_score", "worst_fixed_algorithm_label", "worst_fixed_composite_stress"]].to_string(index=False))


if __name__ == "__main__":
    main()
