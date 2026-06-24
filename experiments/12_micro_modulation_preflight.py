from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.preflight_risk import HIGH_RISK_THRESHOLD, modulation_aware_preflight_risk_score, preflight_risk_score, risk_level
from metrics.signal_features import compute_signal_features
from visualization.modulation_trap_plot import save_modulation_trap_plot


SR = 22050
DURATION = 2.0


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


def _outcome(catastrophic: bool, high_risk: bool) -> str:
    if catastrophic and high_risk:
        return "caught"
    if catastrophic and not high_risk:
        return "missed"
    if not catastrophic and high_risk:
        return "false_alarm"
    return "safe"


def _evaluate(predicted: pd.Series, actual: pd.Series) -> dict[str, float | int]:
    predicted_values = predicted.to_numpy(dtype=bool)
    actual_values = actual.to_numpy(dtype=bool)
    tp = int(np.sum(predicted_values & actual_values))
    fp = int(np.sum(predicted_values & ~actual_values))
    tn = int(np.sum(~predicted_values & ~actual_values))
    fn = int(np.sum(~predicted_values & actual_values))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _build_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    baseline = _evaluate(predictions["baseline_high_risk"], predictions["catastrophic_any_fixed"])
    modulation = _evaluate(predictions["modulation_aware_high_risk"], predictions["catastrophic_any_fixed"])
    return pd.DataFrame(
        [
            {
                "scorer": "baseline_preflight",
                "cases": int(len(predictions)),
                **baseline,
            },
            {
                "scorer": "modulation_aware_preflight",
                "cases": int(len(predictions)),
                **modulation,
            },
        ]
    )


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame]:
    cases_path = REPO_ROOT / "artifacts" / "11_modulation_cases.csv"
    if not cases_path.exists():
        raise FileNotFoundError("Experiment 12 expects artifacts/11_modulation_cases.csv. Run Experiment 11 first.")

    cases = pd.read_csv(cases_path)
    rows = []
    for _, case in cases.iterrows():
        depth = float(case["depth_semitones"])
        rate = float(case["rate_hz"])
        shift = int(case["shift_semitones"])
        source = _vibrato_tone(220.0, depth, rate, sr=SR, duration=DURATION)
        features = compute_signal_features(source, sr=SR).as_dict()
        baseline_score, baseline_reasons = preflight_risk_score(features, shift)
        aware_score, aware_reasons = modulation_aware_preflight_risk_score(features, shift)
        baseline_high = baseline_score >= HIGH_RISK_THRESHOLD
        aware_high = aware_score >= HIGH_RISK_THRESHOLD
        catastrophic = bool(case["catastrophic_any_fixed"])
        rows.append(
            {
                "depth_semitones": depth,
                "rate_hz": rate,
                "shift_semitones": shift,
                "catastrophic_any_fixed": catastrophic,
                "catastrophic_fixed_count": int(case["catastrophic_fixed_count"]),
                "worst_fixed_algorithm_label": case["worst_fixed_algorithm_label"],
                "worst_fixed_composite_stress": float(case["worst_fixed_composite_stress"]),
                "baseline_risk_score": baseline_score,
                "baseline_risk_level": risk_level(baseline_score),
                "baseline_high_risk": baseline_high,
                "baseline_reasons": baseline_reasons,
                "baseline_outcome": _outcome(catastrophic, baseline_high),
                "modulation_aware_risk_score": aware_score,
                "modulation_aware_risk_level": risk_level(aware_score),
                "modulation_aware_high_risk": aware_high,
                "modulation_aware_reasons": aware_reasons,
                "modulation_aware_outcome": _outcome(catastrophic, aware_high),
                **features,
            }
        )

    predictions = pd.DataFrame(rows).sort_values(["rate_hz", "depth_semitones", "shift_semitones"]).reset_index(drop=True)
    summary = _build_summary(predictions)

    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(artifacts_dir / "12_micro_modulation_predictions.csv", index=False)
    summary.to_csv(artifacts_dir / "12_micro_modulation_summary.csv", index=False)

    plot_frame = predictions.rename(
        columns={
            "modulation_aware_risk_score": "preflight_risk_score",
            "modulation_aware_risk_level": "preflight_risk_level",
            "modulation_aware_high_risk": "preflight_high_risk",
            "modulation_aware_outcome": "preflight_outcome",
        }
    )
    save_modulation_trap_plot(
        plot_frame,
        artifacts_dir / "12_micro_modulation_preflight_map.png",
        shift=3,
        title="Exp12 Micro-Modulation Preflight",
        subtitle="Modulation-aware risk at +3 semitones.",
        miss_note="missed = catastrophic fixed-algorithm stress not flagged by modulation-aware preflight",
    )
    return predictions, summary


def main() -> None:
    predictions, summary = run_experiment()
    print(f"Wrote {len(predictions)} prediction rows to artifacts/12_micro_modulation_predictions.csv")
    print("Wrote artifacts/12_micro_modulation_summary.csv")
    print("Wrote artifacts/12_micro_modulation_preflight_map.png")
    print("Question: Can a micro-modulation descriptor catch the subtle vibrato trap?")
    print(summary.to_string(index=False))
    improved = predictions[
        (predictions["baseline_outcome"] == "missed") & (predictions["modulation_aware_outcome"] == "caught")
    ]
    print(improved[["depth_semitones", "rate_hz", "shift_semitones", "modulation_aware_risk_score", "pitch_modulation_rms_cents", "pitch_modulation_peak_rate_hz"]].to_string(index=False))


if __name__ == "__main__":
    main()
