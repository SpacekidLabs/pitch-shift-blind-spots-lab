from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.signal_features import compute_signal_features
from signals.library import SIGNAL_SPECS, build_signal_atlas
from visualization.preflight_risk_plot import save_preflight_risk_plot


SR = 22050
DURATION = 2.0
SHIFT_STEPS = [3, 7, 12, -12]
FIXED_ALGORITHMS = {"phase_vocoder", "psola", "wsola", "rubber_band"}


def _risk_level(score: float) -> str:
    if score >= 0.32:
        return "high"
    if score >= 0.18:
        return "medium"
    return "low"


def _finite(value: float, fallback: float = 0.0) -> float:
    return float(value) if np.isfinite(value) else fallback


def _preflight_risk_score(features: dict[str, float], shift: int) -> tuple[float, str]:
    score = 0.0
    reasons = []

    pitch_valid = _finite(features["pitch_valid_fraction"])
    if pitch_valid < 0.20:
        score += 0.30
        reasons.append("untracked_pitch")
    elif pitch_valid < 0.65:
        score += 0.15
        reasons.append("partial_pitch_tracking")

    pitch_iqr = _finite(features["pitch_iqr_cents"])
    if pitch_iqr > 300.0:
        score += 0.30
        reasons.append("unstable_pitch")
    elif pitch_iqr > 150.0:
        score += 0.25
        reasons.append("strong_pitch_motion")
    elif pitch_iqr > 80.0:
        score += 0.15
        reasons.append("pitch_motion")

    flatness = _finite(features["spectral_flatness"])
    if pitch_valid > 0.95 and flatness < 0.02:
        score += 0.25
        reasons.append("periodic_representation_bias")
    if flatness > 0.45:
        score += 0.30
        reasons.append("noise_like_spectrum")
    elif flatness > 0.12:
        score += 0.15
        reasons.append("diffuse_spectrum")

    transient = _finite(features["transient_score"])
    if transient > 120.0:
        score += 0.22
        reasons.append("sparse_transients")
    elif transient > 18.0:
        score += 0.12
        reasons.append("transient_energy")

    bandwidth = _finite(features["spectral_bandwidth_hz"])
    centroid = _finite(features["spectral_centroid_hz"])
    if bandwidth > 2500.0:
        score += 0.20
        reasons.append("broadband_structure")
    elif bandwidth > 1400.0:
        score += 0.12
        reasons.append("wide_harmonic_structure")
    elif bandwidth > 800.0 and flatness < 0.02:
        score += 0.20
        reasons.append("sharp_harmonic_structure")
    if centroid > 2600.0:
        score += 0.08
        reasons.append("high_centroid")

    crest = _finite(features["crest_factor"])
    if crest > 12.0:
        score += 0.12
        reasons.append("high_crest_factor")
    elif crest > 5.0:
        score += 0.06
        reasons.append("peaky_waveform")

    zero_crossing = _finite(features["zero_crossing_rate"])
    if zero_crossing > 0.22:
        score += 0.08
        reasons.append("dense_zero_crossings")

    if abs(shift) >= 12:
        score += 0.20
        reasons.append("large_shift")
    elif abs(shift) >= 7:
        score += 0.08
        reasons.append("moderate_shift")

    return min(score, 1.0), "|".join(reasons) if reasons else "stable_source"


def _load_fixed_results() -> tuple[pd.DataFrame, float]:
    path = REPO_ROOT / "artifacts" / "09_results.csv"
    if not path.exists():
        raise FileNotFoundError("Experiment 10 expects artifacts/09_results.csv. Run Experiment 09 first.")
    results = pd.read_csv(path)
    fixed = results[results["algorithm"].isin(FIXED_ALGORITHMS)].copy()
    threshold = float(fixed["composite_stress"].quantile(0.85))
    return fixed, threshold


def _build_predictions(features: pd.DataFrame, fixed_results: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for _, feature_row in features.iterrows():
        feature_dict = feature_row.to_dict()
        signal_rows = fixed_results[fixed_results["signal_name"] == feature_row["signal_name"]]
        for shift in SHIFT_STEPS:
            risk_score, reasons = _preflight_risk_score(feature_dict, shift)
            fixed_shift = signal_rows[signal_rows["shift_semitones"] == shift]
            worst = fixed_shift.sort_values("composite_stress", ascending=False).iloc[0]
            safest = fixed_shift.sort_values("composite_stress", ascending=True).iloc[0]
            catastrophic_count = int(np.sum(fixed_shift["composite_stress"].to_numpy(dtype=np.float64) >= threshold))
            rows.append(
                {
                    "signal_family": feature_row["signal_family"],
                    "signal_name": feature_row["signal_name"],
                    "signal_label": feature_row["signal_label"],
                    "shift_semitones": shift,
                    "preflight_risk_score": risk_score,
                    "preflight_risk_level": _risk_level(risk_score),
                    "preflight_high_risk": risk_score >= 0.32,
                    "preflight_reasons": reasons,
                    "catastrophic_threshold": threshold,
                    "catastrophic_fixed_count": catastrophic_count,
                    "catastrophic_any_fixed": catastrophic_count > 0,
                    "worst_fixed_algorithm": worst["algorithm"],
                    "worst_fixed_algorithm_label": worst["algorithm_label"],
                    "worst_fixed_composite_stress": float(worst["composite_stress"]),
                    "safest_fixed_algorithm": safest["algorithm"],
                    "safest_fixed_algorithm_label": safest["algorithm_label"],
                    "safest_fixed_composite_stress": float(safest["composite_stress"]),
                }
            )
    return pd.DataFrame(rows)


def _build_evaluation(predictions: pd.DataFrame) -> pd.DataFrame:
    actual = predictions["catastrophic_any_fixed"].to_numpy(dtype=bool)
    predicted = predictions["preflight_high_risk"].to_numpy(dtype=bool)
    tp = int(np.sum(predicted & actual))
    fp = int(np.sum(predicted & ~actual))
    tn = int(np.sum(~predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return pd.DataFrame(
        [
            {
                "cases": int(len(predictions)),
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": f1,
            }
        ]
    )


def _build_signal_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for signal_name, frame in predictions.groupby("signal_name", sort=False):
        actual = frame["catastrophic_any_fixed"].to_numpy(dtype=bool)
        predicted = frame["preflight_high_risk"].to_numpy(dtype=bool)
        actual_count = int(np.sum(actual))
        predicted_count = int(np.sum(predicted))
        caught = int(np.sum(actual & predicted))
        rows.append(
            {
                "signal_family": frame["signal_family"].iloc[0],
                "signal_name": signal_name,
                "signal_label": frame["signal_label"].iloc[0],
                "actual_catastrophic_cases": actual_count,
                "predicted_high_risk_cases": predicted_count,
                "caught_catastrophic_cases": caught,
                "signal_recall": caught / max(actual_count, 1),
                "mean_preflight_risk_score": float(frame["preflight_risk_score"].mean()),
                "max_preflight_risk_score": float(frame["preflight_risk_score"].max()),
                "dominant_reasons": "|".join(sorted(set("|".join(frame["preflight_reasons"]).split("|")))),
            }
        )
    return pd.DataFrame(rows).sort_values("max_preflight_risk_score", ascending=False).reset_index(drop=True)


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fixed_results, threshold = _load_fixed_results()
    atlas = build_signal_atlas(sr=SR, duration=DURATION, seed=7)
    feature_rows = []
    for spec in SIGNAL_SPECS:
        feature_values = compute_signal_features(atlas[spec.name], sr=SR).as_dict()
        feature_rows.append(
            {
                "signal_family": spec.family,
                "signal_name": spec.name,
                "signal_label": f"{spec.family} / {spec.name}",
                **feature_values,
            }
        )
    features = pd.DataFrame(feature_rows)
    predictions = _build_predictions(features, fixed_results=fixed_results, threshold=threshold)
    evaluation = _build_evaluation(predictions)
    signal_summary = _build_signal_summary(predictions)

    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(artifacts_dir / "10_preflight_features.csv", index=False)
    predictions.to_csv(artifacts_dir / "10_preflight_risk_predictions.csv", index=False)
    evaluation.to_csv(artifacts_dir / "10_preflight_evaluation.csv", index=False)
    signal_summary.to_csv(artifacts_dir / "10_preflight_signal_summary.csv", index=False)
    save_preflight_risk_plot(predictions, signal_summary, artifacts_dir / "10_preflight_risk_map.png")
    return features, predictions, evaluation, signal_summary


def main() -> None:
    features, predictions, evaluation, signal_summary = run_experiment()
    print(f"Wrote {len(features)} feature rows to artifacts/10_preflight_features.csv")
    print(f"Wrote {len(predictions)} prediction rows to artifacts/10_preflight_risk_predictions.csv")
    print("Wrote artifacts/10_preflight_evaluation.csv")
    print("Wrote artifacts/10_preflight_signal_summary.csv")
    print("Wrote artifacts/10_preflight_risk_map.png")
    print("Question: Can source-only features predict catastrophic pitch-shift risk before processing?")
    print(evaluation.to_string(index=False))
    print(signal_summary.to_string(index=False))


if __name__ == "__main__":
    main()
