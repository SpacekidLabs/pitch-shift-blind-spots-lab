from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.preflight_risk import HIGH_RISK_THRESHOLD, preflight_risk_score, risk_level
from metrics.signal_features import compute_signal_features
from signals.library import SIGNAL_SPECS, build_signal_atlas
from visualization.preflight_risk_plot import save_preflight_risk_plot


SR = 22050
DURATION = 2.0
SHIFT_STEPS = [3, 7, 12, -12]
FIXED_ALGORITHMS = {"phase_vocoder", "psola", "wsola", "rubber_band"}


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
            risk_score, reasons = preflight_risk_score(feature_dict, shift)
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
                    "preflight_risk_level": risk_level(risk_score),
                    "preflight_high_risk": risk_score >= HIGH_RISK_THRESHOLD,
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
