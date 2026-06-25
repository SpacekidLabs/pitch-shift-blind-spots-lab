from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visualization.listening_score_plot import save_listening_score_plot


EXP22_DIR = REPO_ROOT / "artifacts" / "22_drum_listening_test"
OUTPUT_DIR = REPO_ROOT / "artifacts" / "23_listening_score_analysis"
RATING_COLUMNS = [
    "artifact_rating_1_bad_5_clean",
    "transient_rating_1_smeared_5_crisp",
    "groove_rating_1_bad_5_good",
    "overall_rating_1_bad_5_good",
]


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def _normalize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    normalized = scores.copy()
    for column in RATING_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized["artifact_rating"] = normalized["artifact_rating_1_bad_5_clean"]
    normalized["transient_rating"] = normalized["transient_rating_1_smeared_5_crisp"]
    normalized["groove_rating"] = normalized["groove_rating_1_bad_5_good"]
    normalized["overall_rating"] = normalized["overall_rating_1_bad_5_good"]
    normalized["silence_flag"] = normalized["notes"].fillna("").str.contains("silence|cant hear", case=False, regex=True)
    return normalized


def _pearson(x: pd.Series, y: pd.Series) -> float:
    values = pd.concat([x, y], axis=1).dropna()
    if len(values) < 2:
        return float("nan")
    a = values.iloc[:, 0].to_numpy(dtype=np.float64)
    b = values.iloc[:, 1].to_numpy(dtype=np.float64)
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _build_strategy_summary(decoded: pd.DataFrame) -> pd.DataFrame:
    return (
        decoded.groupby(["strategy", "strategy_label"], as_index=False)
        .agg(
            case_count=("blind_id", "count"),
            mean_artifact_rating=("artifact_rating", "mean"),
            mean_transient_rating=("transient_rating", "mean"),
            mean_groove_rating=("groove_rating", "mean"),
            mean_overall_rating=("overall_rating", "mean"),
            silence_flags=("silence_flag", "sum"),
            mean_composite_stress=("composite_stress", "mean"),
        )
        .sort_values(["mean_overall_rating", "mean_groove_rating"], ascending=False)
        .reset_index(drop=True)
    )


def _build_shift_summary(decoded: pd.DataFrame) -> pd.DataFrame:
    return (
        decoded.groupby("shift_semitones", as_index=False)
        .agg(
            case_count=("blind_id", "count"),
            mean_artifact_rating=("artifact_rating", "mean"),
            mean_transient_rating=("transient_rating", "mean"),
            mean_groove_rating=("groove_rating", "mean"),
            mean_overall_rating=("overall_rating", "mean"),
            silence_flags=("silence_flag", "sum"),
            mean_composite_stress=("composite_stress", "mean"),
        )
        .sort_values("shift_semitones")
        .reset_index(drop=True)
    )


def _build_correlation_summary(decoded: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["rms_error", "spectral_distance", "spectral_centroid_difference", "composite_stress"]:
        rows.append(
            {
                "metric": metric,
                "pearson_r_with_overall": _pearson(decoded[metric], decoded["overall_rating"]),
                "pearson_r_with_artifact": _pearson(decoded[metric], decoded["artifact_rating"]),
                "pearson_r_with_transient": _pearson(decoded[metric], decoded["transient_rating"]),
                "pearson_r_with_groove": _pearson(decoded[metric], decoded["groove_rating"]),
            }
        )
    return pd.DataFrame(rows)


def _build_duplicate_summary(decoded: pd.DataFrame) -> pd.DataFrame:
    duplicate_groups = decoded.groupby(
        ["shift_semitones", "rms_error", "spectral_distance", "spectral_centroid_difference"],
        dropna=False,
    )
    rows = []
    for _, frame in duplicate_groups:
        if len(frame) < 2:
            continue
        rows.append(
            {
                "blind_ids": "|".join(frame["blind_id"].astype(str)),
                "strategy_labels": "|".join(frame["strategy_label"].astype(str)),
                "shift_semitones": int(frame["shift_semitones"].iloc[0]),
                "overall_scores": "|".join(frame["overall_rating"].astype(str)),
                "overall_range": float(frame["overall_rating"].max() - frame["overall_rating"].min()),
                "artifact_scores": "|".join(frame["artifact_rating"].astype(str)),
                "artifact_range": float(frame["artifact_rating"].max() - frame["artifact_rating"].min()),
            }
        )
    return pd.DataFrame(rows)


def _build_note_flags(decoded: pd.DataFrame) -> pd.DataFrame:
    flagged = decoded[decoded["notes"].fillna("").str.strip() != ""].copy()
    columns = ["blind_id", "strategy_label", "shift_semitones", "overall_rating", "silence_flag", "notes"]
    return flagged[columns].sort_values(["silence_flag", "overall_rating"], ascending=[False, True]).reset_index(drop=True)


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = _normalize_scores(_read_required_csv(EXP22_DIR / "listener_scores.csv"))
    metrics = _read_required_csv(EXP22_DIR / "objective_metrics.csv")
    key_path = EXP22_DIR / "private_answer_key.csv"
    if not key_path.exists():
        raise FileNotFoundError(
            "Experiment 23 needs the local private answer key generated by Experiment 22. "
            "Run Experiment 22 locally before decoding scores."
        )
    answer_key = pd.read_csv(key_path)

    decoded = scores.merge(answer_key, on=["blind_id", "wav_file", "shift_semitones"], how="left")
    decoded = decoded.merge(metrics, on=["blind_id", "wav_file", "shift_semitones"], how="left")
    decoded = decoded.sort_values(["strategy_label", "shift_semitones", "blind_id"]).reset_index(drop=True)

    strategy_summary = _build_strategy_summary(decoded)
    shift_summary = _build_shift_summary(decoded)
    correlation_summary = _build_correlation_summary(decoded)
    duplicate_summary = _build_duplicate_summary(decoded)
    note_flags = _build_note_flags(decoded)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    decoded.to_csv(OUTPUT_DIR / "23_decoded_listener_scores.csv", index=False)
    strategy_summary.to_csv(OUTPUT_DIR / "23_strategy_summary.csv", index=False)
    shift_summary.to_csv(OUTPUT_DIR / "23_shift_summary.csv", index=False)
    correlation_summary.to_csv(OUTPUT_DIR / "23_metric_correlation_summary.csv", index=False)
    duplicate_summary.to_csv(OUTPUT_DIR / "23_duplicate_consistency.csv", index=False)
    note_flags.to_csv(OUTPUT_DIR / "23_note_flags.csv", index=False)
    save_listening_score_plot(strategy_summary, correlation_summary, OUTPUT_DIR / "23_listening_score_plot.png")
    return decoded, strategy_summary, shift_summary, correlation_summary, duplicate_summary


def main() -> None:
    decoded, strategy_summary, shift_summary, correlation_summary, duplicate_summary = run_experiment()
    print(f"Wrote {len(decoded)} decoded listener rows to artifacts/23_listening_score_analysis")
    print("Question: Do objective stress metrics agree with human drum-loop preference?")
    print(strategy_summary.to_string(index=False))
    print(shift_summary.to_string(index=False))
    print(correlation_summary.to_string(index=False))
    print(duplicate_summary.to_string(index=False))


if __name__ == "__main__":
    main()
