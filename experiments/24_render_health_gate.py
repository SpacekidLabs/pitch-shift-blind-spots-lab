from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.render_health import render_health_metrics
from signals.audio_io import read_wav
from visualization.render_health_plot import save_render_health_plot


EXP22_DIR = REPO_ROOT / "artifacts" / "22_drum_listening_test"
EXP23_DIR = REPO_ROOT / "artifacts" / "23_listening_score_analysis"
OUTPUT_DIR = REPO_ROOT / "artifacts" / "24_render_health_gate"


def _load_decoded_scores() -> pd.DataFrame:
    path = EXP23_DIR / "23_decoded_listener_scores.csv"
    if not path.exists():
        raise FileNotFoundError("Experiment 24 expects Experiment 23 decoded scores. Run Experiment 23 first.")
    return pd.read_csv(path)


def _build_results(decoded: pd.DataFrame) -> pd.DataFrame:
    reference_path = EXP22_DIR / "audio" / "REFERENCE_original_excerpt.wav"
    if not reference_path.exists():
        raise FileNotFoundError("Experiment 24 needs the local Experiment 22 audio folder.")
    _, reference = read_wav(reference_path)

    rows = []
    for row in decoded.to_dict("records"):
        wav_path = EXP22_DIR / str(row["wav_file"])
        if not wav_path.exists():
            raise FileNotFoundError(f"Missing rendered audio file: {wav_path}")
        _, rendered = read_wav(wav_path)
        health = render_health_metrics(reference, rendered)
        listener_silence = bool(row.get("silence_flag", False))
        rows.append(
            {
                "blind_id": row["blind_id"],
                "wav_file": row["wav_file"],
                "strategy": row["strategy"],
                "strategy_label": row["strategy_label"],
                "shift_semitones": row["shift_semitones"],
                "overall_rating": row["overall_rating"],
                "artifact_rating": row["artifact_rating"],
                "transient_rating": row["transient_rating"],
                "groove_rating": row["groove_rating"],
                "listener_silence_flag": listener_silence,
                "listener_unusable_flag": float(row["overall_rating"]) <= 1.0,
                "notes": row.get("notes", ""),
                **health,
            }
        )
    results = pd.DataFrame(rows)
    results["true_positive"] = results["silence_like_health_flag"] & results["listener_silence_flag"]
    results["false_positive"] = results["silence_like_health_flag"] & ~results["listener_silence_flag"]
    results["false_negative"] = ~results["silence_like_health_flag"] & results["listener_silence_flag"]
    results["true_negative"] = ~results["silence_like_health_flag"] & ~results["listener_silence_flag"]
    return results


def _build_detection_summary(results: pd.DataFrame) -> pd.DataFrame:
    true_positive = int(results["true_positive"].sum())
    false_positive = int(results["false_positive"].sum())
    false_negative = int(results["false_negative"].sum())
    true_negative = int(results["true_negative"].sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return pd.DataFrame(
        [
            {
                "case_count": int(len(results)),
                "listener_silence_count": int(results["listener_silence_flag"].sum()),
                "health_flag_count": int(results["silence_like_health_flag"].sum()),
                "true_positive_count": true_positive,
                "false_positive_count": false_positive,
                "false_negative_count": false_negative,
                "true_negative_count": true_negative,
                "silence_precision": precision,
                "silence_recall": recall,
            }
        ]
    )


def _build_strategy_summary(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["strategy", "strategy_label"], as_index=False)
        .agg(
            case_count=("blind_id", "count"),
            listener_silence_count=("listener_silence_flag", "sum"),
            health_flag_count=("silence_like_health_flag", "sum"),
            mean_relative_rms_db=("relative_rms_db", "mean"),
            mean_active_fraction=("active_fraction", "mean"),
            mean_overall_rating=("overall_rating", "mean"),
        )
        .sort_values(["health_flag_count", "mean_overall_rating"], ascending=[False, True])
        .reset_index(drop=True)
    )


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    decoded = _load_decoded_scores()
    results = _build_results(decoded)
    detection_summary = _build_detection_summary(results)
    strategy_summary = _build_strategy_summary(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "24_render_health_results.csv", index=False)
    detection_summary.to_csv(OUTPUT_DIR / "24_render_health_detection_summary.csv", index=False)
    strategy_summary.to_csv(OUTPUT_DIR / "24_render_health_strategy_summary.csv", index=False)
    save_render_health_plot(results, detection_summary, OUTPUT_DIR / "24_render_health_plot.png")
    return results, detection_summary, strategy_summary


def main() -> None:
    results, detection_summary, strategy_summary = run_experiment()
    print(f"Wrote {len(results)} render-health rows to artifacts/24_render_health_gate")
    print("Question: Can render health detect perceived silence failures?")
    print(detection_summary.to_string(index=False))
    print(strategy_summary.to_string(index=False))


if __name__ == "__main__":
    main()
