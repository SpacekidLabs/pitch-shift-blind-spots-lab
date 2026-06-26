from __future__ import annotations

from io import StringIO
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visualization.adaptive_v4_score_plot import save_adaptive_v4_score_plot


EXP26_DIR = REPO_ROOT / "artifacts" / "26_adaptive_v4_blind_listening_test"
OUTPUT_DIR = REPO_ROOT / "artifacts" / "27_adaptive_v4_score_analysis"
RATING_COLUMNS = [
    "artifact_rating_1_bad_5_clean",
    "transient_rating_1_smeared_5_crisp",
    "groove_rating_1_bad_5_good",
    "overall_rating_1_bad_5_good",
]

LISTENER_SCORES_CSV = """blind_id,wav_file,shift_semitones,artifact_rating_1_bad_5_clean,transient_rating_1_smeared_5_crisp,groove_rating_1_bad_5_good,overall_rating_1_bad_5_good,notes
V4LT_001,audio/V4LT_001.wav,-12,3,2,4,4,
V4LT_002,audio/V4LT_002.wav,-12,3,2,3,2,
V4LT_003,audio/V4LT_003.wav,3,3,3,3,3,
V4LT_004,audio/V4LT_004.wav,3,4,3,3,2,
V4LT_005,audio/V4LT_005.wav,7,3,2,3,2,
V4LT_006,audio/V4LT_006.wav,-12,3,2,3,3,
V4LT_007,audio/V4LT_007.wav,7,3,3,3,3,
V4LT_008,audio/V4LT_008.wav,7,2,3,3,4,
V4LT_009,audio/V4LT_009.wav,7,3,3,3,4,
V4LT_010,audio/V4LT_010.wav,3,2,3,3,3,
V4LT_011,audio/V4LT_011.wav,7,2,2,3,2,
V4LT_012,audio/V4LT_012.wav,3,3,3,3,3,
V4LT_013,audio/V4LT_013.wav,-12,3,2,3,2,
V4LT_014,audio/V4LT_014.wav,3,3,3,3,3,
V4LT_015,audio/V4LT_015.wav,-12,4,3,3,4,
"""


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
    normalized["notes"] = normalized["notes"].fillna("")
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


def _summary_by(decoded: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    return (
        decoded.groupby(group_columns, dropna=False, as_index=False)
        .agg(
            case_count=("blind_id", "count"),
            mean_artifact_rating=("artifact_rating", "mean"),
            mean_transient_rating=("transient_rating", "mean"),
            mean_groove_rating=("groove_rating", "mean"),
            mean_overall_rating=("overall_rating", "mean"),
            overall_std=("overall_rating", "std"),
            overall_min=("overall_rating", "min"),
            overall_max=("overall_rating", "max"),
            overall_range=("overall_rating", lambda values: float(values.max() - values.min())),
            fallback_count=("fallback_used", "sum"),
            mean_composite_stress=("composite_stress", "mean"),
        )
        .sort_values(["mean_overall_rating", "mean_groove_rating"], ascending=False)
        .reset_index(drop=True)
    )


def _correlation_summary(decoded: pd.DataFrame) -> pd.DataFrame:
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


def _equivalent_render_summary(decoded: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "final_algorithm_label",
        "shift_semitones",
        "rms_error",
        "spectral_distance",
        "spectral_centroid_difference",
    ]
    rows = []
    for _, frame in decoded.groupby(group_columns, dropna=False):
        if len(frame) < 2:
            continue
        rows.append(
            {
                "final_algorithm_label": frame["final_algorithm_label"].iloc[0],
                "shift_semitones": int(frame["shift_semitones"].iloc[0]),
                "case_count": len(frame),
                "blind_ids": "|".join(frame["blind_id"].astype(str)),
                "render_labels": "|".join(frame["render_label"].astype(str)),
                "overall_scores": "|".join(frame["overall_rating"].astype(str)),
                "overall_mean": float(frame["overall_rating"].mean()),
                "overall_std": float(frame["overall_rating"].std(ddof=0)),
                "overall_range": float(frame["overall_rating"].max() - frame["overall_rating"].min()),
                "artifact_scores": "|".join(frame["artifact_rating"].astype(str)),
                "artifact_range": float(frame["artifact_rating"].max() - frame["artifact_rating"].min()),
            }
        )
    return pd.DataFrame(rows).sort_values(["overall_range", "shift_semitones"], ascending=[False, True]).reset_index(drop=True)


def _write_interpretation(
    decoded: pd.DataFrame,
    policy_summary: pd.DataFrame,
    shift_summary: pd.DataFrame,
    equivalent_summary: pd.DataFrame,
    correlation_summary: pd.DataFrame,
) -> None:
    top_policy = policy_summary.iloc[0]
    all_final_algorithms = ", ".join(sorted(decoded["final_algorithm_label"].dropna().unique()))
    widest_duplicate = equivalent_summary.iloc[0]
    stress_corr = correlation_summary.loc[
        correlation_summary["metric"] == "composite_stress", "pearson_r_with_overall"
    ].iloc[0]

    lines = [
        "# Experiment 27 Adaptive v4 Score Analysis",
        "",
        "Question:",
        "",
        "```text",
        "Did the unblinded Adaptive v4 Normal preference survive blind scoring?",
        "```",
        "",
        "Finding:",
        "",
        f"- highest mean overall policy: `{top_policy['render_label']}` at `{top_policy['mean_overall_rating']:.2f}`",
        f"- final rendered algorithm set: `{all_final_algorithms}`",
        f"- composite-stress correlation with overall rating: `{stress_corr:.2f}`",
        f"- widest equivalent-render overall-score range: `{widest_duplicate['overall_range']:.1f}` at `{int(widest_duplicate['shift_semitones']):+d}` semitones",
        "",
        "Interpretation:",
        "",
        "All Experiment 26 final renders resolved to Phase Vocoder, either directly or through the post-render health fallback.",
        "That means the decoded policy means should not be treated as proof that one waveform-producing strategy sounded better.",
        "Instead, this test exposes listener/context variability among equivalent or near-equivalent renders.",
        "",
        "The practical result still supports Adaptive v4 as a safe product behavior:",
        "",
        "- Adaptive v4 Normal avoided collapse without needing fallback on this drum loop.",
        "- Forced WSOLA and Rubber Band paths were caught by the health gate and rescued.",
        "- No blind file was described as silence, unlike Experiment 22.",
        "",
        "The deeper lab result is that blind listening needs duplicate controls.",
        "Without them, a small score difference can look like an algorithm preference even when the final rendered audio path is the same.",
        "",
        "Outputs:",
        "",
        "- `27_listener_scores.csv`",
        "- `27_decoded_listener_scores.csv`",
        "- `27_policy_summary.csv`",
        "- `27_shift_summary.csv`",
        "- `27_fallback_summary.csv`",
        "- `27_equivalent_render_summary.csv`",
        "- `27_metric_correlation_summary.csv`",
        "- `27_adaptive_v4_score_plot.png`",
        "",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = _normalize_scores(pd.read_csv(StringIO(LISTENER_SCORES_CSV)))
    key_path = EXP26_DIR / "private_answer_key.csv"
    if not key_path.exists():
        raise FileNotFoundError(
            "Experiment 27 needs the local private answer key from Experiment 26. "
            "Run Experiment 26 locally before decoding blind scores."
        )
    answer_key = _read_required_csv(key_path)
    metrics = _read_required_csv(EXP26_DIR / "objective_metrics.csv")

    decoded = scores.merge(answer_key, on=["blind_id", "wav_file", "shift_semitones"], how="left")
    decoded = decoded.merge(metrics, on=["blind_id", "wav_file", "shift_semitones"], how="left")
    decoded = decoded.sort_values(["shift_semitones", "blind_id"]).reset_index(drop=True)

    policy_summary = _summary_by(decoded, ["render_policy", "render_label"])
    shift_summary = _summary_by(decoded, ["shift_semitones"]).sort_values("shift_semitones").reset_index(drop=True)
    fallback_summary = _summary_by(decoded, ["fallback_used", "final_guard"])
    equivalent_summary = _equivalent_render_summary(decoded)
    correlation_summary = _correlation_summary(decoded)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUTPUT_DIR / "27_listener_scores.csv", index=False)
    decoded.to_csv(OUTPUT_DIR / "27_decoded_listener_scores.csv", index=False)
    policy_summary.to_csv(OUTPUT_DIR / "27_policy_summary.csv", index=False)
    shift_summary.to_csv(OUTPUT_DIR / "27_shift_summary.csv", index=False)
    fallback_summary.to_csv(OUTPUT_DIR / "27_fallback_summary.csv", index=False)
    equivalent_summary.to_csv(OUTPUT_DIR / "27_equivalent_render_summary.csv", index=False)
    correlation_summary.to_csv(OUTPUT_DIR / "27_metric_correlation_summary.csv", index=False)
    save_adaptive_v4_score_plot(
        policy_summary,
        shift_summary,
        equivalent_summary,
        OUTPUT_DIR / "27_adaptive_v4_score_plot.png",
    )
    _write_interpretation(decoded, policy_summary, shift_summary, equivalent_summary, correlation_summary)
    return decoded, policy_summary, shift_summary, equivalent_summary, correlation_summary


def main() -> None:
    decoded, policy_summary, shift_summary, equivalent_summary, correlation_summary = run_experiment()
    print(f"Wrote {len(decoded)} decoded Experiment 26 listener rows to {OUTPUT_DIR}")
    print("Question: Did Adaptive v4 Normal survive blind scoring?")
    print(policy_summary.to_string(index=False))
    print(shift_summary.to_string(index=False))
    print(equivalent_summary.to_string(index=False))
    print(correlation_summary.to_string(index=False))


if __name__ == "__main__":
    main()
