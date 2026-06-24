from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.registry import ALGORITHMS
from metrics.pitch_tracking import framewise_autocorrelation_f0
from signals.ambiguity import AmbiguityCase, build_octave_ambiguity_cases
from visualization.octave_preference_plot import save_octave_preference_map


SR = 22050
DURATION = 2.0
SHIFT_SEMITONES = -12


def _trajectory_rows(case: AmbiguityCase, algorithm_name: str, algorithm_label: str, implementation: str, audio: np.ndarray) -> list[dict[str, float | int | str]]:
    factor = 2.0 ** (SHIFT_SEMITONES / 12.0)
    rows = []
    for frame_index, frame in enumerate(framewise_autocorrelation_f0(audio, sr=SR, fmin=55.0, fmax=1200.0)):
        observed = float(frame["f0_hz"]) if np.isfinite(frame["f0_hz"]) else np.nan
        inferred = observed / factor if factor > 0 and np.isfinite(observed) else np.nan
        rows.append(
            {
                "case_id": case.case_id,
                "case_family": case.family,
                "case_label": case.label,
                "parameter_summary": case.parameter_summary,
                "algorithm": algorithm_name,
                "algorithm_label": algorithm_label,
                "implementation": implementation,
                "shift_semitones": SHIFT_SEMITONES,
                "frame_index": frame_index,
                "time_seconds": float(frame["time_seconds"]),
                "observed_output_f0_hz": observed,
                "inferred_input_f0_hz": inferred,
                "pitch_confidence": float(frame["confidence"]),
            }
        )
    return rows


def _build_trajectories(cases: list[AmbiguityCase]) -> pd.DataFrame:
    rows = []
    for case in cases:
        for algorithm in ALGORITHMS:
            shifted = algorithm.pitch_shift(case.audio, sr=SR, n_steps=SHIFT_SEMITONES)
            rows.extend(
                _trajectory_rows(
                    case=case,
                    algorithm_name=algorithm.name,
                    algorithm_label=algorithm.display_name,
                    implementation=algorithm.implementation,
                    audio=shifted,
                )
            )
    return pd.DataFrame(rows).reset_index(drop=True)


def _frequencies_from_summary(parameter_summary: str) -> list[float]:
    import json

    params = json.loads(parameter_summary)
    return [float(value) for value in str(params["frequencies_hz"]).split(",")]


def _nearest_ratio(pitch_hz: float, anchors: list[float]) -> tuple[float, float]:
    best_anchor = min(anchors, key=lambda anchor: abs(np.log2(max(pitch_hz, 1e-9) / anchor)))
    ratio = pitch_hz / best_anchor if best_anchor else np.nan
    return best_anchor, ratio


def _classify_observer_belief(pitch_hz: float, physical_freqs: list[float]) -> str:
    if not np.isfinite(pitch_hz):
        return "untracked"

    anchors = sorted(set(physical_freqs + [110.0, 220.0, 440.0, 880.0, 1320.0, 1760.0]))
    nearest, ratio = _nearest_ratio(pitch_hz, anchors)
    cents = abs(1200.0 * np.log2(max(ratio, 1e-9)))
    if cents > 120.0:
        return "mixed"

    if nearest < min(physical_freqs) * 0.75:
        return "subharmonic_seeking"
    if nearest <= 330.0:
        return "subharmonic_seeking"
    if nearest <= 660.0:
        return "fundamental_seeking"
    if nearest <= 1050.0:
        return "octave_seeking"
    return "upper_partial_seeking"


def _build_belief_map(trajectories: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (case_id, algorithm_label), frame in trajectories.groupby(["case_id", "algorithm_label"], sort=False):
        values = frame["inferred_input_f0_hz"].to_numpy(dtype=np.float64)
        clean = values[np.isfinite(values)]
        median_pitch = float(np.nanmedian(clean)) if clean.size else np.nan
        iqr_pitch = float(np.nanpercentile(clean, 75) - np.nanpercentile(clean, 25)) if clean.size else np.nan
        physical_freqs = _frequencies_from_summary(str(frame["parameter_summary"].iloc[0]))
        belief = _classify_observer_belief(median_pitch, physical_freqs)
        rows.append(
            {
                "case_id": case_id,
                "case_family": str(frame["case_family"].iloc[0]),
                "case_label": str(frame["case_label"].iloc[0]),
                "physical_frequencies_hz": ",".join(f"{freq:g}" for freq in physical_freqs),
                "algorithm": str(frame["algorithm"].iloc[0]),
                "algorithm_label": algorithm_label,
                "dominant_inferred_pitch_hz": median_pitch,
                "iqr_inferred_f0_hz": iqr_pitch,
                "mean_confidence": float(frame["pitch_confidence"].mean()),
                "valid_frame_fraction": float(np.mean(np.isfinite(values))),
                "observer_belief": belief,
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def _build_algorithm_summary(belief_map: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for algorithm_label, frame in belief_map.groupby("algorithm_label"):
        counts = frame["observer_belief"].value_counts()
        dominant = str(counts.idxmax()) if not counts.empty else "untracked"
        rows.append(
            {
                "algorithm_label": algorithm_label,
                "case_count": int(len(frame)),
                "dominant_observer_bias": dominant,
                "dominant_count": int(counts.max()) if not counts.empty else 0,
                "subharmonic_count": int(counts.get("subharmonic_seeking", 0)),
                "fundamental_count": int(counts.get("fundamental_seeking", 0)),
                "octave_count": int(counts.get("octave_seeking", 0)),
                "upper_partial_count": int(counts.get("upper_partial_seeking", 0)),
                "mixed_count": int(counts.get("mixed", 0)),
            }
        )
    return pd.DataFrame(rows).sort_values("algorithm_label").reset_index(drop=True)


def _build_case_summary(belief_map: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case_id, frame in belief_map.groupby("case_id"):
        values = frame["dominant_inferred_pitch_hz"].to_numpy(dtype=np.float64)
        clean = values[np.isfinite(values)]
        rows.append(
            {
                "case_id": case_id,
                "case_family": str(frame["case_family"].iloc[0]),
                "case_label": str(frame["case_label"].iloc[0]),
                "belief_set": "|".join(sorted(frame["observer_belief"].unique())),
                "belief_count": int(frame["observer_belief"].nunique()),
                "variance_inferred_state": float(np.var(clean, ddof=0)) if clean.size else np.nan,
                "state_std_hz": float(np.std(clean, ddof=0)) if clean.size else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("variance_inferred_state", ascending=False).reset_index(drop=True)


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cases = build_octave_ambiguity_cases(sr=SR, duration=DURATION)
    trajectories = _build_trajectories(cases)
    belief_map = _build_belief_map(trajectories)
    algorithm_summary = _build_algorithm_summary(belief_map)
    case_summary = _build_case_summary(belief_map)

    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    trajectories.to_csv(artifacts_dir / "08_pitch_trajectories.csv", index=False)
    belief_map.to_csv(artifacts_dir / "08_octave_preference_map.csv", index=False)
    algorithm_summary.to_csv(artifacts_dir / "08_algorithm_bias_summary.csv", index=False)
    case_summary.to_csv(artifacts_dir / "08_case_belief_summary.csv", index=False)
    save_octave_preference_map(belief_map, algorithm_summary, artifacts_dir / "08_octave_preference_map.png")
    return trajectories, belief_map, algorithm_summary, case_summary


def main() -> None:
    trajectories, belief_map, algorithm_summary, case_summary = run_experiment()
    print(f"Wrote {len(trajectories)} frame rows to artifacts/08_pitch_trajectories.csv")
    print("Wrote artifacts/08_octave_preference_map.csv")
    print("Wrote artifacts/08_algorithm_bias_summary.csv")
    print("Wrote artifacts/08_case_belief_summary.csv")
    print("Wrote artifacts/08_octave_preference_map.png")
    print("Question: When multiple pitch interpretations are plausible, what does each observer believe?")
    print(algorithm_summary.to_string(index=False))
    print(case_summary.head(12).to_string(index=False))
    print(belief_map[["case_label", "algorithm_label", "dominant_inferred_pitch_hz", "observer_belief"]].to_string(index=False))


if __name__ == "__main__":
    main()
