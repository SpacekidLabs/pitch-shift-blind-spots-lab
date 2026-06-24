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
from signals.ambiguity import AmbiguityCase, build_critical_ambiguity_sweep
from visualization.observer_state_plot import save_observer_state_plot


SR = 22050
DURATION = 2.0
SHIFT_SEMITONES = -12
TARGET_CASE_IDS = {
    "critical_two_oscillators_2_0hz",
    "critical_vibrato_depth_1_0st",
    "critical_beating_rate_0_5hz",
}


def _select_cases() -> list[AmbiguityCase]:
    return [case for case in build_critical_ambiguity_sweep(sr=SR, duration=DURATION) if case.case_id in TARGET_CASE_IDS]


def _trajectory_rows(
    case: AmbiguityCase,
    algorithm: str,
    algorithm_label: str,
    implementation: str,
    audio: np.ndarray,
    shift_semitones: int,
) -> list[dict[str, float | int | str]]:
    factor = 2.0 ** (shift_semitones / 12.0)
    rows = []
    for frame_index, frame in enumerate(framewise_autocorrelation_f0(audio, sr=SR)):
        observed = float(frame["f0_hz"]) if np.isfinite(frame["f0_hz"]) else np.nan
        inferred = observed / factor if factor > 0 and np.isfinite(observed) else np.nan
        rows.append(
            {
                "case_id": case.case_id,
                "sweep_family": case.family,
                "case_label": case.label,
                "ambiguity_amount": case.ambiguity_amount,
                "ambiguity_units": case.ambiguity_units,
                "parameter_summary": case.parameter_summary,
                "algorithm": algorithm,
                "algorithm_label": algorithm_label,
                "implementation": implementation,
                "shift_semitones": shift_semitones,
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
        rows.extend(
            _trajectory_rows(
                case=case,
                algorithm="input",
                algorithm_label="Input",
                implementation="source signal",
                audio=case.audio,
                shift_semitones=0,
            )
        )
        for algorithm in ALGORITHMS:
            shifted = algorithm.pitch_shift(case.audio, sr=SR, n_steps=SHIFT_SEMITONES)
            rows.extend(
                _trajectory_rows(
                    case=case,
                    algorithm=algorithm.name,
                    algorithm_label=algorithm.display_name,
                    implementation=algorithm.implementation,
                    audio=shifted,
                    shift_semitones=SHIFT_SEMITONES,
                )
            )
    return pd.DataFrame(rows).sort_values(["case_id", "algorithm_label", "frame_index"]).reset_index(drop=True)


def _count_jumps(values: np.ndarray, semitone_threshold: float = 0.5) -> int:
    clean = values[np.isfinite(values)]
    if clean.size < 2:
        return 0
    cents = 12.0 * np.log2(clean[1:] / clean[:-1])
    return int(np.sum(np.abs(cents) >= semitone_threshold))


def _oscillation_index(values: np.ndarray) -> float:
    clean = values[np.isfinite(values)]
    if clean.size < 3:
        return 0.0
    diffs = np.diff(clean)
    signs = np.sign(diffs[np.abs(diffs) > 1e-6])
    if signs.size < 2:
        return 0.0
    return float(np.mean(signs[1:] != signs[:-1]))


def _classify_trajectory(median_f0: float, iqr_f0: float, jump_count: int, oscillation_index: float) -> str:
    if not np.isfinite(median_f0):
        return "untracked"
    if median_f0 >= 660.0:
        return "octave_attractor"
    if jump_count > 0:
        return "jumping"
    if iqr_f0 >= 10.0 or oscillation_index >= 0.25:
        return "oscillating"
    return "stable_lock"


def _build_trajectory_summary(trajectories: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (case_id, algorithm_label), frame in trajectories.groupby(["case_id", "algorithm_label"]):
        values = frame["inferred_input_f0_hz"].to_numpy(dtype=np.float64)
        clean = values[np.isfinite(values)]
        median_f0 = float(np.nanmedian(values)) if clean.size else np.nan
        iqr_f0 = float(np.nanpercentile(values, 75) - np.nanpercentile(values, 25)) if clean.size else np.nan
        jump_count = _count_jumps(values)
        oscillation_index = _oscillation_index(values)
        rows.append(
            {
                "case_id": case_id,
                "case_label": str(frame["case_label"].iloc[0]),
                "algorithm_label": algorithm_label,
                "median_inferred_f0_hz": median_f0,
                "iqr_inferred_f0_hz": iqr_f0,
                "min_inferred_f0_hz": float(np.nanmin(values)) if clean.size else np.nan,
                "max_inferred_f0_hz": float(np.nanmax(values)) if clean.size else np.nan,
                "mean_confidence": float(frame["pitch_confidence"].mean()),
                "valid_frame_fraction": float(np.mean(np.isfinite(values))),
                "jump_count": jump_count,
                "oscillation_index": oscillation_index,
                "state_behavior": _classify_trajectory(median_f0, iqr_f0, jump_count, oscillation_index),
            }
        )
    return pd.DataFrame(rows).sort_values(["case_id", "algorithm_label"]).reset_index(drop=True)


def _classify_case(frame_std: pd.Series) -> str:
    if frame_std.empty:
        return "untracked"
    collapsed_fraction = float(np.mean(frame_std < 5.0))
    divergent_fraction = float(np.mean(frame_std > 20.0))
    if collapsed_fraction >= 0.8:
        return "collapsed"
    if divergent_fraction >= 0.8:
        return "persistent_divergence"
    return "mixed_state"


def _build_case_state_summary(trajectories: pd.DataFrame) -> pd.DataFrame:
    rows = []
    algorithm_frames = trajectories[trajectories["algorithm_label"] != "Input"]
    for case_id, frame in algorithm_frames.groupby("case_id"):
        pivot = frame.pivot_table(index="frame_index", columns="algorithm_label", values="inferred_input_f0_hz", aggfunc="mean")
        frame_std = pivot.std(axis=1, skipna=True)
        rows.append(
            {
                "case_id": case_id,
                "case_label": str(frame["case_label"].iloc[0]),
                "mean_cross_observer_std_hz": float(frame_std.mean()),
                "max_cross_observer_std_hz": float(frame_std.max()),
                "collapsed_frame_fraction_std_lt_5hz": float(np.mean(frame_std < 5.0)),
                "divergent_frame_fraction_std_gt_20hz": float(np.mean(frame_std > 20.0)),
                "valid_frame_count": int(frame_std.notna().sum()),
                "case_state_behavior": _classify_case(frame_std),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_cross_observer_std_hz", ascending=False).reset_index(drop=True)


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cases = _select_cases()
    trajectories = _build_trajectories(cases)
    trajectory_summary = _build_trajectory_summary(trajectories)
    case_summary = _build_case_state_summary(trajectories)

    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    trajectories.to_csv(artifacts_dir / "06_pitch_trajectories.csv", index=False)
    trajectory_summary.to_csv(artifacts_dir / "06_trajectory_summary.csv", index=False)
    case_summary.to_csv(artifacts_dir / "06_case_state_summary.csv", index=False)
    save_observer_state_plot(trajectories, artifacts_dir / "06_observer_state_space.png")
    return trajectories, trajectory_summary, case_summary


def main() -> None:
    trajectories, trajectory_summary, case_summary = run_experiment()
    print(f"Wrote {len(trajectories)} frame rows to artifacts/06_pitch_trajectories.csv")
    print("Wrote artifacts/06_trajectory_summary.csv")
    print("Wrote artifacts/06_case_state_summary.csv")
    print("Wrote artifacts/06_observer_state_space.png")
    print("Question: What internal state trajectory does each observer infer?")
    print(case_summary.to_string(index=False))
    print(trajectory_summary.to_string(index=False))


if __name__ == "__main__":
    main()
