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
from signals.ambiguity import AmbiguityCase, build_attractor_basin_sweep
from visualization.attractor_basin_plot import save_attractor_basin_plot


SR = 22050
DURATION = 2.0
SHIFT_SEMITONES = -12


def _trajectory_rows(case: AmbiguityCase, algorithm_name: str, algorithm_label: str, implementation: str, audio: np.ndarray) -> list[dict[str, float | int | str]]:
    factor = 2.0 ** (SHIFT_SEMITONES / 12.0)
    rows = []
    for frame_index, frame in enumerate(framewise_autocorrelation_f0(audio, sr=SR)):
        observed = float(frame["f0_hz"]) if np.isfinite(frame["f0_hz"]) else np.nan
        inferred = observed / factor if factor > 0 and np.isfinite(observed) else np.nan
        rows.append(
            {
                "case_id": case.case_id,
                "case_label": case.label,
                "delta_hz": case.ambiguity_amount,
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
    return pd.DataFrame(rows).sort_values(["delta_hz", "algorithm_label", "frame_index"]).reset_index(drop=True)


def _classify_branch(values: np.ndarray, lower_reference_hz: float) -> tuple[str, float, float, float]:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return "untracked", np.nan, 0.0, 0.0

    median = float(np.nanmedian(clean))
    lower_cutoff = lower_reference_hz * 1.25
    upper_cutoff = lower_reference_hz * 1.75
    upper_frame_fraction = float(np.mean(clean >= upper_cutoff))
    lower_frame_fraction = float(np.mean(clean <= lower_cutoff))
    iqr = float(np.nanpercentile(clean, 75) - np.nanpercentile(clean, 25))

    if 0.10 <= upper_frame_fraction <= 0.90:
        return "mixed_branch", median, upper_frame_fraction, iqr
    if iqr >= 35.0:
        return "mixed_branch", median, upper_frame_fraction, iqr
    if upper_frame_fraction > 0.90 or median >= upper_cutoff:
        return "upper_branch", median, upper_frame_fraction, iqr
    if lower_frame_fraction > 0.90 and median <= lower_cutoff:
        return "lower_branch", median, upper_frame_fraction, iqr
    return "middle_branch", median, upper_frame_fraction, iqr


def _build_branch_map(trajectories: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (delta_hz, algorithm_label), frame in trajectories.groupby(["delta_hz", "algorithm_label"]):
        values = frame["inferred_input_f0_hz"].to_numpy(dtype=np.float64)
        lower_reference = 440.0 + float(delta_hz) / 2.0
        branch, dominant_pitch, upper_fraction, iqr = _classify_branch(values, lower_reference)
        rows.append(
            {
                "delta_hz": float(delta_hz),
                "algorithm": str(frame["algorithm"].iloc[0]),
                "algorithm_label": algorithm_label,
                "dominant_inferred_pitch_hz": dominant_pitch,
                "lower_reference_hz": lower_reference,
                "upper_reference_hz": 2.0 * lower_reference,
                "upper_frame_fraction": upper_fraction,
                "iqr_inferred_f0_hz": iqr,
                "mean_confidence": float(frame["pitch_confidence"].mean()),
                "valid_frame_fraction": float(np.mean(np.isfinite(values))),
                "branch": branch,
            }
        )
    return pd.DataFrame(rows).sort_values(["algorithm_label", "delta_hz"]).reset_index(drop=True)


def _build_state_disagreement(branch_map: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for delta_hz, frame in branch_map.groupby("delta_hz"):
        values = frame["dominant_inferred_pitch_hz"].to_numpy(dtype=np.float64)
        clean = values[np.isfinite(values)]
        branch_counts = frame["branch"].value_counts().to_dict()
        rows.append(
            {
                "delta_hz": float(delta_hz),
                "mean_inferred_state_hz": float(np.mean(clean)) if clean.size else np.nan,
                "variance_inferred_state": float(np.var(clean, ddof=0)) if clean.size else np.nan,
                "state_std_hz": float(np.std(clean, ddof=0)) if clean.size else np.nan,
                "min_inferred_state_hz": float(np.min(clean)) if clean.size else np.nan,
                "max_inferred_state_hz": float(np.max(clean)) if clean.size else np.nan,
                "branch_set": "|".join(sorted(branch_counts)),
                "branch_count": len(branch_counts),
            }
        )
    return pd.DataFrame(rows).sort_values("delta_hz").reset_index(drop=True)


def _build_branch_switches(branch_map: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "algorithm_label",
        "from_delta_hz",
        "to_delta_hz",
        "switch_midpoint_hz",
        "from_branch",
        "to_branch",
        "from_pitch_hz",
        "to_pitch_hz",
    ]
    rows = []
    for algorithm_label, frame in branch_map.groupby("algorithm_label"):
        ordered = frame.sort_values("delta_hz").reset_index(drop=True)
        for index in range(1, len(ordered)):
            previous = ordered.iloc[index - 1]
            current = ordered.iloc[index]
            if previous["branch"] != current["branch"]:
                rows.append(
                    {
                        "algorithm_label": algorithm_label,
                        "from_delta_hz": float(previous["delta_hz"]),
                        "to_delta_hz": float(current["delta_hz"]),
                        "switch_midpoint_hz": float((previous["delta_hz"] + current["delta_hz"]) / 2.0),
                        "from_branch": str(previous["branch"]),
                        "to_branch": str(current["branch"]),
                        "from_pitch_hz": float(previous["dominant_inferred_pitch_hz"]),
                        "to_pitch_hz": float(current["dominant_inferred_pitch_hz"]),
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def _build_hysteresis_note() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "hysteresis_tested": False,
                "reason": "The current algorithm wrappers are stateless batch processors and do not expose continuation state between delta values.",
                "next_step": "Add stateful continuation or warm-started observers before interpreting order-dependent hysteresis.",
            }
        ]
    )


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cases = build_attractor_basin_sweep(sr=SR, duration=DURATION)
    trajectories = _build_trajectories(cases)
    branch_map = _build_branch_map(trajectories)
    state_disagreement = _build_state_disagreement(branch_map)
    branch_switches = _build_branch_switches(branch_map)

    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    trajectories.to_csv(artifacts_dir / "07_pitch_trajectories.csv", index=False)
    branch_map.to_csv(artifacts_dir / "07_branch_map.csv", index=False)
    state_disagreement.to_csv(artifacts_dir / "07_state_disagreement.csv", index=False)
    branch_switches.to_csv(artifacts_dir / "07_branch_switches.csv", index=False)
    _build_hysteresis_note().to_csv(artifacts_dir / "07_hysteresis_note.csv", index=False)
    save_attractor_basin_plot(branch_map, state_disagreement, artifacts_dir / "07_attractor_basin_map.png")
    return trajectories, branch_map, state_disagreement, branch_switches


def main() -> None:
    trajectories, branch_map, state_disagreement, branch_switches = run_experiment()
    print(f"Wrote {len(trajectories)} frame rows to artifacts/07_pitch_trajectories.csv")
    print("Wrote artifacts/07_branch_map.csv")
    print("Wrote artifacts/07_state_disagreement.csv")
    print("Wrote artifacts/07_branch_switches.csv")
    print("Wrote artifacts/07_hysteresis_note.csv")
    print("Wrote artifacts/07_attractor_basin_map.png")
    print("Question: Where do observer attractor branch switches occur?")
    print(branch_switches.to_string(index=False))
    print(state_disagreement.sort_values("variance_inferred_state", ascending=False).head(10).to_string(index=False))
    print(branch_map.groupby(["algorithm_label", "branch"]).size().reset_index(name="count").to_string(index=False))


if __name__ == "__main__":
    main()
