from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.phase_vocoder import pitch_shift_phase_vocoder
from metrics.audio_metrics import rms_error, spectral_centroid_difference, spectral_distance
from metrics.disagreement import add_composite_stress
from signals.library import SIGNAL_SPECS, build_signal_atlas
from visualization.noise_guard_plot import save_noise_guard_plot


SR = 22050
DURATION = 2.0
SHIFT_STEPS = [3, 7, 12, -12]
DRY_MIXES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]


def _blend(source: np.ndarray, shifted: np.ndarray, dry_mix: float) -> np.ndarray:
    length = min(len(source), len(shifted))
    blended = shifted.copy()
    blended[:length] = (1.0 - dry_mix) * shifted[:length] + dry_mix * source[:length]
    return blended


def _metric_row(
    source: np.ndarray,
    shifted: np.ndarray,
    signal_family: str,
    signal_name: str,
    shift: int,
    dry_mix: float,
) -> dict[str, float | int | str]:
    return {
        "signal_family": signal_family,
        "signal_name": signal_name,
        "signal_label": f"{signal_family} / {signal_name}",
        "shift_semitones": shift,
        "dry_mix": dry_mix,
        "algorithm": "phase_vocoder_guarded",
        "algorithm_label": "Phase Vocoder Guarded",
        "rms_error": rms_error(source, shifted),
        "spectral_distance": spectral_distance(source, shifted),
        "spectral_centroid_difference": spectral_centroid_difference(source, shifted, sr=SR),
    }


def _load_catastrophic_threshold() -> float:
    path = REPO_ROOT / "artifacts" / "16_results.csv"
    if path.exists():
        results = pd.read_csv(path)
        fixed = results[results["algorithm"].isin({"phase_vocoder", "psola", "wsola", "rubber_band"})]
        return float(fixed["composite_stress"].quantile(0.85))
    return 0.404


def _build_summary(results: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for dry_mix, frame in results.groupby("dry_mix"):
        stresses = frame["composite_stress"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "dry_mix": dry_mix,
                "case_count": int(len(frame)),
                "catastrophic_count": int(np.sum(stresses >= threshold)),
                "mean_stress": float(np.mean(stresses)),
                "max_stress": float(np.max(stresses)),
            }
        )
    table = pd.DataFrame(rows).sort_values(["catastrophic_count", "mean_stress"]).reset_index(drop=True)
    best = table.iloc[0]
    return pd.DataFrame(
        [
            {
                "catastrophic_threshold": threshold,
                "best_dry_mix": float(best["dry_mix"]),
                "best_catastrophic_count": int(best["catastrophic_count"]),
                "best_mean_stress": float(best["mean_stress"]),
                "best_max_stress": float(best["max_stress"]),
            }
        ]
    )


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    atlas = build_signal_atlas(sr=SR, duration=DURATION, seed=7)
    rows = []
    for spec in SIGNAL_SPECS:
        if spec.family != "noise":
            continue
        source = atlas[spec.name]
        for shift in SHIFT_STEPS:
            shifted = pitch_shift_phase_vocoder(source, sr=SR, n_steps=shift)
            for dry_mix in DRY_MIXES:
                guarded = _blend(source, shifted, dry_mix=dry_mix)
                rows.append(_metric_row(source, guarded, spec.family, spec.name, shift, dry_mix))

    results = add_composite_stress(pd.DataFrame(rows))
    threshold = _load_catastrophic_threshold()
    results["catastrophic_threshold"] = threshold
    results["catastrophic"] = results["composite_stress"] >= threshold
    dry_mix_summary = (
        results.groupby("dry_mix", as_index=False)
        .agg(
            catastrophic_count=("catastrophic", "sum"),
            mean_stress=("composite_stress", "mean"),
            max_stress=("composite_stress", "max"),
        )
        .sort_values(["catastrophic_count", "mean_stress"])
        .reset_index(drop=True)
    )
    summary = _build_summary(results, threshold=threshold)

    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(artifacts_dir / "17_noise_guard_results.csv", index=False)
    dry_mix_summary.to_csv(artifacts_dir / "17_noise_guard_dry_mix_summary.csv", index=False)
    summary.to_csv(artifacts_dir / "17_noise_guard_summary.csv", index=False)
    save_noise_guard_plot(results, summary, artifacts_dir / "17_noise_guard_map.png")
    return results, dry_mix_summary, summary


def main() -> None:
    results, dry_mix_summary, summary = run_experiment()
    print(f"Wrote {len(results)} rows to artifacts/17_noise_guard_results.csv")
    print("Wrote artifacts/17_noise_guard_dry_mix_summary.csv")
    print("Wrote artifacts/17_noise_guard_summary.csv")
    print("Wrote artifacts/17_noise_guard_map.png")
    print("Question: Can a dry/wet guard remove the Phase Vocoder fallback failure on noise?")
    print(dry_mix_summary.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
