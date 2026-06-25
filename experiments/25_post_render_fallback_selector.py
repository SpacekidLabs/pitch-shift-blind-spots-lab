from __future__ import annotations

from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.adaptive_postrender import POST_RENDER_FALLBACK_CHAIN
from algorithms.adaptive_preflight import analyze_guarded_preflight_adaptive_state
from algorithms.registry import get_algorithm
from metrics.audio_metrics import rms_error, spectral_centroid_difference, spectral_distance
from metrics.render_health import render_health_metrics
from signals.audio_io import read_wav
from visualization.postrender_fallback_plot import save_postrender_fallback_plot


DEFAULT_SAMPLE_PATH = Path(
    "/Users/user/Documents/Samples/377DrumLoopsSamplePack/2023-01-02 - 002 - 121 bpm - crash-y tom-y.wav"
)
SAMPLE_PATH = Path(os.environ.get("DRUM_LISTENING_SAMPLE", DEFAULT_SAMPLE_PATH))
OUTPUT_DIR = REPO_ROOT / "artifacts" / "25_post_render_fallback_selector"
EXCERPT_SECONDS = 8.0
PREFLIGHT_SECONDS = 2.0
SHIFT_STEPS = [3, 7, -12]

RENDER_POLICIES = [
    ("fixed_phase_vocoder", "Fixed Phase Vocoder", "fixed", "phase_vocoder"),
    ("fixed_wsola", "Fixed WSOLA", "fixed", "wsola"),
    ("fixed_rubber_band", "Fixed Rubber Band", "fixed", "rubber_band"),
    ("fixed_psola", "Fixed PSOLA", "fixed", "psola"),
    ("adaptive_v3", "Adaptive v3", "adaptive_v3", None),
    ("adaptive_v4", "Adaptive v4", "adaptive_v4", None),
    ("adaptive_v4_wsola_probe", "Adaptive v4 from WSOLA", "adaptive_v4", "wsola"),
    ("adaptive_v4_rubber_probe", "Adaptive v4 from Rubber Band", "adaptive_v4", "rubber_band"),
]


def _to_mono(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y
    return np.mean(y, axis=1)


def _fade_edges(y: np.ndarray, sr: int, fade_ms: float = 12.0) -> np.ndarray:
    faded = np.asarray(y, dtype=np.float64).copy()
    fade_len = min(faded.shape[0] // 2, int(round(sr * fade_ms / 1000.0)))
    if fade_len <= 1:
        return faded
    ramp = np.linspace(0.0, 1.0, fade_len, dtype=np.float64)
    faded[:fade_len] *= ramp[:, None]
    faded[-fade_len:] *= ramp[::-1, None]
    return faded


def _match_rms(reference: np.ndarray, rendered: np.ndarray, peak_limit: float = 0.98) -> np.ndarray:
    ref = _to_mono(reference)
    out = np.asarray(rendered, dtype=np.float64).copy()
    out_mono = _to_mono(out)
    ref_rms = float(np.sqrt(np.mean(ref**2))) if ref.size else 0.0
    out_rms = float(np.sqrt(np.mean(out_mono**2))) if out_mono.size else 0.0
    if ref_rms > 1e-12 and out_rms > 1e-12:
        out *= ref_rms / out_rms
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > peak_limit:
        out *= peak_limit / peak
    return out


def _apply_to_channels(y: np.ndarray, fn) -> np.ndarray:
    channels = []
    for channel in range(y.shape[1]):
        channels.append(fn(y[:, channel]))
    min_len = min(channel.size for channel in channels)
    return np.column_stack([channel[:min_len] for channel in channels])


def _preflight_source(y: np.ndarray, sr: int) -> np.ndarray:
    length = min(y.shape[0], int(round(sr * PREFLIGHT_SECONDS)))
    return _to_mono(y[:length])


def _blend_with_source(source: np.ndarray, rendered: np.ndarray, dry_mix: float) -> np.ndarray:
    length = min(source.shape[0], rendered.shape[0])
    out = rendered.copy()
    out[:length] = (1.0 - dry_mix) * rendered[:length] + dry_mix * source[:length]
    return out


def _cached_algorithm_render(
    y: np.ndarray,
    sr: int,
    shift: int,
    algorithm_name: str,
    cache: dict[tuple[int, str], np.ndarray],
) -> np.ndarray:
    key = (shift, algorithm_name)
    if key not in cache:
        algorithm = get_algorithm(algorithm_name)
        rendered = _apply_to_channels(y, lambda channel: algorithm.pitch_shift(channel, sr=sr, n_steps=shift))
        cache[key] = _match_rms(y, rendered)
    return cache[key]


def _render_policy(
    y: np.ndarray,
    sr: int,
    shift: int,
    mode: str,
    candidate: str | None,
    cache: dict[tuple[int, str], np.ndarray],
) -> tuple[np.ndarray, dict[str, object]]:
    if mode == "fixed":
        if candidate is None:
            raise ValueError("Fixed render policy requires an algorithm candidate.")
        algorithm = get_algorithm(candidate)
        rendered = _cached_algorithm_render(y, sr=sr, shift=shift, algorithm_name=candidate, cache=cache)
        return rendered, {
            "initial_algorithm": candidate,
            "initial_algorithm_label": algorithm.display_name,
            "final_algorithm": candidate,
            "final_algorithm_label": algorithm.display_name,
            "fallback_used": False,
            "attempted_algorithms": candidate,
            "rejected_algorithms": "none",
            "final_guard": "none",
        }

    if mode == "adaptive_v3":
        analysis = analyze_guarded_preflight_adaptive_state(_preflight_source(y, sr), sr=sr, n_steps=shift)
        algorithm = get_algorithm(analysis.selected_algorithm)
        rendered = _cached_algorithm_render(y, sr=sr, shift=shift, algorithm_name=analysis.selected_algorithm, cache=cache)
        if analysis.dry_mix > 0.0:
            rendered = _match_rms(y, _blend_with_source(y, rendered, analysis.dry_mix))
        return rendered, {
            "initial_algorithm": analysis.selected_algorithm,
            "initial_algorithm_label": analysis.selected_algorithm_label,
            "final_algorithm": analysis.selected_algorithm,
            "final_algorithm_label": analysis.selected_algorithm_label,
            "fallback_used": False,
            "attempted_algorithms": analysis.selected_algorithm,
            "rejected_algorithms": "none",
            "final_guard": analysis.guard,
            "preflight_reasons": analysis.preflight_reasons,
            "preflight_risk_score": analysis.preflight_risk_score,
            "preflight_risk_level": analysis.preflight_risk_level,
        }

    if mode == "adaptive_v4":
        preflight = analyze_guarded_preflight_adaptive_state(_preflight_source(y, sr), sr=sr, n_steps=shift)
        candidate_name = candidate or preflight.selected_algorithm
        attempted: list[str] = []
        rejected: list[str] = []
        algorithms_to_try = [candidate_name]
        for fallback in POST_RENDER_FALLBACK_CHAIN:
            if fallback not in algorithms_to_try:
                algorithms_to_try.append(fallback)

        final_render = None
        final_algorithm = candidate_name
        final_guard = "post_render_health_pass"
        final_health = None
        for algorithm_name in algorithms_to_try:
            rendered = _cached_algorithm_render(y, sr=sr, shift=shift, algorithm_name=algorithm_name, cache=cache)
            if algorithm_name == preflight.selected_algorithm and preflight.dry_mix > 0.0:
                rendered = _match_rms(y, _blend_with_source(y, rendered, preflight.dry_mix))
            health = render_health_metrics(y, rendered)
            attempted.append(algorithm_name)
            final_render = rendered
            final_algorithm = algorithm_name
            final_health = health
            if not bool(health["silence_like_health_flag"]):
                break
            rejected.append(algorithm_name)
            final_guard = "post_render_health_fallback"

        if final_render is None or final_health is None:
            raise RuntimeError("Adaptive v4 could not produce a render.")
        if bool(final_health["silence_like_health_flag"]):
            final_guard = "post_render_health_failed_open"

        initial = get_algorithm(candidate_name)
        final = get_algorithm(final_algorithm)
        return final_render, {
            "initial_algorithm": candidate_name,
            "initial_algorithm_label": initial.display_name,
            "final_algorithm": final_algorithm,
            "final_algorithm_label": final.display_name,
            "fallback_used": final_algorithm != candidate_name,
            "attempted_algorithms": "|".join(attempted),
            "rejected_algorithms": "|".join(rejected) if rejected else "none",
            "final_guard": final_guard,
            "preflight_reasons": preflight.preflight_reasons,
            "preflight_risk_score": preflight.preflight_risk_score,
            "preflight_risk_level": preflight.preflight_risk_level,
        }

    raise ValueError(f"Unknown render policy mode: {mode}")


def _metric_row(reference: np.ndarray, rendered: np.ndarray, sr: int, shift: int, policy: str, label: str, metadata: dict[str, object]) -> dict[str, object]:
    ref_mono = _to_mono(reference)
    out_mono = _to_mono(rendered)
    health = render_health_metrics(reference, rendered)
    return {
        "render_policy": policy,
        "render_label": label,
        "shift_semitones": shift,
        "rms_error": rms_error(ref_mono, out_mono),
        "spectral_distance": spectral_distance(ref_mono, out_mono),
        "spectral_centroid_difference": spectral_centroid_difference(ref_mono, out_mono, sr=sr),
        **health,
        **metadata,
    }


def _add_composite_stress(results: pd.DataFrame) -> pd.DataFrame:
    scored = results.copy()
    normalized_columns = []
    for column in ["rms_error", "spectral_distance", "spectral_centroid_difference"]:
        normalized = f"{column}_normalized"
        values = scored[column].to_numpy(dtype=np.float64)
        min_value = float(np.min(values))
        max_value = float(np.max(values))
        scored[normalized] = (values - min_value) / (max_value - min_value) if max_value > min_value else 0.0
        normalized_columns.append(normalized)
    scored["composite_stress"] = scored[normalized_columns].mean(axis=1)
    return scored


def _build_summary(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby(["render_policy", "render_label"], as_index=False)
        .agg(
            case_count=("shift_semitones", "count"),
            health_flag_count=("silence_like_health_flag", "sum"),
            fallback_used_count=("fallback_used", "sum"),
            mean_relative_rms_db=("relative_rms_db", "mean"),
            mean_active_fraction=("active_fraction", "mean"),
            mean_composite_stress=("composite_stress", "mean"),
            final_algorithms=("final_algorithm_label", lambda values: "|".join(sorted(set(map(str, values))))),
        )
        .reset_index(drop=True)
    )
    for column in ["health_flag_count", "fallback_used_count"]:
        summary[column] = summary[column].astype(int)
    return summary.sort_values(["health_flag_count", "fallback_used_count", "mean_composite_stress"]).reset_index(drop=True)


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Drum listening sample not found: {SAMPLE_PATH}")

    sr, source = read_wav(SAMPLE_PATH)
    excerpt_len = min(source.shape[0], int(round(sr * EXCERPT_SECONDS)))
    reference = _fade_edges(source[:excerpt_len], sr=sr)
    rows = []
    cache: dict[tuple[int, str], np.ndarray] = {}
    for shift in SHIFT_STEPS:
        for policy, label, mode, candidate in RENDER_POLICIES:
            rendered, metadata = _render_policy(reference, sr=sr, shift=shift, mode=mode, candidate=candidate, cache=cache)
            rows.append(_metric_row(reference, rendered, sr=sr, shift=shift, policy=policy, label=label, metadata=metadata))

    results = _add_composite_stress(pd.DataFrame(rows))
    summary = _build_summary(results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "25_post_render_fallback_results.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "25_post_render_fallback_summary.csv", index=False)
    save_postrender_fallback_plot(summary, OUTPUT_DIR / "25_post_render_fallback_plot.png")
    return results, summary


def main() -> None:
    results, summary = run_experiment()
    print(f"Wrote {len(results)} rows to artifacts/25_post_render_fallback_selector")
    print("Question: Can a post-render health gate rescue collapsed candidate renders?")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
