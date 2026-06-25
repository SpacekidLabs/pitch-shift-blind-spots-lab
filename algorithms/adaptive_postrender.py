from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metrics.render_health import render_health_metrics

from .adaptive_preflight import analyze_guarded_preflight_adaptive_state
from .registry import get_algorithm


POST_RENDER_FALLBACK_CHAIN = ("phase_vocoder", "psola")
PREFLIGHT_MAX_SECONDS = 2.0


@dataclass(frozen=True)
class PostRenderFallbackAnalysis:
    initial_algorithm: str
    initial_algorithm_label: str
    final_algorithm: str
    final_algorithm_label: str
    fallback_used: bool
    attempted_algorithms: str
    rejected_algorithms: str
    final_guard: str
    final_relative_rms_db: float
    final_active_fraction: float
    final_silence_like_health_flag: bool
    preflight_reasons: str
    preflight_risk_score: float
    preflight_risk_level: str


def _blend_with_source(source: np.ndarray, rendered: np.ndarray, dry_mix: float) -> np.ndarray:
    length = min(source.size, rendered.size)
    out = rendered.copy()
    out[:length] = (1.0 - dry_mix) * rendered[:length] + dry_mix * source[:length]
    return out


def _render_algorithm(y: np.ndarray, sr: int, n_steps: float, algorithm_name: str) -> tuple[np.ndarray, str]:
    algorithm = get_algorithm(algorithm_name)
    return algorithm.pitch_shift(y, sr=sr, n_steps=n_steps), algorithm.display_name


def pitch_shift_adaptive_v4(
    y: np.ndarray,
    sr: int,
    n_steps: float,
    initial_algorithm: str | None = None,
    fallback_chain: tuple[str, ...] = POST_RENDER_FALLBACK_CHAIN,
) -> tuple[np.ndarray, PostRenderFallbackAnalysis]:
    preflight_len = min(y.size, int(round(sr * PREFLIGHT_MAX_SECONDS)))
    preflight_source = y[:preflight_len] if preflight_len > 0 else y
    preflight = analyze_guarded_preflight_adaptive_state(preflight_source, sr=sr, n_steps=n_steps)
    candidate_name = initial_algorithm or preflight.selected_algorithm
    candidate_label = get_algorithm(candidate_name).display_name
    attempted: list[str] = []
    rejected: list[str] = []

    algorithms_to_try = [candidate_name]
    for fallback in fallback_chain:
        if fallback not in algorithms_to_try:
            algorithms_to_try.append(fallback)

    best_render: np.ndarray | None = None
    best_name = candidate_name
    best_label = candidate_label
    best_health: dict[str, float | bool] | None = None
    final_guard = "post_render_health_pass"

    for algorithm_name in algorithms_to_try:
        rendered, label = _render_algorithm(y, sr=sr, n_steps=n_steps, algorithm_name=algorithm_name)
        if algorithm_name == preflight.selected_algorithm and preflight.dry_mix > 0.0:
            rendered = _blend_with_source(y, rendered, preflight.dry_mix)
        health = render_health_metrics(y, rendered)
        attempted.append(algorithm_name)
        best_render = rendered
        best_name = algorithm_name
        best_label = label
        best_health = health
        if not bool(health["silence_like_health_flag"]):
            break
        rejected.append(algorithm_name)
        final_guard = "post_render_health_fallback"

    if best_render is None or best_health is None:
        raise RuntimeError("Adaptive v4 could not produce a render.")

    fallback_used = best_name != candidate_name
    if bool(best_health["silence_like_health_flag"]):
        final_guard = "post_render_health_failed_open"

    analysis = PostRenderFallbackAnalysis(
        initial_algorithm=candidate_name,
        initial_algorithm_label=candidate_label,
        final_algorithm=best_name,
        final_algorithm_label=best_label,
        fallback_used=fallback_used,
        attempted_algorithms="|".join(attempted),
        rejected_algorithms="|".join(rejected) if rejected else "none",
        final_guard=final_guard,
        final_relative_rms_db=float(best_health["relative_rms_db"]),
        final_active_fraction=float(best_health["active_fraction"]),
        final_silence_like_health_flag=bool(best_health["silence_like_health_flag"]),
        preflight_reasons=preflight.preflight_reasons,
        preflight_risk_score=preflight.preflight_risk_score,
        preflight_risk_level=preflight.preflight_risk_level,
    )
    return best_render, analysis
