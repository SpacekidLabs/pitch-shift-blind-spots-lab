from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metrics.preflight_risk import HIGH_RISK_THRESHOLD, modulation_aware_preflight_risk_score, risk_level
from metrics.signal_features import compute_signal_features

from .registry import get_algorithm


@dataclass(frozen=True)
class PreflightAdaptiveAnalysis:
    selected_algorithm: str
    selected_algorithm_label: str
    safe_mode: bool
    guard: str
    dry_mix: float
    preflight_risk_score: float
    preflight_risk_level: str
    preflight_reasons: str
    pitch_valid_fraction: float
    pitch_iqr_cents: float
    pitch_modulation_rms_cents: float
    pitch_modulation_peak_rate_hz: float
    pitch_modulation_peak_strength: float
    spectral_flatness: float


def analyze_preflight_adaptive_state(y: np.ndarray, sr: int, n_steps: float) -> PreflightAdaptiveAnalysis:
    features = compute_signal_features(y, sr=sr).as_dict()
    risk_score, reasons = modulation_aware_preflight_risk_score(features, int(n_steps))
    high_risk = risk_score >= HIGH_RISK_THRESHOLD

    if "micro_modulation_trap" in reasons:
        algorithm = get_algorithm("phase_vocoder")
        guard = "micro_modulation_phase_vocoder_guard"
        dry_mix = 0.06
    elif "noise_like_spectrum" in reasons or "untracked_pitch" in reasons:
        algorithm = get_algorithm("phase_vocoder")
        guard = "avoid_period_tracking"
        dry_mix = 0.08 if high_risk else 0.0
    elif "sparse_transients" in reasons:
        algorithm = get_algorithm("phase_vocoder") if high_risk else get_algorithm("wsola")
        guard = "transient_safe_preflight"
        dry_mix = 0.08 if high_risk else 0.0
    elif high_risk:
        algorithm = get_algorithm("phase_vocoder")
        guard = "high_risk_phase_vocoder_fallback"
        dry_mix = 0.06
    elif abs(n_steps) <= 7:
        algorithm = get_algorithm("psola")
        guard = "stable_periodic_light_shift"
        dry_mix = 0.0
    else:
        algorithm = get_algorithm("rubber_band")
        guard = "stable_periodic_large_shift"
        dry_mix = 0.0

    return PreflightAdaptiveAnalysis(
        selected_algorithm=algorithm.name,
        selected_algorithm_label=algorithm.display_name,
        safe_mode=high_risk,
        guard=guard,
        dry_mix=dry_mix,
        preflight_risk_score=risk_score,
        preflight_risk_level=risk_level(risk_score),
        preflight_reasons=reasons,
        pitch_valid_fraction=float(features["pitch_valid_fraction"]),
        pitch_iqr_cents=float(features["pitch_iqr_cents"]),
        pitch_modulation_rms_cents=float(features["pitch_modulation_rms_cents"]),
        pitch_modulation_peak_rate_hz=float(features["pitch_modulation_peak_rate_hz"]),
        pitch_modulation_peak_strength=float(features["pitch_modulation_peak_strength"]),
        spectral_flatness=float(features["spectral_flatness"]),
    )


def pitch_shift_preflight_adaptive_v1(y: np.ndarray, sr: int, n_steps: float) -> tuple[np.ndarray, PreflightAdaptiveAnalysis]:
    analysis = analyze_preflight_adaptive_state(y, sr=sr, n_steps=n_steps)
    shifted = get_algorithm(analysis.selected_algorithm).pitch_shift(y, sr=sr, n_steps=n_steps)
    if analysis.dry_mix <= 0.0:
        return shifted, analysis

    length = min(len(y), len(shifted))
    guarded = shifted.copy()
    guarded[:length] = (1.0 - analysis.dry_mix) * shifted[:length] + analysis.dry_mix * y[:length]
    return guarded, analysis
