from __future__ import annotations

import numpy as np


HIGH_RISK_THRESHOLD = 0.32
MEDIUM_RISK_THRESHOLD = 0.18


def risk_level(score: float) -> str:
    if score >= HIGH_RISK_THRESHOLD:
        return "high"
    if score >= MEDIUM_RISK_THRESHOLD:
        return "medium"
    return "low"


def _finite(value: float, fallback: float = 0.0) -> float:
    return float(value) if np.isfinite(value) else fallback


def preflight_risk_score(features: dict[str, float], shift: int) -> tuple[float, str]:
    score = 0.0
    reasons = []

    pitch_valid = _finite(features["pitch_valid_fraction"])
    if pitch_valid < 0.20:
        score += 0.30
        reasons.append("untracked_pitch")
    elif pitch_valid < 0.65:
        score += 0.15
        reasons.append("partial_pitch_tracking")

    pitch_iqr = _finite(features["pitch_iqr_cents"])
    if pitch_iqr > 300.0:
        score += 0.30
        reasons.append("unstable_pitch")
    elif pitch_iqr > 150.0:
        score += 0.25
        reasons.append("strong_pitch_motion")
    elif pitch_iqr > 80.0:
        score += 0.15
        reasons.append("pitch_motion")

    flatness = _finite(features["spectral_flatness"])
    if pitch_valid > 0.95 and flatness < 0.02:
        score += 0.25
        reasons.append("periodic_representation_bias")
    if flatness > 0.45:
        score += 0.30
        reasons.append("noise_like_spectrum")
    elif flatness > 0.12:
        score += 0.15
        reasons.append("diffuse_spectrum")

    transient = _finite(features["transient_score"])
    if transient > 120.0:
        score += 0.22
        reasons.append("sparse_transients")
    elif transient > 18.0:
        score += 0.12
        reasons.append("transient_energy")

    bandwidth = _finite(features["spectral_bandwidth_hz"])
    centroid = _finite(features["spectral_centroid_hz"])
    if bandwidth > 2500.0:
        score += 0.20
        reasons.append("broadband_structure")
    elif bandwidth > 1400.0:
        score += 0.12
        reasons.append("wide_harmonic_structure")
    elif bandwidth > 800.0 and flatness < 0.02:
        score += 0.20
        reasons.append("sharp_harmonic_structure")
    if centroid > 2600.0:
        score += 0.08
        reasons.append("high_centroid")

    crest = _finite(features["crest_factor"])
    if crest > 12.0:
        score += 0.12
        reasons.append("high_crest_factor")
    elif crest > 5.0:
        score += 0.06
        reasons.append("peaky_waveform")

    zero_crossing = _finite(features["zero_crossing_rate"])
    if zero_crossing > 0.22:
        score += 0.08
        reasons.append("dense_zero_crossings")

    if abs(shift) >= 12:
        score += 0.20
        reasons.append("large_shift")
    elif abs(shift) >= 7:
        score += 0.08
        reasons.append("moderate_shift")

    return min(score, 1.0), "|".join(reasons) if reasons else "stable_source"


def modulation_aware_preflight_risk_score(features: dict[str, float], shift: int) -> tuple[float, str]:
    score, reasons_text = preflight_risk_score(features, shift)
    reasons = [] if reasons_text == "stable_source" else reasons_text.split("|")

    pitch_valid = _finite(features["pitch_valid_fraction"])
    flatness = _finite(features["spectral_flatness"])
    modulation_rms = _finite(features.get("pitch_modulation_rms_cents", np.nan))
    modulation_strength = _finite(features.get("pitch_modulation_peak_strength", np.nan))
    modulation_rate = _finite(features.get("pitch_modulation_peak_rate_hz", np.nan))

    if (
        pitch_valid > 0.95
        and flatness < 0.02
        and modulation_rms > 0.70
        and modulation_strength > 0.35
        and 1.0 <= modulation_rate <= 12.0
    ):
        score += 0.10
        reasons.append("micro_modulation_trap")

    return min(score, 1.0), "|".join(reasons) if reasons else "stable_source"
