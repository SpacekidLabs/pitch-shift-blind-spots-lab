from __future__ import annotations

import numpy as np

from .audio_metrics import rms_error, spectral_distance


def blend_with_source(source: np.ndarray, shifted: np.ndarray, dry_mix: float) -> np.ndarray:
    length = min(len(source), len(shifted))
    blended = shifted.copy()
    blended[:length] = (1.0 - dry_mix) * shifted[:length] + dry_mix * source[:length]
    return blended


def shift_retention(source: np.ndarray, fully_shifted: np.ndarray, guarded: np.ndarray) -> float:
    full_shift_distance = spectral_distance(source, fully_shifted)
    guarded_distance = spectral_distance(source, guarded)
    if full_shift_distance <= 1e-12:
        return 0.0
    return float(np.clip(guarded_distance / full_shift_distance, 0.0, 1.5))


def dry_similarity(source: np.ndarray, guarded: np.ndarray) -> float:
    source = np.asarray(source, dtype=np.float64)
    guarded = np.asarray(guarded, dtype=np.float64)
    length = min(source.size, guarded.size)
    if length == 0:
        return 1.0
    source_rms = np.sqrt(np.mean(source[:length] ** 2))
    error = rms_error(source[:length], guarded[:length])
    return float(max(0.0, 1.0 - error / max(source_rms, 1e-12)))


def peak_preservation(source: np.ndarray, guarded: np.ndarray) -> float:
    source_peak = float(np.max(np.abs(source))) if source.size else 0.0
    guarded_peak = float(np.max(np.abs(guarded))) if guarded.size else 0.0
    if source_peak <= 1e-12:
        return 1.0
    return float(np.clip(guarded_peak / source_peak, 0.0, 2.0))
