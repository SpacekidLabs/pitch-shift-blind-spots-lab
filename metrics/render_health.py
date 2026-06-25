from __future__ import annotations

import numpy as np


def mono_mix(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        return y
    return np.mean(y, axis=1)


def rms_level(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(y**2)))


def peak_level(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return 0.0
    return float(np.max(np.abs(y)))


def frame_rms(y: np.ndarray, frame_length: int = 2048) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return np.zeros(0, dtype=np.float64)
    frame_count = max(1, int(np.ceil(y.size / frame_length)))
    frames = np.array_split(y, frame_count)
    return np.array([rms_level(frame) for frame in frames if frame.size], dtype=np.float64)


def render_health_metrics(reference: np.ndarray, rendered: np.ndarray, frame_length: int = 2048) -> dict[str, float | bool]:
    ref = mono_mix(reference)
    out = mono_mix(rendered)
    ref_rms = rms_level(ref)
    out_rms = rms_level(out)
    ref_peak = peak_level(reference)
    out_peak = peak_level(rendered)
    out_frames = frame_rms(out, frame_length=frame_length)

    relative_rms_db = 20.0 * np.log10(max(out_rms / max(ref_rms, 1e-12), 1e-12))
    peak_ratio = out_peak / max(ref_peak, 1e-12)
    active_fraction = float(np.mean(out_frames > 0.05 * max(ref_rms, 1e-12))) if out_frames.size else 0.0
    median_frame_rms_db = 20.0 * np.log10(max(float(np.median(out_frames)) / max(ref_rms, 1e-12), 1e-12)) if out_frames.size else -240.0
    crest_factor = out_peak / max(out_rms, 1e-12)

    silence_like = relative_rms_db <= -24.0 or active_fraction <= 0.10
    return {
        "reference_rms": ref_rms,
        "render_rms": out_rms,
        "relative_rms_db": float(relative_rms_db),
        "peak_ratio": float(peak_ratio),
        "active_fraction": active_fraction,
        "median_frame_rms_db": float(median_frame_rms_db),
        "crest_factor": float(crest_factor),
        "silence_like_health_flag": bool(silence_like),
    }
