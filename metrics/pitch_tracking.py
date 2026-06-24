from __future__ import annotations

import numpy as np


def _parabolic_peak(values: np.ndarray, index: int) -> float:
    if index <= 0 or index >= values.size - 1:
        return float(index)
    left = values[index - 1]
    center = values[index]
    right = values[index + 1]
    denom = left - 2.0 * center + right
    if abs(denom) < 1e-12:
        return float(index)
    return float(index + 0.5 * (left - right) / denom)


def framewise_autocorrelation_f0(
    y: np.ndarray,
    sr: int,
    frame_length: int = 4096,
    hop_length: int = 512,
    fmin: float = 60.0,
    fmax: float = 900.0,
    confidence_threshold: float = 0.12,
) -> list[dict[str, float]]:
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return []

    pad = frame_length // 2
    y_pad = np.pad(y, pad_width=pad, mode="reflect")
    if y_pad.size < frame_length:
        y_pad = np.pad(y_pad, (0, frame_length - y_pad.size), mode="constant")

    min_lag = max(1, int(sr / fmax))
    max_lag = min(frame_length - 1, int(sr / fmin))
    window = np.hanning(frame_length)
    frames = np.lib.stride_tricks.sliding_window_view(y_pad, frame_length)[::hop_length]
    trajectories: list[dict[str, float]] = []

    for frame_index, frame in enumerate(frames):
        centered = (frame - np.mean(frame)) * window
        energy = float(np.dot(centered, centered))
        time_seconds = frame_index * hop_length / float(sr)
        if energy <= 1e-12:
            trajectories.append({"time_seconds": time_seconds, "f0_hz": np.nan, "confidence": 0.0})
            continue

        autocorr = np.correlate(centered, centered, mode="full")[frame_length - 1 :]
        autocorr = autocorr / max(float(autocorr[0]), 1e-12)
        search = autocorr[min_lag : max_lag + 1]
        if search.size == 0:
            trajectories.append({"time_seconds": time_seconds, "f0_hz": np.nan, "confidence": 0.0})
            continue

        peak_index = int(np.argmax(search) + min_lag)
        confidence = float(autocorr[peak_index])
        if confidence < confidence_threshold:
            f0 = np.nan
        else:
            refined_lag = _parabolic_peak(autocorr, peak_index)
            f0 = float(sr / refined_lag) if refined_lag > 0 else np.nan
        trajectories.append({"time_seconds": time_seconds, "f0_hz": f0, "confidence": confidence})

    return trajectories

