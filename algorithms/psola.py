from __future__ import annotations

import numpy as np


def _resample_linear(y: np.ndarray, target_len: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if target_len <= 0:
        return np.zeros(0, dtype=np.float64)
    if y.size == 0:
        return np.zeros(target_len, dtype=np.float64)
    if y.size == 1:
        return np.full(target_len, float(y[0]), dtype=np.float64)
    x_old = np.linspace(0.0, 1.0, num=y.size, endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=True)
    return np.interp(x_new, x_old, y)


def _estimate_period(y: np.ndarray, sr: int) -> int:
    y = np.asarray(y, dtype=np.float64)
    if y.size < 8:
        return max(1, y.size)

    centered = y - np.mean(y)
    window = centered[: min(centered.size, int(0.5 * sr))]
    if window.size < 8:
        window = centered
    if np.allclose(window, 0.0):
        return max(32, sr // 200)

    min_period = max(16, sr // 1000)
    max_period = min(window.size - 1, sr // 40)
    if max_period <= min_period:
        return max(32, sr // 200)

    autocorr = np.correlate(window, window, mode="full")[window.size - 1 :]
    segment = autocorr[min_period:max_period]
    if segment.size == 0:
        return max(32, sr // 200)

    peak = int(np.argmax(segment) + min_period)
    return max(16, peak)


def _pitch_marks(length: int, period: int) -> np.ndarray:
    period = max(1, period)
    return np.arange(period, max(period + 1, length), period, dtype=int)


def _grains_from_marks(y: np.ndarray, marks: np.ndarray, period: int) -> list[np.ndarray]:
    half = max(8, period)
    window = np.hanning(2 * half + 1)
    grains: list[np.ndarray] = []
    for mark in marks:
        start = max(0, mark - half)
        stop = min(y.size, mark + half + 1)
        grain = y[start:stop]
        if grain.size < window.size:
            grain = np.pad(grain, (0, window.size - grain.size), mode="constant")
        grains.append(grain * window)
    return grains


def _psola_time_stretch(y: np.ndarray, sr: int, rate: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return y.copy()
    if rate <= 0:
        raise ValueError("rate must be positive")
    if np.isclose(rate, 1.0):
        return y.copy()

    period = _estimate_period(y, sr)
    analysis_marks = _pitch_marks(y.size, period)
    if analysis_marks.size == 0:
        return y.copy()

    grains = _grains_from_marks(y, analysis_marks, period=period)
    synth_period = max(1, int(round(period * rate)))
    synth_marks = _pitch_marks(int(round(y.size * rate)), synth_period)
    if synth_marks.size == 0:
        synth_marks = np.array([period], dtype=int)

    out_len = max(y.size, synth_marks[-1] + 2 * period + 1)
    out = np.zeros(out_len, dtype=np.float64)
    norm = np.zeros(out_len, dtype=np.float64)

    for synth_index, synth_mark in enumerate(synth_marks):
        grain = grains[int(round(synth_index / max(rate, 1e-9))) % len(grains)]
        center = len(grain) // 2
        start = max(0, synth_mark - center)
        stop = min(out_len, start + grain.size)
        grain_slice = grain[: stop - start]
        out[start:stop] += grain_slice
        norm[start:stop] += np.abs(grain_slice) > 0

    nonzero = norm > 0
    out[nonzero] /= norm[nonzero]
    return out


def pitch_shift_psola(y: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    pitch_factor = 2.0 ** (n_steps / 12.0)
    stretch_rate = 1.0 / pitch_factor
    stretched = _psola_time_stretch(y, sr=sr, rate=stretch_rate)
    return _resample_linear(stretched, target_len=len(y))

