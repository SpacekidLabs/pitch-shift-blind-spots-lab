from __future__ import annotations

import numpy as np

from .phase_vocoder import pitch_shift_phase_vocoder
from .wsola import pitch_shift_wsola


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


def _moving_average(y: np.ndarray, width: int = 129) -> np.ndarray:
    width = max(3, int(width))
    if width % 2 == 0:
        width += 1
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(np.asarray(y, dtype=np.float64), kernel, mode="same")


def _blend(a: np.ndarray, b: np.ndarray, mix: float) -> np.ndarray:
    mix = float(np.clip(mix, 0.0, 1.0))
    length = min(a.size, b.size)
    if length == 0:
        return np.zeros(0, dtype=np.float64)
    return (1.0 - mix) * a[:length] + mix * b[:length]


def pitch_shift_rubber_band(y: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    pitch_factor = 2.0 ** (n_steps / 12.0)
    stretch_rate = 1.0 / pitch_factor

    low = _moving_average(y, width=max(31, sr // 200))
    high = np.asarray(y, dtype=np.float64) - low

    low_shifted = pitch_shift_phase_vocoder(low, sr=sr, n_steps=n_steps)
    high_shifted = pitch_shift_wsola(high, sr=sr, n_steps=n_steps)

    mix = 0.55 if abs(n_steps) <= 7 else 0.65
    combined = _blend(low_shifted, high_shifted, mix=mix)
    return _resample_linear(combined, target_len=len(y))

