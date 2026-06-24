from __future__ import annotations

import math

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


def _stft(y: np.ndarray, n_fft: int, hop_length: int) -> tuple[np.ndarray, int]:
    window = np.hanning(n_fft)
    pad = n_fft // 2
    y_pad = np.pad(np.asarray(y, dtype=np.float64), pad_width=pad, mode="reflect")
    if y_pad.size < n_fft:
        y_pad = np.pad(y_pad, (0, n_fft - y_pad.size), mode="constant")
    n_frames = 1 + max(0, (y_pad.size - n_fft) // hop_length)
    frames = np.lib.stride_tricks.sliding_window_view(y_pad, n_fft)[::hop_length][:n_frames]
    return np.fft.rfft(frames * window, axis=1).T, pad


def _istft(D: np.ndarray, n_fft: int, hop_length: int, pad: int) -> np.ndarray:
    window = np.hanning(n_fft)
    n_frames = D.shape[1]
    if n_frames == 0:
        return np.zeros(0, dtype=np.float64)

    out_len = n_fft + hop_length * (n_frames - 1)
    y = np.zeros(out_len, dtype=np.float64)
    norm = np.zeros(out_len, dtype=np.float64)

    for frame_index in range(n_frames):
        start = frame_index * hop_length
        frame = np.fft.irfft(D[:, frame_index], n=n_fft)
        y[start : start + n_fft] += frame * window
        norm[start : start + n_fft] += window**2

    nonzero = norm > np.finfo(float).eps
    y[nonzero] /= norm[nonzero]
    if pad > 0 and y.size > 2 * pad:
        y = y[pad:-pad]
    return y


def _phase_vocoder(D: np.ndarray, rate: float, hop_length: int) -> np.ndarray:
    if rate <= 0:
        raise ValueError("rate must be positive")

    n_bins, n_frames = D.shape
    if n_frames == 0:
        return np.zeros((n_bins, 0), dtype=np.complex128)
    if n_frames == 1:
        return D.copy()

    time_steps = np.arange(0.0, n_frames, rate, dtype=np.float64)
    phi_advance = 2.0 * np.pi * hop_length * np.arange(n_bins, dtype=np.float64) / (2 * (n_bins - 1))
    phase_acc = np.angle(D[:, 0]).astype(np.float64)
    output = np.empty((n_bins, time_steps.size), dtype=np.complex128)

    for out_index, step in enumerate(time_steps):
        frame_index = int(math.floor(step))
        if frame_index >= n_frames - 1:
            mag = np.abs(D[:, -1])
            output[:, out_index] = mag * np.exp(1j * phase_acc)
            continue

        alpha = step - frame_index
        left = D[:, frame_index]
        right = D[:, frame_index + 1]
        mag = (1.0 - alpha) * np.abs(left) + alpha * np.abs(right)
        delta = np.angle(right) - np.angle(left) - phi_advance
        delta = delta - 2.0 * np.pi * np.round(delta / (2.0 * np.pi))
        phase_acc += phi_advance + delta
        output[:, out_index] = mag * np.exp(1j * phase_acc)

    return output


def _local_pitch_shift(y: np.ndarray, sr: int, n_steps: float, n_fft: int, hop_length: int) -> np.ndarray:
    pitch_factor = 2.0 ** (n_steps / 12.0)
    stretch_rate = 1.0 / pitch_factor
    stft_matrix, pad = _stft(y, n_fft=n_fft, hop_length=hop_length)
    stretched = _phase_vocoder(stft_matrix, rate=stretch_rate, hop_length=hop_length)
    stretched_y = _istft(stretched, n_fft=n_fft, hop_length=hop_length, pad=pad)
    return _resample_linear(stretched_y, target_len=len(y))


def pitch_shift_phase_vocoder(y: np.ndarray, sr: int, n_steps: float, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    try:
        import librosa

        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    except Exception:
        return _local_pitch_shift(y, sr=sr, n_steps=n_steps, n_fft=n_fft, hop_length=hop_length)

