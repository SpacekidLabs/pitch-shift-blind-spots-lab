from __future__ import annotations

import numpy as np


def _stft_magnitude(y: np.ndarray, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return np.zeros((n_fft // 2 + 1, 0), dtype=np.float64)
    window = np.hanning(n_fft)
    pad = n_fft // 2
    y_pad = np.pad(y, pad_width=pad, mode="reflect")
    if y_pad.size < n_fft:
        y_pad = np.pad(y_pad, (0, n_fft - y_pad.size), mode="constant")
    n_frames = 1 + max(0, (y_pad.size - n_fft) // hop_length)
    frames = np.lib.stride_tricks.sliding_window_view(y_pad, n_fft)[::hop_length][:n_frames]
    stft = np.fft.rfft(frames * window, axis=1).T
    return np.abs(stft)


def rms_error(reference: np.ndarray, estimated: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    estimated = np.asarray(estimated, dtype=np.float64)
    length = min(reference.size, estimated.size)
    if length == 0:
        return 0.0
    diff = reference[:length] - estimated[:length]
    return float(np.sqrt(np.mean(diff**2)))


def spectral_distance(reference: np.ndarray, estimated: np.ndarray, n_fft: int = 2048, hop_length: int = 512) -> float:
    ref_mag = np.log1p(_stft_magnitude(reference, n_fft=n_fft, hop_length=hop_length))
    est_mag = np.log1p(_stft_magnitude(estimated, n_fft=n_fft, hop_length=hop_length))
    min_frames = min(ref_mag.shape[1], est_mag.shape[1])
    if min_frames == 0:
        return 0.0
    diff = ref_mag[:, :min_frames] - est_mag[:, :min_frames]
    return float(np.sqrt(np.mean(diff**2)))


def _spectral_centroid(y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    mag = _stft_magnitude(y, n_fft=n_fft, hop_length=hop_length)
    if mag.shape[1] == 0:
        return np.zeros(0, dtype=np.float64)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    weighted_sum = np.sum(freqs[:, None] * mag, axis=0)
    magnitude_sum = np.sum(mag, axis=0)
    centroid = np.divide(weighted_sum, magnitude_sum, out=np.zeros_like(weighted_sum), where=magnitude_sum > 0)
    return centroid


def spectral_centroid_difference(reference: np.ndarray, estimated: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512) -> float:
    ref_centroid = _spectral_centroid(reference, sr=sr, n_fft=n_fft, hop_length=hop_length)
    est_centroid = _spectral_centroid(estimated, sr=sr, n_fft=n_fft, hop_length=hop_length)
    min_frames = min(ref_centroid.size, est_centroid.size)
    if min_frames == 0:
        return 0.0
    return float(np.mean(np.abs(ref_centroid[:min_frames] - est_centroid[:min_frames])))

