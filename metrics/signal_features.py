from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .pitch_tracking import framewise_autocorrelation_f0


@dataclass(frozen=True)
class SignalFeatures:
    pitch_valid_fraction: float
    median_pitch_hz: float
    pitch_iqr_cents: float
    spectral_flatness: float
    spectral_centroid_hz: float
    spectral_bandwidth_hz: float
    transient_score: float
    crest_factor: float
    zero_crossing_rate: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _stft_magnitude(y: np.ndarray, n_fft: int = 1024, hop_length: int = 256) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return np.zeros((n_fft // 2 + 1, 0), dtype=np.float64)
    pad = n_fft // 2
    y_pad = np.pad(y, pad_width=pad, mode="reflect")
    if y_pad.size < n_fft:
        y_pad = np.pad(y_pad, (0, n_fft - y_pad.size), mode="constant")
    frames = np.lib.stride_tricks.sliding_window_view(y_pad, n_fft)[::hop_length]
    window = np.hanning(n_fft)
    return np.abs(np.fft.rfft(frames * window, axis=1).T)


def pitch_summary(y: np.ndarray, sr: int) -> tuple[float, float, float]:
    frames = framewise_autocorrelation_f0(y, sr=sr, fmin=55.0, fmax=1200.0, confidence_threshold=0.14)
    values = np.array([frame["f0_hz"] for frame in frames], dtype=np.float64)
    clean = values[np.isfinite(values)]
    valid_fraction = float(clean.size / max(values.size, 1))
    if clean.size == 0:
        return valid_fraction, np.nan, np.nan
    median = float(np.median(clean))
    log_values = 1200.0 * np.log2(np.maximum(clean, 1e-9) / max(median, 1e-9))
    iqr_cents = float(np.percentile(log_values, 75) - np.percentile(log_values, 25))
    return valid_fraction, median, iqr_cents


def spectral_flatness(y: np.ndarray) -> float:
    mag = _stft_magnitude(y)
    if mag.shape[1] == 0:
        return 0.0
    power = np.maximum(mag**2, 1e-12)
    geometric = np.exp(np.mean(np.log(power), axis=0))
    arithmetic = np.mean(power, axis=0)
    flatness = np.divide(geometric, arithmetic, out=np.zeros_like(geometric), where=arithmetic > 0)
    return float(np.mean(flatness))


def transient_score(y: np.ndarray, frame_length: int = 512, hop_length: int = 128) -> float:
    y = np.asarray(y, dtype=np.float64)
    if y.size < frame_length:
        return 0.0
    frames = np.lib.stride_tricks.sliding_window_view(y, frame_length)[::hop_length]
    energy = np.mean(frames**2, axis=1)
    median = float(np.median(energy))
    if median <= 1e-12:
        return float(np.percentile(energy, 95) / 1e-12)
    return float(np.percentile(energy, 95) / median)


def _spectral_moments(y: np.ndarray, sr: int) -> tuple[float, float]:
    mag = _stft_magnitude(y)
    if mag.shape[1] == 0:
        return 0.0, 0.0
    freqs = np.fft.rfftfreq((mag.shape[0] - 1) * 2, d=1.0 / sr)
    magnitude_sum = np.sum(mag, axis=0)
    centroid = np.divide(
        np.sum(freqs[:, None] * mag, axis=0),
        magnitude_sum,
        out=np.zeros_like(magnitude_sum),
        where=magnitude_sum > 0,
    )
    bandwidth = np.divide(
        np.sum(np.abs(freqs[:, None] - centroid[None, :]) * mag, axis=0),
        magnitude_sum,
        out=np.zeros_like(magnitude_sum),
        where=magnitude_sum > 0,
    )
    return float(np.mean(centroid)), float(np.mean(bandwidth))


def _crest_factor(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    rms = float(np.sqrt(np.mean(y**2))) if y.size else 0.0
    if rms <= 1e-12:
        return 0.0
    return float(np.max(np.abs(y)) / rms)


def _zero_crossing_rate(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    if y.size < 2:
        return 0.0
    signs = np.signbit(y)
    return float(np.mean(signs[1:] != signs[:-1]))


def compute_signal_features(y: np.ndarray, sr: int) -> SignalFeatures:
    valid_fraction, median_pitch, iqr_cents = pitch_summary(y, sr=sr)
    centroid, bandwidth = _spectral_moments(y, sr=sr)
    return SignalFeatures(
        pitch_valid_fraction=valid_fraction,
        median_pitch_hz=median_pitch,
        pitch_iqr_cents=iqr_cents,
        spectral_flatness=spectral_flatness(y),
        spectral_centroid_hz=centroid,
        spectral_bandwidth_hz=bandwidth,
        transient_score=transient_score(y),
        crest_factor=_crest_factor(y),
        zero_crossing_rate=_zero_crossing_rate(y),
    )
