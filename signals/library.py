from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


Generator = Callable[[int, float, np.random.Generator], np.ndarray]


@dataclass(frozen=True)
class SignalSpec:
    name: str
    family: str
    generator: Generator


def _time_axis(sr: int, duration: float) -> np.ndarray:
    n_samples = int(round(sr * duration))
    return np.arange(n_samples, dtype=np.float64) / float(sr)


def _normalize(y: np.ndarray, peak: float = 0.9) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    max_abs = float(np.max(np.abs(y))) if y.size else 0.0
    if max_abs <= 0.0:
        return y.copy()
    return (peak / max_abs) * y


def _sine(sr: int, duration: float, rng: np.random.Generator) -> np.ndarray:
    t = _time_axis(sr, duration)
    return _normalize(np.sin(2 * np.pi * 220.0 * t))


def _saw(sr: int, duration: float, rng: np.random.Generator) -> np.ndarray:
    t = _time_axis(sr, duration)
    harmonics = 24
    y = np.zeros_like(t)
    for k in range(1, harmonics + 1):
        y += np.sin(2 * np.pi * 180.0 * k * t) / k
    return _normalize(y)


def _square(sr: int, duration: float, rng: np.random.Generator) -> np.ndarray:
    t = _time_axis(sr, duration)
    y = np.sign(np.sin(2 * np.pi * 150.0 * t))
    y[y == 0] = 1.0
    return _normalize(y)


def _bell_resonator(sr: int, duration: float, rng: np.random.Generator) -> np.ndarray:
    t = _time_axis(sr, duration)
    base = 240.0
    ratios = np.array([1.0, 2.74, 5.80, 8.99, 12.20])
    decays = np.array([4.5, 5.5, 6.5, 8.0, 10.0])
    amplitudes = np.array([1.0, 0.9, 0.7, 0.5, 0.35])
    phases = rng.uniform(0.0, 2 * np.pi, size=ratios.size)
    y = np.zeros_like(t)
    for amp, ratio, decay, phase in zip(amplitudes, ratios, decays, phases):
        freq = base * ratio
        y += amp * np.exp(-decay * t) * np.sin(2 * np.pi * freq * t + phase)
    y += 0.02 * rng.standard_normal(size=t.size)
    return _normalize(y)


def _random_modal_resonator(sr: int, duration: float, rng: np.random.Generator) -> np.ndarray:
    t = _time_axis(sr, duration)
    n_modes = 8
    freqs = np.sort(rng.uniform(140.0, 4200.0, size=n_modes))
    decays = rng.uniform(2.0, 9.5, size=n_modes)
    amps = rng.uniform(0.2, 1.0, size=n_modes)
    phases = rng.uniform(0.0, 2 * np.pi, size=n_modes)
    y = np.zeros_like(t)
    for amp, freq, decay, phase in zip(amps, freqs, decays, phases):
        y += amp * np.exp(-decay * t) * np.sin(2 * np.pi * freq * t + phase)
    return _normalize(y)


def _white_noise(sr: int, duration: float, rng: np.random.Generator) -> np.ndarray:
    n_samples = int(round(sr * duration))
    return _normalize(rng.standard_normal(size=n_samples))


def _pink_noise(sr: int, duration: float, rng: np.random.Generator) -> np.ndarray:
    n_samples = int(round(sr * duration))
    spectrum = rng.standard_normal(n_samples // 2 + 1) + 1j * rng.standard_normal(n_samples // 2 + 1)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sr)
    scale = np.ones_like(freqs)
    nonzero = freqs > 0
    scale[nonzero] = 1.0 / np.sqrt(freqs[nonzero])
    scale[0] = 0.0
    y = np.fft.irfft(spectrum * scale, n=n_samples)
    return _normalize(y)


def _impulse_train(sr: int, duration: float, rng: np.random.Generator) -> np.ndarray:
    n_samples = int(round(sr * duration))
    y = np.zeros(n_samples, dtype=np.float64)
    period = max(1, int(round(sr * 0.09)))
    y[::period] = 1.0
    return _normalize(y)


def _click_sequence(sr: int, duration: float, rng: np.random.Generator) -> np.ndarray:
    n_samples = int(round(sr * duration))
    y = np.zeros(n_samples, dtype=np.float64)
    click_len = max(8, int(round(sr * 0.004)))
    window = np.hanning(click_len)
    period = max(click_len + 1, int(round(sr * 0.14)))
    for start in range(0, n_samples - click_len, period):
        amplitude = rng.uniform(0.65, 1.0)
        y[start : start + click_len] += amplitude * window
    return _normalize(y)


def _vibrato_tone(sr: int, duration: float, rng: np.random.Generator) -> np.ndarray:
    t = _time_axis(sr, duration)
    carrier = 220.0
    vibrato_rate = 5.5
    vibrato_depth = 0.025
    inst_freq = carrier * (1.0 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t))
    phase = 2 * np.pi * np.cumsum(inst_freq) / sr
    return _normalize(np.sin(phase))


def _fm_tone(sr: int, duration: float, rng: np.random.Generator) -> np.ndarray:
    t = _time_axis(sr, duration)
    carrier = 180.0
    mod_freq = 3.0
    mod_index = 5.5
    phase = 2 * np.pi * carrier * t + mod_index * np.sin(2 * np.pi * mod_freq * t)
    return _normalize(np.sin(phase))


SIGNAL_SPECS: tuple[SignalSpec, ...] = (
    SignalSpec("sine", "harmonic", _sine),
    SignalSpec("saw", "harmonic", _saw),
    SignalSpec("square", "harmonic", _square),
    SignalSpec("bell_resonator", "inharmonic", _bell_resonator),
    SignalSpec("random_modal_resonator", "inharmonic", _random_modal_resonator),
    SignalSpec("white_noise", "noise", _white_noise),
    SignalSpec("pink_noise", "noise", _pink_noise),
    SignalSpec("impulse_train", "transient", _impulse_train),
    SignalSpec("click_sequence", "transient", _click_sequence),
    SignalSpec("vibrato_tone", "modulated", _vibrato_tone),
    SignalSpec("fm_tone", "modulated", _fm_tone),
)


def generate_signal(name: str, sr: int = 22050, duration: float = 2.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    for spec in SIGNAL_SPECS:
        if spec.name == name:
            return spec.generator(sr, duration, rng)
    raise KeyError(f"Unknown signal: {name}")


def build_signal_atlas(sr: int = 22050, duration: float = 2.0, seed: int = 0) -> dict[str, np.ndarray]:
    atlas: dict[str, np.ndarray] = {}
    for index, spec in enumerate(SIGNAL_SPECS):
        atlas[spec.name] = generate_signal(spec.name, sr=sr, duration=duration, seed=seed + index)
    return atlas

