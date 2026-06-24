from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class DiscoveryCandidate:
    candidate_id: str
    family: str
    name: str
    label: str
    params: dict[str, float | int | str]
    audio: np.ndarray

    @property
    def parameter_summary(self) -> str:
        return json.dumps(self.params, sort_keys=True)


Generator = Callable[[int, float, np.random.Generator], tuple[np.ndarray, dict[str, float | int | str]]]


def _time_axis(sr: int, duration: float) -> np.ndarray:
    n_samples = int(round(sr * duration))
    return np.arange(n_samples, dtype=np.float64) / float(sr)


def _normalize(y: np.ndarray, peak: float = 0.9) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    max_abs = float(np.max(np.abs(y))) if y.size else 0.0
    if max_abs <= 0.0:
        return y.copy()
    return (peak / max_abs) * y


def _pink_noise(n_samples: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    spectrum = rng.standard_normal(n_samples // 2 + 1) + 1j * rng.standard_normal(n_samples // 2 + 1)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sr)
    scale = np.ones_like(freqs)
    nonzero = freqs > 0
    scale[nonzero] = 1.0 / np.sqrt(freqs[nonzero])
    scale[0] = 0.0
    return np.fft.irfft(spectrum * scale, n=n_samples)


def _detuned_oscillators(sr: int, duration: float, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float | int | str]]:
    t = _time_axis(sr, duration)
    base = float(rng.uniform(90.0, 520.0))
    count = int(rng.integers(2, 6))
    detune_cents = float(rng.uniform(3.0, 38.0))
    spread = rng.normal(0.0, detune_cents, size=count)
    amps = rng.uniform(0.35, 1.0, size=count)
    phases = rng.uniform(0.0, 2 * np.pi, size=count)
    y = np.zeros_like(t)
    for amp, cents, phase in zip(amps, spread, phases):
        freq = base * (2.0 ** (cents / 1200.0))
        y += amp * np.sin(2 * np.pi * freq * t + phase)
    return _normalize(y), {"base_hz": round(base, 3), "count": count, "detune_cents": round(detune_cents, 3)}


def _beating_oscillators(sr: int, duration: float, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float | int | str]]:
    t = _time_axis(sr, duration)
    base = float(rng.uniform(80.0, 440.0))
    beat_hz = float(rng.uniform(0.4, 12.0))
    phase_offset = float(rng.uniform(0.0, 2 * np.pi))
    y = np.sin(2 * np.pi * base * t) + rng.uniform(0.7, 1.0) * np.sin(2 * np.pi * (base + beat_hz) * t + phase_offset)
    if rng.random() > 0.5:
        y += 0.35 * np.sin(2 * np.pi * (2 * base + beat_hz * 0.5) * t)
    return _normalize(y), {"base_hz": round(base, 3), "beat_hz": round(beat_hz, 3)}


def _chirp(sr: int, duration: float, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float | int | str]]:
    t = _time_axis(sr, duration)
    f0 = float(rng.uniform(80.0, 700.0))
    f1 = float(rng.uniform(900.0, 5200.0))
    if rng.random() > 0.5:
        f0, f1 = f1, f0
    curve = float(rng.uniform(0.7, 2.4))
    progress = (t / max(duration, 1e-9)) ** curve
    inst_freq = f0 + (f1 - f0) * progress
    phase = 2 * np.pi * np.cumsum(inst_freq) / sr
    envelope = np.sin(np.pi * t / duration) ** rng.uniform(0.2, 1.2)
    return _normalize(envelope * np.sin(phase)), {"start_hz": round(f0, 3), "end_hz": round(f1, 3), "curve": round(curve, 3)}


def _glissando(sr: int, duration: float, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float | int | str]]:
    t = _time_axis(sr, duration)
    f0 = float(rng.uniform(110.0, 420.0))
    ratio = float(rng.uniform(1.5, 5.0))
    direction = -1.0 if rng.random() > 0.5 else 1.0
    f1 = f0 * (ratio if direction > 0 else 1.0 / ratio)
    glide = np.linspace(np.log(max(f0, 1.0)), np.log(max(f1, 1.0)), t.size)
    wobble = 0.015 * np.sin(2 * np.pi * rng.uniform(2.0, 9.0) * t)
    inst_freq = np.exp(glide) * (1.0 + wobble)
    phase = 2 * np.pi * np.cumsum(inst_freq) / sr
    y = np.sin(phase) + 0.35 * np.sin(2.0 * phase + rng.uniform(0.0, 2 * np.pi))
    return _normalize(y), {"start_hz": round(f0, 3), "end_hz": round(f1, 3), "ratio": round(ratio, 3)}


def _chaotic_oscillator(sr: int, duration: float, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float | int | str]]:
    t = _time_axis(sr, duration)
    r = float(rng.uniform(3.72, 3.99))
    x = float(rng.uniform(0.12, 0.88))
    state = np.empty(t.size, dtype=np.float64)
    for index in range(t.size):
        x = r * x * (1.0 - x)
        state[index] = x
    base = float(rng.uniform(90.0, 360.0))
    depth = float(rng.uniform(50.0, 680.0))
    inst_freq = np.clip(base + depth * (state - 0.5), 20.0, sr * 0.45)
    phase = 2 * np.pi * np.cumsum(inst_freq) / sr
    y = np.sin(phase) * (0.35 + 0.65 * state)
    return _normalize(y), {"r": round(r, 5), "base_hz": round(base, 3), "depth_hz": round(depth, 3)}


def _quasi_periodic(sr: int, duration: float, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float | int | str]]:
    t = _time_axis(sr, duration)
    base = float(rng.uniform(90.0, 360.0))
    ratios = np.array([1.0, np.sqrt(2.0), (1.0 + np.sqrt(5.0)) / 2.0, np.pi / 2.0])
    amps = rng.uniform(0.25, 1.0, size=ratios.size)
    phases = rng.uniform(0.0, 2 * np.pi, size=ratios.size)
    y = np.zeros_like(t)
    for amp, ratio, phase in zip(amps, ratios, phases):
        y += amp * np.sin(2 * np.pi * base * ratio * t + phase)
    y *= 0.75 + 0.25 * np.sin(2 * np.pi * rng.uniform(0.3, 3.0) * t)
    return _normalize(y), {"base_hz": round(base, 3), "ratio_set": "sqrt2_phi_pi"}


def _modal_bank(sr: int, duration: float, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float | int | str]]:
    t = _time_axis(sr, duration)
    count = int(rng.integers(6, 18))
    freqs = np.sort(rng.uniform(80.0, 6200.0, size=count))
    decays = rng.uniform(0.4, 8.0, size=count)
    amps = rng.uniform(0.15, 1.0, size=count)
    phases = rng.uniform(0.0, 2 * np.pi, size=count)
    y = np.zeros_like(t)
    for amp, freq, decay, phase in zip(amps, freqs, decays, phases):
        y += amp * np.exp(-decay * t) * np.sin(2 * np.pi * freq * t + phase)
    return _normalize(y), {"mode_count": count, "min_hz": round(float(freqs[0]), 3), "max_hz": round(float(freqs[-1]), 3)}


def _noise_mixture(sr: int, duration: float, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float | int | str]]:
    n_samples = int(round(sr * duration))
    t = _time_axis(sr, duration)
    white_weight = float(rng.uniform(0.15, 0.8))
    pink_weight = float(rng.uniform(0.15, 0.8))
    tone_weight = float(rng.uniform(0.0, 0.5))
    brown = np.cumsum(rng.standard_normal(n_samples))
    brown = brown - np.mean(brown)
    y = white_weight * rng.standard_normal(n_samples)
    y += pink_weight * _pink_noise(n_samples, sr, rng)
    y += rng.uniform(0.05, 0.35) * brown / max(float(np.max(np.abs(brown))), 1e-9)
    y += tone_weight * np.sin(2 * np.pi * rng.uniform(70.0, 1800.0) * t + rng.uniform(0.0, 2 * np.pi))
    return _normalize(y), {"white": round(white_weight, 3), "pink": round(pink_weight, 3), "tone": round(tone_weight, 3)}


DISCOVERY_GENERATORS: tuple[tuple[str, Generator], ...] = (
    ("detuned_oscillators", _detuned_oscillators),
    ("beating_oscillators", _beating_oscillators),
    ("chirps", _chirp),
    ("glissandi", _glissando),
    ("chaotic_oscillators", _chaotic_oscillator),
    ("quasi_periodic_signals", _quasi_periodic),
    ("modal_banks", _modal_bank),
    ("noise_mixtures", _noise_mixture),
)


def build_discovery_candidates(
    sr: int = 22050,
    duration: float = 2.0,
    candidates_per_family: int = 3,
    seed: int = 303,
) -> list[DiscoveryCandidate]:
    rng = np.random.default_rng(seed)
    candidates: list[DiscoveryCandidate] = []
    for family, generator in DISCOVERY_GENERATORS:
        for index in range(candidates_per_family):
            candidate_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
            audio, params = generator(sr, duration, candidate_rng)
            candidate_id = f"{family}_{index + 1:02d}"
            candidates.append(
                DiscoveryCandidate(
                    candidate_id=candidate_id,
                    family=family,
                    name=candidate_id,
                    label=f"{family} / {candidate_id}",
                    params=params,
                    audio=audio,
                )
            )
    return candidates

