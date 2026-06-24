from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np


@dataclass(frozen=True)
class AmbiguityCase:
    case_id: str
    family: str
    label: str
    ambiguity_amount: float
    ambiguity_units: str
    params: dict[str, float | int | str]
    audio: np.ndarray

    @property
    def parameter_summary(self) -> str:
        return json.dumps(self.params, sort_keys=True)


def _time_axis(sr: int, duration: float) -> np.ndarray:
    n_samples = int(round(sr * duration))
    return np.arange(n_samples, dtype=np.float64) / float(sr)


def _normalize(y: np.ndarray, peak: float = 0.9) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    max_abs = float(np.max(np.abs(y))) if y.size else 0.0
    if max_abs <= 0.0:
        return y.copy()
    return (peak / max_abs) * y


def _sum_oscillators(freqs: list[float], sr: int, duration: float, phases: list[float] | None = None) -> np.ndarray:
    t = _time_axis(sr, duration)
    if phases is None:
        phases = [0.0] * len(freqs)
    y = np.zeros_like(t)
    for freq, phase in zip(freqs, phases):
        y += np.sin(2 * np.pi * freq * t + phase)
    return _normalize(y)


def _vibrato(base_hz: float, depth_semitones: float, sr: int, duration: float, rate_hz: float = 5.5) -> np.ndarray:
    t = _time_axis(sr, duration)
    semitone_offset = depth_semitones * np.sin(2 * np.pi * rate_hz * t)
    inst_freq = base_hz * (2.0 ** (semitone_offset / 12.0))
    phase = 2 * np.pi * np.cumsum(inst_freq) / sr
    return _normalize(np.sin(phase))


def build_ambiguity_sweep(sr: int = 22050, duration: float = 2.0) -> list[AmbiguityCase]:
    cases: list[AmbiguityCase] = []

    two_oscillator_pairs = [
        (440.0, 440.0),
        (440.0, 441.0),
        (440.0, 442.0),
        (440.0, 445.0),
        (440.0, 450.0),
        (440.0, 460.0),
        (440.0, 480.0),
    ]
    for first, second in two_oscillator_pairs:
        delta = second - first
        case_id = f"two_oscillators_{int(delta):02d}hz"
        cases.append(
            AmbiguityCase(
                case_id=case_id,
                family="two_oscillators",
                label=f"440 + {second:g}",
                ambiguity_amount=delta,
                ambiguity_units="Hz separation",
                params={"frequencies_hz": f"{first:g},{second:g}", "separation_hz": delta},
                audio=_sum_oscillators([first, second], sr=sr, duration=duration),
            )
        )

    three_oscillator_sets = [
        [440.0, 443.0, 447.0],
        [440.0, 450.0, 460.0],
        [440.0, 480.0, 520.0],
    ]
    for freqs in three_oscillator_sets:
        spread = max(freqs) - min(freqs)
        case_id = f"three_oscillators_{int(spread):02d}hz"
        cases.append(
            AmbiguityCase(
                case_id=case_id,
                family="three_oscillators",
                label=" + ".join(f"{freq:g}" for freq in freqs),
                ambiguity_amount=spread,
                ambiguity_units="Hz spread",
                params={"frequencies_hz": ",".join(f"{freq:g}" for freq in freqs), "spread_hz": spread},
                audio=_sum_oscillators(freqs, sr=sr, duration=duration),
            )
        )

    for depth in [0.0, 1.0, 2.0, 4.0, 8.0, 12.0]:
        case_id = f"vibrato_depth_{int(depth):02d}st"
        cases.append(
            AmbiguityCase(
                case_id=case_id,
                family="vibrato_depth",
                label=f"{depth:g} semitones",
                ambiguity_amount=depth,
                ambiguity_units="semitones",
                params={"base_hz": 440.0, "depth_semitones": depth, "rate_hz": 5.5},
                audio=_vibrato(440.0, depth, sr=sr, duration=duration),
            )
        )

    for rate in [0.5, 1.0, 2.0, 5.0, 10.0]:
        case_id = f"beating_rate_{str(rate).replace('.', '_')}hz"
        cases.append(
            AmbiguityCase(
                case_id=case_id,
                family="beating_rate",
                label=f"{rate:g} Hz",
                ambiguity_amount=rate,
                ambiguity_units="Hz beat rate",
                params={"frequencies_hz": f"440,{440 + rate:g}", "beat_rate_hz": rate},
                audio=_sum_oscillators([440.0, 440.0 + rate], sr=sr, duration=duration, phases=[0.0, np.pi / 3.0]),
            )
        )

    return cases

