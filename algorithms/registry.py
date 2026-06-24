from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .phase_vocoder import pitch_shift_phase_vocoder
from .psola import pitch_shift_psola
from .rubber_band import pitch_shift_rubber_band
from .wsola import pitch_shift_wsola


PitchShifter = Callable[[np.ndarray, int, float], np.ndarray]


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    display_name: str
    implementation: str
    pitch_shift: PitchShifter


ALGORITHMS: tuple[AlgorithmSpec, ...] = (
    AlgorithmSpec("phase_vocoder", "Phase Vocoder", "local phase vocoder", pitch_shift_phase_vocoder),
    AlgorithmSpec("wsola", "WSOLA", "local waveform-similarity overlap-add", pitch_shift_wsola),
    AlgorithmSpec("rubber_band", "Rubber Band", "local hybrid spectral/time-domain proxy", pitch_shift_rubber_band),
    AlgorithmSpec("psola", "PSOLA", "local pitch-synchronous overlap-add", pitch_shift_psola),
)


def get_algorithm(name: str) -> AlgorithmSpec:
    for spec in ALGORITHMS:
        if spec.name == name:
            return spec
    raise KeyError(f"Unknown algorithm: {name}")

