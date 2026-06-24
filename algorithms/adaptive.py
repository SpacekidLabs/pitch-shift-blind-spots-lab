from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metrics.pitch_tracking import framewise_autocorrelation_f0

from .registry import ALGORITHMS, AlgorithmSpec, get_algorithm


STATE_NAMES = (
    "stable_periodic",
    "octave_ambiguous",
    "subharmonic_ambiguous",
    "detuned_competing",
    "noise_like",
    "transient_like",
    "modulated_pitch",
    "untracked",
)


@dataclass(frozen=True)
class ObserverAnalysis:
    state: str
    selected_algorithm: str
    selected_algorithm_label: str
    safe_mode: bool
    guard: str
    dry_mix: float
    state_disagreement_octaves: float
    state_disagreement_hz: float
    observer_spread_octaves: float
    pitch_valid_fraction: float
    source_median_pitch_hz: float
    pitch_iqr_cents: float
    spectral_flatness: float
    transient_score: float
    pv_inferred_pitch_hz: float
    psola_inferred_pitch_hz: float
    wsola_inferred_pitch_hz: float
    rubberband_inferred_pitch_hz: float


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


def _spectral_flatness(y: np.ndarray) -> float:
    mag = _stft_magnitude(y)
    if mag.shape[1] == 0:
        return 0.0
    power = np.maximum(mag**2, 1e-12)
    geometric = np.exp(np.mean(np.log(power), axis=0))
    arithmetic = np.mean(power, axis=0)
    flatness = np.divide(geometric, arithmetic, out=np.zeros_like(geometric), where=arithmetic > 0)
    return float(np.mean(flatness))


def _transient_score(y: np.ndarray, frame_length: int = 512, hop_length: int = 128) -> float:
    y = np.asarray(y, dtype=np.float64)
    if y.size < frame_length:
        return 0.0
    frames = np.lib.stride_tricks.sliding_window_view(y, frame_length)[::hop_length]
    energy = np.mean(frames**2, axis=1)
    median = float(np.median(energy))
    if median <= 1e-12:
        return 0.0
    return float(np.percentile(energy, 95) / median)


def _pitch_summary(y: np.ndarray, sr: int) -> tuple[float, float, float]:
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


def _observer_inferred_pitches(y: np.ndarray, sr: int, n_steps: float) -> dict[str, float]:
    factor = 2.0 ** (n_steps / 12.0)
    inferred = {}
    for algorithm in ALGORITHMS:
        shifted = algorithm.pitch_shift(y, sr=sr, n_steps=n_steps)
        _, median_pitch, _ = _pitch_summary(shifted, sr=sr)
        inferred[algorithm.name] = median_pitch / factor if factor > 0 and np.isfinite(median_pitch) else np.nan
    return inferred


def _pitch_variance(values: list[float]) -> tuple[float, float, float]:
    clean = np.array([value for value in values if np.isfinite(value) and value > 0], dtype=np.float64)
    if clean.size < 2:
        return 0.0, 0.0, 0.0
    state_disagreement_hz = float(np.var(clean, ddof=0))
    log_values = np.log2(clean)
    state_disagreement_octaves = float(np.var(log_values, ddof=0))
    observer_spread_octaves = float(np.max(log_values) - np.min(log_values))
    return state_disagreement_octaves, state_disagreement_hz, observer_spread_octaves


def _classify_state(
    valid_fraction: float,
    median_pitch: float,
    pitch_iqr_cents: float,
    flatness: float,
    transient_score: float,
    inferred: dict[str, float],
    disagreement_octaves: float,
    spread_octaves: float,
) -> str:
    if flatness > 0.35 and valid_fraction < 0.65:
        return "noise_like"
    if transient_score > 18.0 and valid_fraction < 0.75:
        return "transient_like"
    if valid_fraction < 0.18 or not np.isfinite(median_pitch):
        return "untracked"
    if np.isfinite(pitch_iqr_cents) and pitch_iqr_cents > 85.0:
        return "modulated_pitch"

    psola_pitch = inferred.get("psola", np.nan)
    others = [inferred.get(name, np.nan) for name in ("phase_vocoder", "wsola", "rubber_band")]
    clean_others = np.array([value for value in others if np.isfinite(value) and value > 0], dtype=np.float64)
    other_median = float(np.median(clean_others)) if clean_others.size else np.nan
    if np.isfinite(psola_pitch) and np.isfinite(other_median):
        ratio = psola_pitch / max(other_median, 1e-9)
        if 1.65 <= ratio <= 2.35:
            return "octave_ambiguous"
        if 0.42 <= ratio <= 0.62:
            return "subharmonic_ambiguous"

    if disagreement_octaves > 0.055 or spread_octaves > 0.65:
        return "detuned_competing"
    return "stable_periodic"


def _select_algorithm(state: str, safe_mode: bool, n_steps: float) -> tuple[AlgorithmSpec, str, float]:
    if state == "stable_periodic":
        algorithm = get_algorithm("psola") if abs(n_steps) <= 7 else get_algorithm("rubber_band")
        return algorithm, "stable_periodic_rule", 0.0
    if state == "octave_ambiguous":
        return get_algorithm("phase_vocoder"), "avoid_psola_octave_bias", 0.08 if safe_mode else 0.0
    if state == "subharmonic_ambiguous":
        return get_algorithm("phase_vocoder"), "phase_vocoder_pitch_guard", 0.10 if safe_mode else 0.0
    if state == "detuned_competing":
        algorithm = get_algorithm("phase_vocoder") if safe_mode else get_algorithm("rubber_band")
        return algorithm, "conservative_competing_pitch_guard", 0.12 if safe_mode else 0.08
    if state == "noise_like":
        return get_algorithm("phase_vocoder"), "avoid_period_trackers", 0.08 if safe_mode else 0.0
    if state == "transient_like":
        algorithm = get_algorithm("phase_vocoder") if safe_mode else get_algorithm("wsola")
        return algorithm, "preserve_transients_with_guard", 0.10 if safe_mode else 0.0
    if state == "modulated_pitch":
        algorithm = get_algorithm("phase_vocoder") if safe_mode else get_algorithm("rubber_band")
        return algorithm, "smooth_modulated_pitch", 0.10 if safe_mode else 0.0
    return get_algorithm("rubber_band"), "fallback_safe_mode", 0.12 if safe_mode else 0.0


def analyze_observer_state(y: np.ndarray, sr: int, n_steps: float) -> ObserverAnalysis:
    valid_fraction, median_pitch, pitch_iqr_cents = _pitch_summary(y, sr=sr)
    flatness = _spectral_flatness(y)
    transient = _transient_score(y)
    inferred = _observer_inferred_pitches(y, sr=sr, n_steps=n_steps)
    disagreement_octaves, disagreement_hz, spread_octaves = _pitch_variance(list(inferred.values()))
    state = _classify_state(
        valid_fraction=valid_fraction,
        median_pitch=median_pitch,
        pitch_iqr_cents=pitch_iqr_cents,
        flatness=flatness,
        transient_score=transient,
        inferred=inferred,
        disagreement_octaves=disagreement_octaves,
        spread_octaves=spread_octaves,
    )
    safe_mode = disagreement_octaves > 0.05 or spread_octaves > 0.60 or state in {"noise_like", "untracked"}
    algorithm, guard, dry_mix = _select_algorithm(state, safe_mode=safe_mode, n_steps=n_steps)
    return ObserverAnalysis(
        state=state,
        selected_algorithm=algorithm.name,
        selected_algorithm_label=algorithm.display_name,
        safe_mode=safe_mode,
        guard=guard,
        dry_mix=dry_mix,
        state_disagreement_octaves=disagreement_octaves,
        state_disagreement_hz=disagreement_hz,
        observer_spread_octaves=spread_octaves,
        pitch_valid_fraction=valid_fraction,
        source_median_pitch_hz=median_pitch,
        pitch_iqr_cents=pitch_iqr_cents,
        spectral_flatness=flatness,
        transient_score=transient,
        pv_inferred_pitch_hz=inferred.get("phase_vocoder", np.nan),
        psola_inferred_pitch_hz=inferred.get("psola", np.nan),
        wsola_inferred_pitch_hz=inferred.get("wsola", np.nan),
        rubberband_inferred_pitch_hz=inferred.get("rubber_band", np.nan),
    )


def pitch_shift_adaptive_v0(y: np.ndarray, sr: int, n_steps: float) -> tuple[np.ndarray, ObserverAnalysis]:
    analysis = analyze_observer_state(y, sr=sr, n_steps=n_steps)
    algorithm = get_algorithm(analysis.selected_algorithm)
    shifted = algorithm.pitch_shift(y, sr=sr, n_steps=n_steps)
    if analysis.dry_mix <= 0.0:
        return shifted, analysis

    length = min(len(y), len(shifted))
    guarded = shifted.copy()
    guarded[:length] = (1.0 - analysis.dry_mix) * shifted[:length] + analysis.dry_mix * y[:length]
    return guarded, analysis
