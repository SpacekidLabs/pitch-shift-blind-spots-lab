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


def _overlap_score(reference: np.ndarray, candidate: np.ndarray) -> float:
    length = min(reference.size, candidate.size)
    if length <= 0:
        return -np.inf
    reference = reference[:length] - np.mean(reference[:length])
    candidate = candidate[:length] - np.mean(candidate[:length])
    denom = float(np.linalg.norm(reference) * np.linalg.norm(candidate))
    if denom <= 0.0:
        return -np.inf
    return float(np.dot(reference, candidate) / denom)


def _wsola_time_stretch(y: np.ndarray, rate: float, frame_length: int = 1024, hop_length: int = 256, search_radius: int = 128) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return y.copy()
    if rate <= 0:
        raise ValueError("rate must be positive")
    if np.isclose(rate, 1.0):
        return y.copy()

    analysis_hop = max(1, int(round(hop_length * rate)))
    synthesis_hop = hop_length
    window = np.hanning(frame_length)
    n_frames = 1 + int(np.ceil(max(0, y.size - frame_length) / analysis_hop))
    out_len = frame_length + synthesis_hop * max(0, n_frames - 1) + search_radius * 2
    out = np.zeros(out_len, dtype=np.float64)
    norm = np.zeros(out_len, dtype=np.float64)

    analysis_positions = np.arange(n_frames, dtype=int) * analysis_hop
    synth_positions = np.arange(n_frames, dtype=int) * synthesis_hop

    first_start = min(max(0, analysis_positions[0]), max(0, y.size - frame_length))
    first_frame = y[first_start : first_start + frame_length]
    out[: first_frame.size] += first_frame * window[: first_frame.size]
    norm[: first_frame.size] += window[: first_frame.size] ** 2

    for frame_index in range(1, n_frames):
        expected = int(min(max(0, analysis_positions[frame_index]), max(0, y.size - frame_length)))
        synth_start = synth_positions[frame_index]
        overlap_start = max(0, synth_start - frame_length + hop_length)
        reference = out[overlap_start:synth_start]
        if reference.size == 0:
            best_start = expected
        else:
            search_start = max(0, expected - search_radius)
            search_stop = min(y.size - frame_length, expected + search_radius) + 1
            best_score = -np.inf
            best_start = expected
            for candidate_start in range(search_start, search_stop):
                candidate = y[candidate_start : candidate_start + reference.size]
                score = _overlap_score(reference, candidate)
                if score > best_score:
                    best_score = score
                    best_start = candidate_start

        frame = y[best_start : best_start + frame_length]
        if frame.size < frame_length:
            frame = np.pad(frame, (0, frame_length - frame.size), mode="constant")

        out_slice = slice(synth_start, synth_start + frame_length)
        if out_slice.stop > out.size:
            pad = out_slice.stop - out.size
            out = np.pad(out, (0, pad), mode="constant")
            norm = np.pad(norm, (0, pad), mode="constant")

        out[out_slice] += frame * window
        norm[out_slice] += window**2
    nonzero = norm > np.finfo(float).eps
    out[nonzero] /= norm[nonzero]
    return out[: max(frame_length, synth_positions[-1] + frame_length)]


def pitch_shift_wsola(y: np.ndarray, sr: int, n_steps: float, frame_length: int = 1024, hop_length: int = 256) -> np.ndarray:
    pitch_factor = 2.0 ** (n_steps / 12.0)
    stretch_rate = 1.0 / pitch_factor
    stretched = _wsola_time_stretch(y, rate=stretch_rate, frame_length=frame_length, hop_length=hop_length)
    return _resample_linear(stretched, target_len=len(y))
