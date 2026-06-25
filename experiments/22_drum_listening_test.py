from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.adaptive_preflight import analyze_guarded_preflight_adaptive_state
from algorithms.registry import get_algorithm
from metrics.audio_metrics import rms_error, spectral_centroid_difference, spectral_distance
from signals.audio_io import read_wav, write_wav
from visualization.listening_session_html import save_blind_listening_session
from visualization.listening_test_plot import save_listening_test_plot


DEFAULT_SAMPLE_PATH = Path(
    "/Users/user/Documents/Samples/377DrumLoopsSamplePack/2023-01-02 - 002 - 121 bpm - crash-y tom-y.wav"
)
SAMPLE_PATH = Path(os.environ.get("DRUM_LISTENING_SAMPLE", DEFAULT_SAMPLE_PATH))
OUTPUT_DIR = REPO_ROOT / "artifacts" / "22_drum_listening_test"
AUDIO_DIR = OUTPUT_DIR / "audio"
EXCERPT_SECONDS = 8.0
SHIFT_STEPS = [3, 7, -12]
RANDOM_SEED = 22022

STRATEGIES = [
    ("phase_vocoder", "Phase Vocoder"),
    ("wsola", "WSOLA"),
    ("rubber_band", "Rubber Band"),
    ("psola", "PSOLA"),
    ("adaptive_v3", "Adaptive v3"),
]


def _to_mono(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y
    return np.mean(y, axis=1)


def _fade_edges(y: np.ndarray, sr: int, fade_ms: float = 12.0) -> np.ndarray:
    faded = np.asarray(y, dtype=np.float64).copy()
    fade_len = min(faded.shape[0] // 2, int(round(sr * fade_ms / 1000.0)))
    if fade_len <= 1:
        return faded
    ramp = np.linspace(0.0, 1.0, fade_len, dtype=np.float64)
    faded[:fade_len] *= ramp[:, None]
    faded[-fade_len:] *= ramp[::-1, None]
    return faded


def _match_rms(reference: np.ndarray, rendered: np.ndarray, peak_limit: float = 0.98) -> np.ndarray:
    ref = _to_mono(reference)
    out = np.asarray(rendered, dtype=np.float64).copy()
    out_mono = _to_mono(out)
    ref_rms = float(np.sqrt(np.mean(ref**2))) if ref.size else 0.0
    out_rms = float(np.sqrt(np.mean(out_mono**2))) if out_mono.size else 0.0
    if ref_rms > 1e-12 and out_rms > 1e-12:
        out *= ref_rms / out_rms
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > peak_limit:
        out *= peak_limit / peak
    return out


def _apply_to_channels(y: np.ndarray, fn) -> np.ndarray:
    channels = []
    for channel in range(y.shape[1]):
        channels.append(fn(y[:, channel]))
    min_len = min(channel.size for channel in channels)
    return np.column_stack([channel[:min_len] for channel in channels])


def _pitch_shift_stereo(y: np.ndarray, sr: int, n_steps: int, strategy: str) -> tuple[np.ndarray, dict[str, float | str | bool]]:
    mono = _to_mono(y)
    if strategy == "adaptive_v3":
        analysis = analyze_guarded_preflight_adaptive_state(mono, sr=sr, n_steps=n_steps)
        algorithm = get_algorithm(analysis.selected_algorithm)
        shifted = _apply_to_channels(y, lambda channel: algorithm.pitch_shift(channel, sr=sr, n_steps=n_steps))
        if analysis.dry_mix > 0.0:
            length = min(y.shape[0], shifted.shape[0])
            shifted = shifted.copy()
            shifted[:length] = (1.0 - analysis.dry_mix) * shifted[:length] + analysis.dry_mix * y[:length]
        metadata = asdict(analysis)
        metadata["render_algorithm"] = analysis.selected_algorithm
        metadata["render_guard"] = analysis.guard
        return shifted, metadata

    algorithm = get_algorithm(strategy)
    shifted = _apply_to_channels(y, lambda channel: algorithm.pitch_shift(channel, sr=sr, n_steps=n_steps))
    return shifted, {
        "render_algorithm": algorithm.name,
        "render_guard": "none",
        "selected_algorithm": algorithm.name,
        "selected_algorithm_label": algorithm.display_name,
        "safe_mode": False,
        "guard": "none",
        "dry_mix": 0.0,
    }


def _metric_row(reference: np.ndarray, rendered: np.ndarray, blind_id: str, wav_file: str, shift: int, sr: int) -> dict[str, float | int | str]:
    ref_mono = _to_mono(reference)
    out_mono = _to_mono(rendered)
    return {
        "blind_id": blind_id,
        "wav_file": wav_file,
        "shift_semitones": shift,
        "rms_error": rms_error(ref_mono, out_mono),
        "spectral_distance": spectral_distance(ref_mono, out_mono),
        "spectral_centroid_difference": spectral_centroid_difference(ref_mono, out_mono, sr=sr),
    }


def _add_composite_stress(metrics: pd.DataFrame) -> pd.DataFrame:
    scored = metrics.copy()
    normalized_columns = []
    for column in ["rms_error", "spectral_distance", "spectral_centroid_difference"]:
        normalized = f"{column}_normalized"
        values = scored[column].to_numpy(dtype=np.float64)
        min_value = float(np.min(values))
        max_value = float(np.max(values))
        scored[normalized] = (values - min_value) / (max_value - min_value) if max_value > min_value else 0.0
        normalized_columns.append(normalized)
    scored["composite_stress"] = scored[normalized_columns].mean(axis=1)
    return scored


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Drum listening sample not found: {SAMPLE_PATH}")

    sr, source = read_wav(SAMPLE_PATH)
    excerpt_len = min(source.shape[0], int(round(sr * EXCERPT_SECONDS)))
    reference = _fade_edges(source[:excerpt_len], sr=sr)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    write_wav(AUDIO_DIR / "REFERENCE_original_excerpt.wav", sr, reference)

    pending = []
    for shift in SHIFT_STEPS:
        for strategy, strategy_label in STRATEGIES:
            rendered, metadata = _pitch_shift_stereo(reference, sr=sr, n_steps=shift, strategy=strategy)
            rendered = _match_rms(reference, rendered)
            pending.append(
                {
                    "shift_semitones": shift,
                    "strategy": strategy,
                    "strategy_label": strategy_label,
                    "rendered": rendered,
                    **metadata,
                }
            )

    rng = np.random.default_rng(RANDOM_SEED)
    order = rng.permutation(len(pending))
    manifest_rows = []
    key_rows = []
    metric_rows = []
    for blind_index, pending_index in enumerate(order, start=1):
        item = pending[int(pending_index)]
        blind_id = f"DLT_{blind_index:03d}"
        wav_name = f"{blind_id}.wav"
        write_wav(AUDIO_DIR / wav_name, sr, item["rendered"])
        manifest_rows.append(
            {
                "blind_id": blind_id,
                "wav_file": f"audio/{wav_name}",
                "shift_semitones": item["shift_semitones"],
                "artifact_rating_1_bad_5_clean": "",
                "transient_rating_1_smeared_5_crisp": "",
                "groove_rating_1_bad_5_good": "",
                "overall_rating_1_bad_5_good": "",
                "notes": "",
            }
        )
        key_rows.append(
            {
                "blind_id": blind_id,
                "wav_file": f"audio/{wav_name}",
                "shift_semitones": item["shift_semitones"],
                "strategy": item["strategy"],
                "strategy_label": item["strategy_label"],
                "render_algorithm": item.get("render_algorithm", item["strategy"]),
                "render_guard": item.get("render_guard", "none"),
                "selected_algorithm": item.get("selected_algorithm", item["strategy"]),
                "selected_algorithm_label": item.get("selected_algorithm_label", item["strategy_label"]),
                "safe_mode": item.get("safe_mode", False),
                "dry_mix": item.get("dry_mix", 0.0),
            }
        )
        metric_rows.append(_metric_row(reference, item["rendered"], blind_id, f"audio/{wav_name}", int(item["shift_semitones"]), sr=sr))

    manifest = pd.DataFrame(manifest_rows)
    answer_key = pd.DataFrame(key_rows)
    metrics = _add_composite_stress(pd.DataFrame(metric_rows))

    manifest.to_csv(OUTPUT_DIR / "listening_sheet.csv", index=False)
    manifest[["blind_id", "wav_file", "shift_semitones"]].to_csv(OUTPUT_DIR / "blind_manifest.csv", index=False)
    answer_key.to_csv(OUTPUT_DIR / "private_answer_key.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "objective_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "source_file": SAMPLE_PATH.name,
                "sample_rate": sr,
                "source_channels": source.shape[1],
                "source_duration_seconds": source.shape[0] / sr,
                "excerpt_seconds": reference.shape[0] / sr,
                "stimulus_count": len(manifest),
                "reference_file": "audio/REFERENCE_original_excerpt.wav",
                "level_matched": True,
                "random_seed": RANDOM_SEED,
            }
        ]
    ).to_csv(OUTPUT_DIR / "session_info.csv", index=False)
    (OUTPUT_DIR / "README.md").write_text(
        "\n".join(
            [
                "# Experiment 22 Drum Listening Test",
                "",
                "Recommended:",
                "",
                "1. Open `blind_listening_session.html` in a browser.",
                "2. Listen to the reference first.",
                "3. Score every blind file in the page.",
                "4. Export your scores as CSV.",
                "5. Only then open `private_answer_key.csv`.",
                "",
                "Manual fallback:",
                "",
                "1. Listen to `audio/REFERENCE_original_excerpt.wav` first.",
                "2. Listen through the blinded files listed in `listening_sheet.csv`.",
                "3. Fill in the rating columns before opening `private_answer_key.csv`.",
                "4. Use `objective_metrics.csv` only as a post-listening diagnostic, not as the decision-maker.",
                "",
                "Rating columns use 1 as poor and 5 as strong.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    save_blind_listening_session(manifest, OUTPUT_DIR / "blind_listening_session.html")
    save_listening_test_plot(metrics, OUTPUT_DIR / "objective_metrics_plot.png")
    return manifest, answer_key, metrics


def main() -> None:
    manifest, answer_key, metrics = run_experiment()
    print(f"Wrote {len(manifest)} listening stimuli to {AUDIO_DIR}")
    print("Wrote artifacts/22_drum_listening_test/listening_sheet.csv")
    print("Wrote artifacts/22_drum_listening_test/blind_manifest.csv")
    print("Wrote artifacts/22_drum_listening_test/private_answer_key.csv")
    print("Wrote artifacts/22_drum_listening_test/blind_listening_session.html")
    print("Wrote artifacts/22_drum_listening_test/objective_metrics.csv")
    print("Wrote artifacts/22_drum_listening_test/objective_metrics_plot.png")
    print("Question: Which pitch-shift render feels best on a real drum loop?")
    print(manifest.to_string(index=False))
    print(answer_key.to_string(index=False))
    print(metrics.sort_values('composite_stress').to_string(index=False))


if __name__ == "__main__":
    main()
