from __future__ import annotations

from pathlib import Path
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.audio_metrics import rms_error, spectral_centroid_difference, spectral_distance
from signals.audio_io import read_wav, write_wav
from visualization.listening_session_html import save_blind_listening_session
from visualization.listening_test_plot import save_listening_test_plot


DEFAULT_SAMPLE_PATH = Path(
    "/Users/user/Documents/Samples/377DrumLoopsSamplePack/2023-01-02 - 002 - 121 bpm - crash-y tom-y.wav"
)
SAMPLE_PATH = Path(os.environ.get("DRUM_LISTENING_SAMPLE", DEFAULT_SAMPLE_PATH))
OUTPUT_DIR = REPO_ROOT / "artifacts" / "26_adaptive_v4_blind_listening_test"
AUDIO_DIR = OUTPUT_DIR / "audio"
EXCERPT_SECONDS = 8.0
SHIFT_STEPS = [3, 7, -12]
RANDOM_SEED = 26026

STIMULI = [
    ("phase_vocoder", "Phase Vocoder", "fixed", "phase_vocoder"),
    ("adaptive_v3", "Adaptive v3", "adaptive_v3", None),
    ("adaptive_v4", "Adaptive v4 Normal", "adaptive_v4", None),
    ("adaptive_v4_wsola_probe", "Adaptive v4 Rescue From WSOLA", "adaptive_v4", "wsola"),
    ("adaptive_v4_rubber_probe", "Adaptive v4 Rescue From Rubber Band", "adaptive_v4", "rubber_band"),
]


def _load_exp25_module():
    module_path = REPO_ROOT / "experiments" / "25_post_render_fallback_selector.py"
    spec = importlib.util.spec_from_file_location("exp25_post_render_fallback_selector", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Experiment 25 module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _to_mono(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y
    return np.mean(y, axis=1)


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

    exp25 = _load_exp25_module()
    sr, source = read_wav(SAMPLE_PATH)
    excerpt_len = min(source.shape[0], int(round(sr * EXCERPT_SECONDS)))
    reference = exp25._fade_edges(source[:excerpt_len], sr=sr)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    write_wav(AUDIO_DIR / "REFERENCE_original_excerpt.wav", sr, reference)

    cache = {}
    pending = []
    for shift in SHIFT_STEPS:
        for policy, label, mode, candidate in STIMULI:
            rendered, metadata = exp25._render_policy(reference, sr=sr, shift=shift, mode=mode, candidate=candidate, cache=cache)
            pending.append(
                {
                    "shift_semitones": shift,
                    "render_policy": policy,
                    "render_label": label,
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
        blind_id = f"V4LT_{blind_index:03d}"
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
                "render_policy": item["render_policy"],
                "render_label": item["render_label"],
                "initial_algorithm_label": item.get("initial_algorithm_label", ""),
                "final_algorithm_label": item.get("final_algorithm_label", ""),
                "fallback_used": item.get("fallback_used", False),
                "rejected_algorithms": item.get("rejected_algorithms", "none"),
                "final_guard": item.get("final_guard", "none"),
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
                "random_seed": RANDOM_SEED,
            }
        ]
    ).to_csv(OUTPUT_DIR / "session_info.csv", index=False)
    (OUTPUT_DIR / "README.md").write_text(
        "\n".join(
            [
                "# Experiment 26 Adaptive v4 Blind Listening Test",
                "",
                "Question:",
                "",
                "```text",
                "Does Adaptive v4 Normal remain preferred when compared blindly against rescue-path renders?",
                "```",
                "",
                "This pack follows the unblinded observation that Adaptive v4 Normal sounded best on the drum loop.",
                "It compares that normal path against Phase Vocoder, Adaptive v3, and forced rescue paths that start",
                "from risky WSOLA/Rubber Band candidates before the post-render health gate falls back.",
                "",
                "1. Open `blind_listening_session.html` in a browser.",
                "2. Listen to the reference first.",
                "3. Score every blind file.",
                "4. Export your scores as CSV.",
                "5. Only then open `private_answer_key.csv`.",
                "",
                "Tracked files:",
                "",
                "- `listening_sheet.csv`",
                "- `blind_manifest.csv`",
                "- `objective_metrics.csv`",
                "- `objective_metrics_plot.png`",
                "- `session_info.csv`",
                "",
                "Local-only files:",
                "",
                "- `audio/`",
                "- `private_answer_key.csv`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    save_blind_listening_session(
        manifest,
        OUTPUT_DIR / "blind_listening_session.html",
        title="Experiment 26 Adaptive v4 Blind Listening Test",
        storage_key="psbsl_exp26_scores_v1",
        export_filename="exp26_adaptive_v4_blind_scores.csv",
    )
    save_listening_test_plot(
        metrics,
        OUTPUT_DIR / "objective_metrics_plot.png",
        title="Exp26 Adaptive v4 Blind Listening Test",
    )
    return manifest, answer_key, metrics


def main() -> None:
    manifest, answer_key, metrics = run_experiment()
    print(f"Wrote {len(manifest)} adaptive-v4 blind listening stimuli to {AUDIO_DIR}")
    print("Wrote artifacts/26_adaptive_v4_blind_listening_test/blind_listening_session.html")
    print("Wrote artifacts/26_adaptive_v4_blind_listening_test/private_answer_key.csv")
    print(manifest.to_string(index=False))
    print(answer_key.to_string(index=False))
    print(metrics.sort_values("composite_stress").to_string(index=False))


if __name__ == "__main__":
    main()
