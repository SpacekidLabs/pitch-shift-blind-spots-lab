from __future__ import annotations

from pathlib import Path
import wave

import numpy as np


def read_wav(path: str | Path) -> tuple[int, np.ndarray]:
    path = Path(path)
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sr = wav.getframerate()
        raw = wav.readframes(wav.getnframes())

    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        audio = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64)
        audio = data / float(2**15)
    elif sample_width == 3:
        bytes_ = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = (
            bytes_[:, 0].astype(np.int32)
            | (bytes_[:, 1].astype(np.int32) << 8)
            | (bytes_[:, 2].astype(np.int32) << 16)
        )
        values = np.where(values & 0x800000, values - 0x1000000, values)
        audio = values.astype(np.float64) / float(2**23)
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64)
        audio = data / float(2**31)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    return sr, audio.reshape(-1, channels)


def write_wav(path: str | Path, sr: int, audio: np.ndarray, sample_width: int = 3) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(audio, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    y = np.clip(y, -1.0, 1.0)

    if sample_width == 2:
        values = np.round(y * (2**15 - 1)).astype("<i2")
        raw = values.tobytes()
    elif sample_width == 3:
        ints = np.round(y * (2**23 - 1)).astype(np.int32).reshape(-1)
        values = np.where(ints < 0, ints + 0x1000000, ints).astype(np.uint32)
        bytes_ = np.empty((values.size, 3), dtype=np.uint8)
        bytes_[:, 0] = values & 0xFF
        bytes_[:, 1] = (values >> 8) & 0xFF
        bytes_[:, 2] = (values >> 16) & 0xFF
        raw = bytes_.tobytes()
    elif sample_width == 4:
        values = np.round(y * (2**31 - 1)).astype("<i4")
        raw = values.tobytes()
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(y.shape[1])
        wav.setsampwidth(sample_width)
        wav.setframerate(sr)
        wav.writeframes(raw)
