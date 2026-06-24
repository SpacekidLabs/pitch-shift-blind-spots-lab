from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ALGORITHM_COLORS = {
    "Input": (70, 70, 70),
    "Phase Vocoder": (27, 109, 146),
    "WSOLA": (75, 166, 124),
    "Rubber Band": (222, 145, 46),
    "PSOLA": (175, 72, 88),
}


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _scale(value: float, min_value: float, max_value: float, low: int, high: int) -> float:
    if max_value <= min_value:
        return (low + high) / 2.0
    ratio = (value - min_value) / (max_value - min_value)
    return low + ratio * (high - low)


def _trajectory_points(frame: pd.DataFrame, y_column: str, left: int, right: int, top: int, bottom: int, ymin: float, ymax: float) -> list[tuple[float, float]]:
    clean = frame.dropna(subset=[y_column])
    if clean.empty:
        return []
    tmin = float(clean["time_seconds"].min())
    tmax = float(clean["time_seconds"].max())
    points = []
    for _, row in clean.iterrows():
        x = _scale(float(row["time_seconds"]), tmin, tmax, left, right)
        y = _scale(float(row[y_column]), ymin, ymax, bottom, top)
        points.append((x, y))
    return points


def save_observer_state_plot(trajectories: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    case_order = list(dict.fromkeys(trajectories["case_label"].tolist()))
    algorithm_order = ["Input", "Phase Vocoder", "PSOLA", "WSOLA", "Rubber Band"]

    font = _font(12)
    small_font = _font(11)
    title_font = _font(15)
    panel_w = 760
    panel_h = 300
    margin = 28
    image = Image.new("RGB", (panel_w + 2 * margin, len(case_order) * panel_h + (len(case_order) + 1) * margin + 42), "white")
    draw = ImageDraw.Draw(image)

    draw.text((margin, 16), "Experiment 06 - Observer State Space", fill=(15, 15, 15), font=title_font)
    draw.text(
        (margin, 36),
        "Framewise inferred input pitch f0(t), estimated from each observer output at -12 semitones",
        fill=(60, 60, 60),
        font=font,
    )

    for case_index, case_label in enumerate(case_order):
        x0 = margin
        y0 = margin + 42 + case_index * (panel_h + margin)
        left = x0 + 64
        right = x0 + panel_w - 22
        top = y0 + 42
        bottom = y0 + panel_h - 54
        case_frame = trajectories[trajectories["case_label"] == case_label]
        values = case_frame["inferred_input_f0_hz"].dropna().to_numpy(dtype=np.float64)
        ymin = max(40.0, float(np.percentile(values, 5)) - 20.0) if values.size else 100.0
        ymax = float(np.percentile(values, 95)) + 20.0 if values.size else 600.0
        if ymax <= ymin:
            ymax = ymin + 1.0

        draw.rectangle([x0, y0, x0 + panel_w, y0 + panel_h], outline=(45, 45, 45), width=1)
        draw.text((x0 + 10, y0 + 10), case_label, fill=(18, 18, 18), font=font)
        draw.line([left, bottom, right, bottom], fill=(45, 45, 45), width=1)
        draw.line([left, top, left, bottom], fill=(45, 45, 45), width=1)
        draw.text((left - 50, top - 6), f"{ymax:.0f}", fill=(70, 70, 70), font=small_font)
        draw.text((left - 50, bottom - 8), f"{ymin:.0f}", fill=(70, 70, 70), font=small_font)
        draw.text((left + 220, bottom + 24), "time seconds", fill=(70, 70, 70), font=small_font)
        draw.text((x0 + 10, top + 72), "inferred pitch Hz", fill=(70, 70, 70), font=small_font)

        for algorithm in algorithm_order:
            algo_frame = case_frame[case_frame["algorithm_label"] == algorithm]
            if algo_frame.empty:
                continue
            points = _trajectory_points(algo_frame, "inferred_input_f0_hz", left, right, top, bottom, ymin, ymax)
            if len(points) > 1:
                draw.line(points, fill=ALGORITHM_COLORS.get(algorithm, (30, 30, 30)), width=3 if algorithm != "Input" else 2)
            for x, y in points[:: max(1, len(points) // 12)]:
                radius = 3 if algorithm != "Input" else 2
                color = ALGORITHM_COLORS.get(algorithm, (30, 30, 30))
                draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)

        legend_x = x0 + panel_w - 270
        legend_y = y0 + 12
        for legend_index, algorithm in enumerate(algorithm_order):
            y = legend_y + legend_index * 18
            color = ALGORITHM_COLORS.get(algorithm, (30, 30, 30))
            draw.line([legend_x, y + 7, legend_x + 26, y + 7], fill=color, width=3 if algorithm != "Input" else 2)
            draw.text((legend_x + 34, y), algorithm, fill=(35, 35, 35), font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

