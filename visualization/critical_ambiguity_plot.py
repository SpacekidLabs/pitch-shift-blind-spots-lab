from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PANEL_ORDER = ["critical_two_oscillators", "critical_vibrato_depth", "critical_beating_rate"]


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


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    frame: pd.DataFrame,
    peak_row: pd.Series,
    x0: int,
    y0: int,
    width: int,
    height: int,
    title: str,
    units: str,
    global_ymax: float,
) -> None:
    font = _font(12)
    small_font = _font(11)
    left = x0 + 58
    right = x0 + width - 22
    top = y0 + 38
    bottom = y0 + height - 82

    draw.rectangle([x0, y0, x0 + width, y0 + height], outline=(45, 45, 45), width=1)
    draw.text((x0 + 10, y0 + 10), title, fill=(18, 18, 18), font=font)
    draw.line([left, bottom, right, bottom], fill=(45, 45, 45), width=1)
    draw.line([left, top, left, bottom], fill=(45, 45, 45), width=1)

    x_values = frame["ambiguity_amount"].to_numpy(dtype=np.float64)
    max_values = frame["max_algorithm_disagreement"].to_numpy(dtype=np.float64)
    mean_values = frame["mean_algorithm_disagreement"].to_numpy(dtype=np.float64)
    xmin = float(np.min(x_values))
    xmax = float(np.max(x_values))
    ymax = max(global_ymax, 1e-9)

    max_points = []
    mean_points = []
    for x_value, max_value, mean_value in zip(x_values, max_values, mean_values):
        x = _scale(x_value, xmin, xmax, left, right)
        max_y = _scale(max_value, 0.0, ymax, bottom, top)
        mean_y = _scale(mean_value, 0.0, ymax, bottom, top)
        max_points.append((x, max_y))
        mean_points.append((x, mean_y))

    if len(mean_points) > 1:
        draw.line(mean_points, fill=(120, 120, 120), width=2)
    if len(max_points) > 1:
        draw.line(max_points, fill=(27, 109, 146), width=3)

    peak_x = _scale(float(peak_row["peak_disagreement_location"]), xmin, xmax, left, right)
    draw.line([peak_x, top, peak_x, bottom], fill=(210, 80, 70), width=1)

    for x, y in mean_points:
        draw.rectangle([x - 3, y - 3, x + 3, y + 3], fill=(235, 235, 235), outline=(90, 90, 90))
    for x, y in max_points:
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(249, 214, 75), outline=(18, 27, 60))

    draw.text((left - 44, top - 5), f"{ymax:.3f}", fill=(70, 70, 70), font=small_font)
    draw.text((left - 22, bottom - 8), "0", fill=(70, 70, 70), font=small_font)
    draw.text((left, bottom + 12), f"{xmin:g}", fill=(70, 70, 70), font=small_font)
    draw.text((right - 26, bottom + 12), f"{xmax:g}", fill=(70, 70, 70), font=small_font)
    draw.text((left + 76, bottom + 28), units, fill=(70, 70, 70), font=small_font)

    draw.text(
        (x0 + 10, y0 + height - 48),
        f"peak {peak_row['peak_disagreement']:.4f} at {peak_row['peak_disagreement_location']:g}",
        fill=(35, 35, 35),
        font=small_font,
    )
    draw.text(
        (x0 + 10, y0 + height - 30),
        f"width90 {peak_row['peak_width']:.3g}; half-max {peak_row['half_max_width']:.3g}",
        fill=(35, 35, 35),
        font=small_font,
    )
    draw.line([x0 + width - 150, y0 + height - 35, x0 + width - 120, y0 + height - 35], fill=(27, 109, 146), width=3)
    draw.text((x0 + width - 114, y0 + height - 41), "max", fill=(35, 35, 35), font=small_font)
    draw.line([x0 + width - 72, y0 + height - 35, x0 + width - 42, y0 + height - 35], fill=(120, 120, 120), width=2)
    draw.text((x0 + width - 36, y0 + height - 41), "mean", fill=(35, 35, 35), font=small_font)


def save_critical_ambiguity_plot(summary: pd.DataFrame, peak_metrics: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    ordered = summary.sort_values(["sweep_family", "ambiguity_amount"]).reset_index(drop=True)
    global_ymax = float(ordered["max_algorithm_disagreement"].max()) * 1.08

    panel_w = 540
    panel_h = 350
    margin = 28
    image = Image.new("RGB", (panel_w + 2 * margin, 3 * panel_h + 4 * margin + 42), color="white")
    draw = ImageDraw.Draw(image)
    title_font = _font(15)
    font = _font(12)
    draw.text((margin, 16), "Experiment 05 - Critical Ambiguity Zoom", fill=(15, 15, 15), font=title_font)
    draw.text(
        (margin, 36),
        "Do disagreement peaks concentrate near the boundary between one pitch and competing pitches?",
        fill=(60, 60, 60),
        font=font,
    )

    titles = {
        "critical_two_oscillators": "Two Oscillators: 440 + delta",
        "critical_vibrato_depth": "Vibrato Depth: 0-2 semitones",
        "critical_beating_rate": "Beat Rate: 0.1-5 Hz",
    }

    for index, family in enumerate(PANEL_ORDER):
        frame = ordered[ordered["sweep_family"] == family]
        if frame.empty:
            continue
        peak_row = peak_metrics[peak_metrics["sweep_family"] == family].iloc[0]
        x0 = margin
        y0 = margin + 42 + index * (panel_h + margin)
        units = str(frame["ambiguity_units"].iloc[0])
        _draw_panel(draw, frame, peak_row, x0, y0, panel_w, panel_h, titles.get(family, family), units, global_ymax)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
