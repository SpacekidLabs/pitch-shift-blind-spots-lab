from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


SHIFT_ORDER = [3, 7, 12, -12]
METRICS = [
    ("RMS Error", "rms_error"),
    ("Spectral Distance", "spectral_distance"),
    ("Centroid Difference", "spectral_centroid_difference"),
    ("Composite Stress", "composite_stress"),
]


def _value_to_color(value: float, min_value: float, max_value: float) -> tuple[int, int, int]:
    if max_value <= min_value:
        ratio = 0.5
    else:
        ratio = (value - min_value) / (max_value - min_value)
    ratio = float(np.clip(ratio, 0.0, 1.0))
    stops = [
        (0.0, np.array([16, 24, 56])),
        (0.35, np.array([28, 111, 145])),
        (0.7, np.array([75, 169, 125])),
        (1.0, np.array([252, 214, 79])),
    ]
    for (left_pos, left_color), (right_pos, right_color) in zip(stops[:-1], stops[1:]):
        if ratio <= right_pos:
            local = (ratio - left_pos) / max(right_pos - left_pos, 1e-9)
            color = (1.0 - local) * left_color + local * right_color
            return tuple(int(round(channel)) for channel in color)
    return tuple(int(round(channel)) for channel in stops[-1][1])


def _panel_min_max(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.min(finite)), float(np.max(finite))


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    title: str,
    frame: pd.DataFrame,
    value_col: str,
    row_labels: list[str],
) -> tuple[int, int]:
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()
    cell_w = 120
    cell_h = 34
    left_margin = 190
    top_margin = 34
    rows = len(row_labels)
    cols = len(SHIFT_ORDER)
    panel_w = left_margin + cols * cell_w + 16
    panel_h = top_margin + rows * cell_h + 20
    values = frame[value_col].to_numpy(dtype=np.float64)
    vmin, vmax = _panel_min_max(values)

    draw.rectangle([x0, y0, x0 + panel_w, y0 + panel_h], outline=(45, 45, 45), width=1)
    draw.text((x0 + 10, y0 + 8), title, fill=(15, 15, 15), font=title_font)

    for col_index, shift in enumerate(SHIFT_ORDER):
        label = f"{shift:+d}"
        tx = x0 + left_margin + col_index * cell_w + cell_w // 2
        draw.text((tx - 12, y0 + 16), label, fill=(40, 40, 40), font=font)

    for row_index, row_label in enumerate(row_labels):
        y = y0 + top_margin + row_index * cell_h
        draw.text((x0 + 10, y + 9), row_label, fill=(40, 40, 40), font=font)
        row_frame = frame[frame["signal_label"] == row_label]
        for col_index, shift in enumerate(SHIFT_ORDER):
            cell_x0 = x0 + left_margin + col_index * cell_w
            cell_y0 = y0 + top_margin + row_index * cell_h
            value = float(row_frame.loc[row_frame["shift_semitones"] == shift, value_col].iloc[0])
            color = _value_to_color(value, vmin, vmax)
            draw.rectangle([cell_x0, cell_y0, cell_x0 + cell_w - 1, cell_y0 + cell_h - 1], fill=color, outline=(255, 255, 255))
            text = f"{value:.2f}"
            text_color = (255, 255, 255) if np.mean(color) < 120 else (15, 15, 15)
            draw.text((cell_x0 + 12, cell_y0 + 10), text, fill=text_color, font=font)

    return panel_w, panel_h


def save_algorithm_comparison_heatmap(results: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    algorithm_order = list(dict.fromkeys(results["algorithm_label"].tolist()))
    signal_order = list(dict.fromkeys(results["signal_label"].tolist()))

    ordered = results.copy()
    ordered["algorithm_label"] = pd.Categorical(ordered["algorithm_label"], categories=algorithm_order, ordered=True)
    ordered["signal_label"] = pd.Categorical(ordered["signal_label"], categories=signal_order, ordered=True)
    ordered = ordered.sort_values(["algorithm_label", "signal_label", "shift_semitones"]).reset_index(drop=True)

    normalized = ordered.copy()
    normalized["composite_stress"] = normalized[
        ["rms_error", "spectral_distance", "spectral_centroid_difference"]
    ].mean(axis=1)

    cell_w = 120
    cell_h = 34
    left_margin = 190
    top_margin = 34
    panel_w = left_margin + len(SHIFT_ORDER) * cell_w + 16
    panel_h = top_margin + len(signal_order) * cell_h + 20
    margin = 24
    canvas_w = len(METRICS) * panel_w + (len(METRICS) + 1) * margin
    canvas_h = len(algorithm_order) * panel_h + (len(algorithm_order) + 1) * margin
    image = Image.new("RGB", (canvas_w, canvas_h), color="white")
    draw = ImageDraw.Draw(image)
    header_font = ImageFont.load_default()

    for row_index, algorithm_label in enumerate(algorithm_order):
        row_df = normalized[normalized["algorithm_label"] == algorithm_label]
        y0 = margin + row_index * (panel_h + margin)
        draw.text((10, y0 + 8), algorithm_label, fill=(20, 20, 20), font=header_font)
        for col_index, (metric_title, metric_column) in enumerate(METRICS):
            x0 = margin + col_index * (panel_w + margin)
            _draw_panel(draw, x0, y0, metric_title, row_df, metric_column, signal_order)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
