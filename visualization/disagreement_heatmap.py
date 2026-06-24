from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


SHIFT_ORDER = [3, 7, 12, -12]


def _value_to_color(value: float, min_value: float, max_value: float) -> tuple[int, int, int]:
    if max_value <= min_value:
        ratio = 0.5
    else:
        ratio = (value - min_value) / (max_value - min_value)
    ratio = float(np.clip(ratio, 0.0, 1.0))
    stops = [
        (0.0, np.array([18, 27, 60])),
        (0.35, np.array([29, 107, 142])),
        (0.7, np.array([74, 166, 124])),
        (1.0, np.array([249, 214, 75])),
    ]
    for (left_pos, left_color), (right_pos, right_color) in zip(stops[:-1], stops[1:]):
        if ratio <= right_pos:
            local = (ratio - left_pos) / max(right_pos - left_pos, 1e-9)
            color = (1.0 - local) * left_color + local * right_color
            return tuple(int(round(channel)) for channel in color)
    return tuple(int(round(channel)) for channel in stops[-1][1])


def _min_max(values: pd.Series) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.empty:
        return 0.0, 1.0
    return float(finite.min()), float(finite.max())


def save_disagreement_heatmap(landscape: pd.DataFrame, summary: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    signal_order = summary.sort_values("mean_algorithm_disagreement", ascending=False)["signal_label"].tolist()
    landscape = landscape.copy()
    landscape["signal_label"] = pd.Categorical(landscape["signal_label"], categories=signal_order, ordered=True)
    landscape = landscape.sort_values(["signal_label", "shift_semitones"]).reset_index(drop=True)

    font = ImageFont.load_default()
    title_font = ImageFont.load_default()
    cell_w = 128
    cell_h = 36
    left_margin = 220
    top_margin = 58
    summary_w = 270
    margin = 26
    rows = len(signal_order)
    cols = len(SHIFT_ORDER)
    heatmap_w = left_margin + cols * cell_w + summary_w + 20
    heatmap_h = top_margin + rows * cell_h + 36
    image = Image.new("RGB", (heatmap_w + 2 * margin, heatmap_h + 2 * margin), color="white")
    draw = ImageDraw.Draw(image)
    x0 = margin
    y0 = margin

    vmin, vmax = _min_max(landscape["algorithm_disagreement"])
    draw.text((x0, y0), "Experiment 02 - Algorithm Disagreement Landscape", fill=(15, 15, 15), font=title_font)
    draw.text(
        (x0, y0 + 20),
        "algorithm_disagreement = variance(composite_stress across algorithms)",
        fill=(60, 60, 60),
        font=font,
    )

    for col_index, shift in enumerate(SHIFT_ORDER):
        tx = x0 + left_margin + col_index * cell_w + cell_w // 2
        draw.text((tx - 12, y0 + top_margin - 22), f"{shift:+d}", fill=(40, 40, 40), font=font)

    summary_x = x0 + left_margin + cols * cell_w + 24
    draw.text((summary_x, y0 + top_margin - 22), "Signal Summary", fill=(40, 40, 40), font=font)

    for row_index, signal_label in enumerate(signal_order):
        y = y0 + top_margin + row_index * cell_h
        row_frame = landscape[landscape["signal_label"] == signal_label]
        summary_row = summary[summary["signal_label"] == signal_label].iloc[0]
        draw.text((x0 + 8, y + 10), signal_label, fill=(35, 35, 35), font=font)

        for col_index, shift in enumerate(SHIFT_ORDER):
            cell_x0 = x0 + left_margin + col_index * cell_w
            cell_y0 = y
            value = float(row_frame.loc[row_frame["shift_semitones"] == shift, "algorithm_disagreement"].iloc[0])
            color = _value_to_color(value, vmin, vmax)
            draw.rectangle([cell_x0, cell_y0, cell_x0 + cell_w - 1, cell_y0 + cell_h - 1], fill=color, outline=(255, 255, 255))
            text_color = (255, 255, 255) if np.mean(color) < 120 else (15, 15, 15)
            draw.text((cell_x0 + 12, cell_y0 + 11), f"{value:.4f}", fill=text_color, font=font)

        summary_text = (
            f"{summary_row['disagreement_level']} | "
            f"mean {summary_row['mean_algorithm_disagreement']:.4f} | "
            f"peak {int(summary_row['most_disagreeing_shift']):+d}"
        )
        draw.text((summary_x, y + 10), summary_text, fill=(35, 35, 35), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

