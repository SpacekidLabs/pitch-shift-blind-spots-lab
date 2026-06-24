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
        (0.35, np.array([31, 111, 145])),
        (0.7, np.array([78, 170, 124])),
        (1.0, np.array([250, 216, 79])),
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


def save_discovery_heatmap(
    landscape: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: str | Path,
    max_rows: int = 24,
) -> None:
    output_path = Path(output_path)
    shown = summary.sort_values("max_algorithm_disagreement", ascending=False).head(max_rows)
    candidate_order = shown["candidate_id"].tolist()
    landscape = landscape[landscape["candidate_id"].isin(candidate_order)].copy()
    landscape["candidate_id"] = pd.Categorical(landscape["candidate_id"], categories=candidate_order, ordered=True)
    landscape = landscape.sort_values(["candidate_id", "shift_semitones"]).reset_index(drop=True)

    font = ImageFont.load_default()
    title_font = ImageFont.load_default()
    cell_w = 122
    cell_h = 36
    left_margin = 280
    top_margin = 60
    summary_w = 360
    margin = 26
    rows = len(candidate_order)
    cols = len(SHIFT_ORDER)
    heatmap_w = left_margin + cols * cell_w + summary_w + 20
    heatmap_h = top_margin + rows * cell_h + 42
    image = Image.new("RGB", (heatmap_w + 2 * margin, heatmap_h + 2 * margin), color="white")
    draw = ImageDraw.Draw(image)
    x0 = margin
    y0 = margin

    vmin, vmax = _min_max(landscape["algorithm_disagreement"])
    draw.text((x0, y0), "Experiment 03 - Blind Spot Discovery", fill=(15, 15, 15), font=title_font)
    draw.text(
        (x0, y0 + 20),
        "Search objective: maximize variance(composite_stress across algorithms)",
        fill=(60, 60, 60),
        font=font,
    )

    for col_index, shift in enumerate(SHIFT_ORDER):
        tx = x0 + left_margin + col_index * cell_w + cell_w // 2
        draw.text((tx - 12, y0 + top_margin - 22), f"{shift:+d}", fill=(40, 40, 40), font=font)

    summary_x = x0 + left_margin + cols * cell_w + 24
    draw.text((summary_x, y0 + top_margin - 22), "Discovery Summary", fill=(40, 40, 40), font=font)

    for row_index, candidate_id in enumerate(candidate_order):
        y = y0 + top_margin + row_index * cell_h
        row_frame = landscape[landscape["candidate_id"] == candidate_id]
        summary_row = shown[shown["candidate_id"] == candidate_id].iloc[0]
        label = f"{int(summary_row['discovery_rank']):02d}. {summary_row['candidate_family']} / {candidate_id[-2:]}"
        draw.text((x0 + 8, y + 10), label, fill=(35, 35, 35), font=font)

        for col_index, shift in enumerate(SHIFT_ORDER):
            cell_x0 = x0 + left_margin + col_index * cell_w
            cell_y0 = y
            value = float(row_frame.loc[row_frame["shift_semitones"] == shift, "algorithm_disagreement"].iloc[0])
            color = _value_to_color(value, vmin, vmax)
            draw.rectangle([cell_x0, cell_y0, cell_x0 + cell_w - 1, cell_y0 + cell_h - 1], fill=color, outline=(255, 255, 255))
            text_color = (255, 255, 255) if np.mean(color) < 120 else (15, 15, 15)
            draw.text((cell_x0 + 12, cell_y0 + 11), f"{value:.4f}", fill=text_color, font=font)

        summary_text = (
            f"best {summary_row['max_algorithm_disagreement']:.4f} "
            f"at {int(summary_row['best_shift_semitones']):+d} | "
            f"mean {summary_row['mean_algorithm_disagreement']:.4f}"
        )
        draw.text((summary_x, y + 10), summary_text, fill=(35, 35, 35), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

