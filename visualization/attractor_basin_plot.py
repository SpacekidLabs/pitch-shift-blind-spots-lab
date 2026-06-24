from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ALGORITHM_COLORS = {
    "Phase Vocoder": (27, 109, 146),
    "PSOLA": (175, 72, 88),
    "WSOLA": (75, 166, 124),
    "Rubber Band": (222, 145, 46),
}

BRANCH_COLORS = {
    "lower_branch": (27, 109, 146),
    "middle_branch": (130, 130, 130),
    "upper_branch": (175, 72, 88),
    "mixed_branch": (222, 145, 46),
    "untracked": (180, 180, 180),
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


def save_attractor_basin_plot(branch_map: pd.DataFrame, state_disagreement: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    font = _font(12)
    small_font = _font(11)
    title_font = _font(15)
    margin = 30
    plot_w = 900
    plot_h = 560
    image = Image.new("RGB", (plot_w + 2 * margin, plot_h + 2 * margin + 48), "white")
    draw = ImageDraw.Draw(image)

    draw.text((margin, 16), "Experiment 07 - Attractor Basin Mapping", fill=(15, 15, 15), font=title_font)
    draw.text((margin, 38), "Dominant inferred pitch branch for 440 + delta, shifted -12 semitones", fill=(60, 60, 60), font=font)

    left = margin + 70
    right = margin + plot_w - 30
    top = margin + 70
    bottom = margin + plot_h - 45
    xmin = float(branch_map["delta_hz"].min())
    xmax = float(branch_map["delta_hz"].max())
    ymin = 390.0
    ymax = 930.0

    draw.rectangle([left, top, right, bottom], outline=(45, 45, 45), width=1)
    for y_value, label in [(440.0, "lower ~440"), (660.0, "middle"), (880.0, "upper ~880")]:
        y = _scale(y_value, ymin, ymax, bottom, top)
        draw.line([left, y, right, y], fill=(220, 220, 220), width=1)
        draw.text((left + 8, y - 14), label, fill=(90, 90, 90), font=small_font)

    for tick in range(0, 6):
        x = _scale(float(tick), xmin, xmax, left, right)
        draw.line([x, bottom, x, bottom + 5], fill=(45, 45, 45), width=1)
        draw.text((x - 4, bottom + 10), str(tick), fill=(70, 70, 70), font=small_font)
    for y_tick in [440, 660, 880]:
        y = _scale(float(y_tick), ymin, ymax, bottom, top)
        draw.line([left - 5, y, left, y], fill=(45, 45, 45), width=1)
        draw.text((left - 46, y - 7), str(y_tick), fill=(70, 70, 70), font=small_font)

    algorithm_order = ["Phase Vocoder", "PSOLA", "WSOLA", "Rubber Band"]
    for algorithm in algorithm_order:
        frame = branch_map[branch_map["algorithm_label"] == algorithm].sort_values("delta_hz")
        points = []
        for _, row in frame.iterrows():
            if not np.isfinite(row["dominant_inferred_pitch_hz"]):
                continue
            x = _scale(float(row["delta_hz"]), xmin, xmax, left, right)
            y = _scale(float(row["dominant_inferred_pitch_hz"]), ymin, ymax, bottom, top)
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=ALGORITHM_COLORS.get(algorithm, (30, 30, 30)), width=3)
        for _, row in frame.iterrows():
            if not np.isfinite(row["dominant_inferred_pitch_hz"]):
                continue
            x = _scale(float(row["delta_hz"]), xmin, xmax, left, right)
            y = _scale(float(row["dominant_inferred_pitch_hz"]), ymin, ymax, bottom, top)
            color = BRANCH_COLORS.get(str(row["branch"]), (30, 30, 30))
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color, outline=(255, 255, 255))

    disagreement = state_disagreement.sort_values("delta_hz")
    if not disagreement.empty:
        max_std = float(disagreement["state_std_hz"].max())
        baseline_y = bottom - 12
        for _, row in disagreement.iterrows():
            x = _scale(float(row["delta_hz"]), xmin, xmax, left, right)
            height = _scale(float(row["state_std_hz"]), 0.0, max(max_std, 1e-9), 0, 52)
            draw.line([x, baseline_y, x, baseline_y - height], fill=(190, 190, 190), width=2)
        draw.text((right - 170, baseline_y - 60), "state std bars", fill=(90, 90, 90), font=small_font)

    draw.text((left + 300, bottom + 30), "delta Hz", fill=(70, 70, 70), font=small_font)
    draw.text((margin + 8, top + 180), "dominant inferred pitch Hz", fill=(70, 70, 70), font=small_font)

    legend_x = right - 250
    legend_y = top + 18
    for index, algorithm in enumerate(algorithm_order):
        y = legend_y + index * 18
        color = ALGORITHM_COLORS.get(algorithm, (30, 30, 30))
        draw.line([legend_x, y + 7, legend_x + 28, y + 7], fill=color, width=3)
        draw.text((legend_x + 36, y), algorithm, fill=(35, 35, 35), font=small_font)

    branch_x = legend_x
    branch_y = legend_y + 94
    for index, branch in enumerate(["lower_branch", "middle_branch", "upper_branch", "mixed_branch"]):
        y = branch_y + index * 18
        color = BRANCH_COLORS[branch]
        draw.ellipse([branch_x, y + 3, branch_x + 10, y + 13], fill=color)
        draw.text((branch_x + 18, y), branch, fill=(35, 35, 35), font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

