from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _color(score: float) -> tuple[int, int, int]:
    ratio = min(max((score - 1.0) / 4.0, 0.0), 1.0)
    low = np.array([188, 75, 88])
    high = np.array([45, 132, 143])
    mixed = (1.0 - ratio) * low + ratio * high
    return tuple(int(round(value)) for value in mixed)


def save_listening_score_plot(strategy_summary: pd.DataFrame, correlation_summary: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    frame = strategy_summary.sort_values("mean_overall_rating", ascending=True).reset_index(drop=True)
    font = _font(11)
    small_font = _font(10)
    title_font = _font(16)
    margin = 30
    label_w = 160
    bar_w = 340
    side_w = 420
    row_h = 58
    top = 88
    width = margin * 2 + label_w + bar_w + side_w
    height = margin * 2 + top + len(frame) * row_h + 92
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw.text((margin, 18), "Exp23 Listening Score Analysis", fill=(15, 15, 15), font=title_font)
    draw.text(
        (margin, 44),
        "Human drum-loop ratings decoded by strategy. Higher overall score is better.",
        fill=(60, 60, 60),
        font=font,
    )

    for index, row in frame.iterrows():
        y = top + index * row_h
        score = float(row["mean_overall_rating"])
        width_px = int(round(((score - 1.0) / 4.0) * bar_w))
        color = _color(score)
        draw.text((margin, y + 19), str(row["strategy_label"]), fill=(35, 35, 35), font=font)
        x = margin + label_w
        draw.rectangle([x, y + 13, x + bar_w, y + 38], fill=(244, 244, 244), outline=(215, 215, 215))
        draw.rectangle([x, y + 13, x + width_px, y + 38], fill=color)
        info_x = x + bar_w + 24
        draw.text(
            (info_x, y + 6),
            f"overall {score:.2f} | artifact {float(row['mean_artifact_rating']):.2f}",
            fill=(35, 35, 35),
            font=small_font,
        )
        draw.text(
            (info_x, y + 27),
            f"transient {float(row['mean_transient_rating']):.2f} | groove {float(row['mean_groove_rating']):.2f}",
            fill=(70, 70, 70),
            font=small_font,
        )

    corr_y = top + len(frame) * row_h + 24
    draw.text((margin, corr_y), "Metric Agreement", fill=(35, 35, 35), font=font)
    if not correlation_summary.empty:
        lines = []
        for _, row in correlation_summary.iterrows():
            lines.append(f"{row['metric']}: r={float(row['pearson_r_with_overall']):.2f}")
        draw.text((margin + 150, corr_y), " | ".join(lines), fill=(70, 70, 70), font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
