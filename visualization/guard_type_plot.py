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


def _color(catastrophic_count: int, retention: float) -> tuple[int, int, int]:
    if catastrophic_count > 0:
        return (190, 78, 88)
    if retention < 0.75:
        return (218, 151, 55)
    return (46, 132, 143)


def save_guard_type_plot(summary: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    frame = summary.sort_values("guard_type_score", ascending=True).reset_index(drop=True)
    font = _font(11)
    small_font = _font(10)
    title_font = _font(16)
    margin = 30
    label_w = 235
    bar_w = 360
    side_w = 385
    row_h = 56
    top = 88
    width = margin * 2 + label_w + bar_w + side_w
    height = margin * 2 + top + len(frame) * row_h + 72
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw.text((margin, 18), "Exp21 Guard Type Comparison", fill=(15, 15, 15), font=title_font)
    draw.text(
        (margin, 44),
        "Different safety actions on the same guarded fallback cases.",
        fill=(60, 60, 60),
        font=font,
    )

    max_score = max(float(frame["guard_type_score"].max()), 1e-9)
    min_score = float(frame["guard_type_score"].min())
    span = max(max_score - min_score, 1e-9)
    for index, row in frame.iterrows():
        y = top + index * row_h
        score = float(row["guard_type_score"])
        normalized = (score - min_score) / span
        fill_w = int(round(normalized * bar_w))
        color = _color(int(row["catastrophic_count"]), float(row["mean_shift_retention"]))
        draw.text((margin, y + 18), str(row["guard_label"]), fill=(35, 35, 35), font=font)
        x = margin + label_w
        draw.rectangle([x, y + 11, x + bar_w, y + 36], fill=(245, 245, 245), outline=(215, 215, 215))
        draw.rectangle([x, y + 11, x + fill_w, y + 36], fill=color)
        info_x = x + bar_w + 26
        draw.text(
            (info_x, y + 6),
            f"score {score:.3f} | cat {int(row['catastrophic_count'])} | stress {float(row['mean_stress']):.3f}",
            fill=(35, 35, 35),
            font=small_font,
        )
        draw.text(
            (info_x, y + 27),
            f"retention {float(row['mean_shift_retention']):.3f} | source sim {float(row['mean_dry_similarity']):.3f}",
            fill=(70, 70, 70),
            font=small_font,
        )

    legend_y = top + len(frame) * row_h + 24
    draw.text(
        (margin, legend_y),
        "Blue: safe with high retention. Orange: safe but shift-suppressing. Red: misses catastrophe.",
        fill=(45, 45, 45),
        font=small_font,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
