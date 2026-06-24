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


def _bar_color(catastrophic_count: int, unnecessary_rate: float) -> tuple[int, int, int]:
    if catastrophic_count > 0:
        return (189, 78, 86)
    if unnecessary_rate > 0.35:
        return (218, 151, 55)
    return (52, 133, 145)


def save_guard_specificity_plot(summary: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    frame = summary.sort_values("guard_specificity_score", ascending=True).reset_index(drop=True)
    font = _font(11)
    small_font = _font(10)
    title_font = _font(16)
    margin = 30
    label_w = 210
    bar_w = 360
    row_h = 54
    top = 86
    side_w = 360
    width = margin * 2 + label_w + bar_w + side_w
    height = margin * 2 + top + len(frame) * row_h + 70
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw.text((margin, 18), "Exp20 Guard Specificity Test", fill=(15, 15, 15), font=title_font)
    draw.text(
        (margin, 44),
        "Safety is useful only when the selector does not become afraid of everything.",
        fill=(60, 60, 60),
        font=font,
    )

    max_score = max(float(frame["guard_specificity_score"].max()), 1e-9)
    min_score = float(frame["guard_specificity_score"].min())
    score_span = max(max_score - min_score, 1e-9)
    for index, row in frame.iterrows():
        y = top + index * row_h
        score = float(row["guard_specificity_score"])
        normalized = (score - min_score) / score_span
        width_px = int(round(normalized * bar_w))
        color = _bar_color(int(row["catastrophic_count"]), float(row["unnecessary_guard_rate"]))
        draw.text((margin, y + 17), str(row["policy_label"]), fill=(35, 35, 35), font=font)
        x = margin + label_w
        draw.rectangle([x, y + 10, x + bar_w, y + 34], outline=(215, 215, 215), fill=(245, 245, 245))
        draw.rectangle([x, y + 10, x + width_px, y + 34], fill=color)
        info_x = x + bar_w + 26
        line = (
            f"score {score:.3f} | cat {int(row['catastrophic_count'])} | "
            f"guarded {int(row['guarded_count'])}"
        )
        draw.text((info_x, y + 7), line, fill=(35, 35, 35), font=small_font)
        line2 = (
            f"rescued {int(row['rescued_count'])} | unnecessary "
            f"{int(row['unnecessary_guard_count'])} | retention {float(row['mean_shift_retention']):.3f}"
        )
        draw.text((info_x, y + 27), line2, fill=(70, 70, 70), font=small_font)

    legend_y = top + len(frame) * row_h + 24
    draw.text((margin, legend_y), "Blue: specific safe policy. Orange: safe but over-guarded. Red: misses catastrophe.", fill=(45, 45, 45), font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
