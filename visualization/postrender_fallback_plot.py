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


def _color(health_flags: int, fallback_rate: float) -> tuple[int, int, int]:
    if health_flags > 0:
        return (188, 75, 88)
    if fallback_rate > 0:
        return (45, 132, 143)
    return (78, 145, 105)


def save_postrender_fallback_plot(summary: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    frame = summary.sort_values(["health_flag_count", "fallback_used_count", "mean_composite_stress"], ascending=[False, True, False])
    font = _font(11)
    small_font = _font(10)
    title_font = _font(16)
    margin = 30
    top = 90
    label_w = 240
    bar_w = 320
    side_w = 380
    row_h = 58
    width = margin * 2 + label_w + bar_w + side_w
    height = margin * 2 + top + len(frame) * row_h + 76
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw.text((margin, 18), "Exp25 Post-Render Fallback Selector", fill=(15, 15, 15), font=title_font)
    draw.text(
        (margin, 44),
        "Health-gated adaptive renders reject collapsed candidates before delivery.",
        fill=(60, 60, 60),
        font=font,
    )

    max_stress = max(float(frame["mean_composite_stress"].max()), 1e-9)
    for index, (_, row) in enumerate(frame.iterrows()):
        y = top + index * row_h
        label = str(row["render_label"])
        stress = float(row["mean_composite_stress"])
        health_flags = int(row["health_flag_count"])
        fallback_count = int(row["fallback_used_count"])
        fallback_rate = fallback_count / max(int(row["case_count"]), 1)
        color = _color(health_flags, fallback_rate)
        draw.text((margin, y + 19), label, fill=(35, 35, 35), font=font)
        x = margin + label_w
        fill_w = int(round((stress / max_stress) * bar_w))
        draw.rectangle([x, y + 12, x + bar_w, y + 37], fill=(244, 244, 244), outline=(215, 215, 215))
        draw.rectangle([x, y + 12, x + fill_w, y + 37], fill=color)
        info_x = x + bar_w + 24
        draw.text(
            (info_x, y + 6),
            f"health flags {health_flags} | fallbacks {fallback_count}",
            fill=(35, 35, 35),
            font=small_font,
        )
        draw.text(
            (info_x, y + 27),
            f"stress {stress:.3f} | active {float(row['mean_active_fraction']):.2f}",
            fill=(70, 70, 70),
            font=small_font,
        )

    legend_y = top + len(frame) * row_h + 24
    draw.text(
        (margin, legend_y),
        "Red: collapsed output survived. Blue: fallback rescued a candidate. Green: healthy without fallback.",
        fill=(45, 45, 45),
        font=small_font,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
