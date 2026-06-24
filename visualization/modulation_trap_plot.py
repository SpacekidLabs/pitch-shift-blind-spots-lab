from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


OUTCOME_COLORS = {
    "safe": (45, 126, 161),
    "caught": (177, 73, 88),
    "missed": (225, 150, 48),
    "false_alarm": (125, 125, 125),
}


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _text_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return (255, 255, 255) if np.mean(color) < 135 else (20, 20, 20)


def save_modulation_trap_plot(
    cases: pd.DataFrame,
    output_path: str | Path,
    shift: int = 3,
    title: str = "Exp11 Subtle Modulation Trap",
    subtitle: str | None = None,
    miss_note: str = "missed = catastrophic fixed-algorithm stress not flagged by source-only preflight",
) -> None:
    output_path = Path(output_path)
    frame = cases[cases["shift_semitones"] == shift].copy()
    depths = sorted(frame["depth_semitones"].unique())
    rates = sorted(frame["rate_hz"].unique())
    font = _font(11)
    small_font = _font(10)
    title_font = _font(15)
    cell_w = 102
    cell_h = 52
    left_margin = 110
    margin = 28
    top_margin = 78
    width = margin * 2 + left_margin + len(depths) * cell_w
    height = margin * 2 + top_margin + len(rates) * cell_h + 125
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw.text((margin, 18), title, fill=(15, 15, 15), font=title_font)
    draw.text((margin, 42), subtitle or f"Vibrato depth/rate risk at {shift:+d} semitones.", fill=(60, 60, 60), font=font)

    for col_index, depth in enumerate(depths):
        x = margin + left_margin + col_index * cell_w
        draw.text((x + 10, margin + top_margin - 30), f"{depth:g} st", fill=(35, 35, 35), font=font)

    for row_index, rate in enumerate(rates):
        y = margin + top_margin + row_index * cell_h
        draw.text((margin + 8, y + 18), f"{rate:g} Hz", fill=(35, 35, 35), font=font)
        for col_index, depth in enumerate(depths):
            x = margin + left_margin + col_index * cell_w
            cell = frame[(frame["rate_hz"] == rate) & (frame["depth_semitones"] == depth)].iloc[0]
            outcome = str(cell["preflight_outcome"])
            color = OUTCOME_COLORS.get(outcome, (210, 210, 210))
            draw.rectangle([x, y, x + cell_w - 2, y + cell_h - 2], fill=color, outline=(255, 255, 255))
            text_color = _text_color(color)
            draw.text((x + 8, y + 8), outcome, fill=text_color, font=small_font)
            draw.text((x + 8, y + 24), f"risk {float(cell['preflight_risk_score']):.2f}", fill=text_color, font=small_font)
            draw.text((x + 8, y + 38), f"stress {float(cell['worst_fixed_composite_stress']):.2f}", fill=text_color, font=small_font)

    legend_y = margin + top_margin + len(rates) * cell_h + 30
    draw.text((margin, legend_y), "Outcome Legend", fill=(35, 35, 35), font=font)
    for index, (outcome, color) in enumerate(OUTCOME_COLORS.items()):
        x = margin + index * 170
        y = legend_y + 26
        draw.rectangle([x, y, x + 14, y + 14], fill=color, outline=(90, 90, 90))
        draw.text((x + 22, y - 1), outcome, fill=(35, 35, 35), font=small_font)
    draw.text((margin, legend_y + 58), miss_note, fill=(35, 35, 35), font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
