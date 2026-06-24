from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


SHIFT_ORDER = [3, 7, 12, -12]


STATE_COLORS = {
    "stable_periodic": (39, 125, 161),
    "octave_ambiguous": (180, 74, 90),
    "subharmonic_ambiguous": (78, 164, 120),
    "detuned_competing": (218, 143, 46),
    "noise_like": (114, 114, 114),
    "transient_like": (113, 91, 171),
    "modulated_pitch": (57, 143, 126),
    "untracked": (205, 205, 205),
}


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _text_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return (255, 255, 255) if np.mean(color) < 130 else (20, 20, 20)


def save_adaptive_selector_plot(
    decisions: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    signal_order = list(dict.fromkeys(decisions["signal_label"].tolist()))
    font = _font(11)
    small_font = _font(10)
    title_font = _font(15)
    cell_w = 155
    cell_h = 48
    left_margin = 220
    summary_w = 320
    margin = 28
    top_margin = 76
    width = margin * 2 + left_margin + len(SHIFT_ORDER) * cell_w + summary_w
    height = margin * 2 + top_margin + len(signal_order) * cell_h + 148
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw.text((margin, 18), "Exp09 Adaptive Selector Prototype", fill=(15, 15, 15), font=title_font)
    draw.text((margin, 42), "State, selected algorithm, and stress.", fill=(60, 60, 60), font=font)

    for col_index, shift in enumerate(SHIFT_ORDER):
        x = margin + left_margin + col_index * cell_w
        draw.text((x + 10, margin + top_margin - 28), f"{shift:+d} semitones", fill=(35, 35, 35), font=font)

    summary_x = margin + left_margin + len(SHIFT_ORDER) * cell_w + 24
    draw.text((summary_x, margin + top_margin - 28), "Signal Summary", fill=(35, 35, 35), font=font)

    for row_index, signal_label in enumerate(signal_order):
        y = margin + top_margin + row_index * cell_h
        row = decisions[decisions["signal_label"] == signal_label]
        draw.text((margin + 8, y + 16), signal_label, fill=(35, 35, 35), font=font)
        for col_index, shift in enumerate(SHIFT_ORDER):
            x = margin + left_margin + col_index * cell_w
            cell = row[row["shift_semitones"] == shift].iloc[0]
            state = str(cell["adaptive_state"])
            color = STATE_COLORS.get(state, (205, 205, 205))
            draw.rectangle([x, y, x + cell_w - 2, y + cell_h - 2], fill=color, outline=(255, 255, 255))
            text_color = _text_color(color)
            algorithm = str(cell["selected_algorithm_label"])
            stress = float(cell["composite_stress"])
            safe = "safe" if bool(cell["safe_mode"]) else "normal"
            draw.text((x + 8, y + 7), state, fill=text_color, font=small_font)
            draw.text((x + 8, y + 22), f"{algorithm} | {safe}", fill=text_color, font=small_font)
            draw.text((x + 8, y + 35), f"stress {stress:.3f}", fill=text_color, font=small_font)

        signal_summary = summary[summary["signal_label"] == signal_label]
        if not signal_summary.empty:
            info = signal_summary.iloc[0]
            summary_text = (
                f"mean {float(info['adaptive_mean_stress']):.3f} | "
                f"cat {int(info['adaptive_catastrophic_count'])} | "
                f"best fixed {info['best_fixed_algorithm_label']}"
            )
            draw.text((summary_x, y + 16), summary_text, fill=(35, 35, 35), font=small_font)

    legend_y = margin + top_margin + len(signal_order) * cell_h + 30
    draw.text((margin, legend_y), "State Legend", fill=(35, 35, 35), font=font)
    for index, (state, color) in enumerate(STATE_COLORS.items()):
        x = margin + (index % 4) * 230
        y = legend_y + 24 + (index // 4) * 22
        draw.rectangle([x, y, x + 14, y + 14], fill=color, outline=(90, 90, 90))
        draw.text((x + 22, y - 1), state, fill=(35, 35, 35), font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
