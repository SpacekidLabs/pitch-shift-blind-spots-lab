from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


SHIFT_ORDER = [3, 7, 12, -12]


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _color(value: float, threshold: float) -> tuple[int, int, int]:
    if value >= threshold:
        return (178, 74, 88)
    ratio = min(max(value / max(threshold, 1e-9), 0.0), 1.0)
    low = np.array([48, 130, 162])
    high = np.array([222, 151, 53])
    mixed = (1.0 - ratio) * low + ratio * high
    return tuple(int(round(channel)) for channel in mixed)


def save_noise_guard_plot(cases: pd.DataFrame, summary: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    guard_order = sorted(cases["dry_mix"].unique())
    signal_order = list(dict.fromkeys(cases["signal_label"].tolist()))
    font = _font(11)
    small_font = _font(10)
    title_font = _font(15)
    cell_w = 94
    cell_h = 42
    left_margin = 195
    margin = 28
    top_margin = 78
    rows = len(signal_order) * len(SHIFT_ORDER)
    width = margin * 2 + left_margin + len(guard_order) * cell_w + 260
    height = margin * 2 + top_margin + rows * cell_h + 110
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw.text((margin, 18), "Exp17 Noise Fallback Guard", fill=(15, 15, 15), font=title_font)
    draw.text((margin, 42), "Dry/wet guard sweep for Phase Vocoder fallback on noise-like signals.", fill=(60, 60, 60), font=font)

    for col_index, dry_mix in enumerate(guard_order):
        x = margin + left_margin + col_index * cell_w
        draw.text((x + 10, margin + top_margin - 28), f"dry {dry_mix:.2f}", fill=(35, 35, 35), font=small_font)

    threshold = float(cases["catastrophic_threshold"].iloc[0])
    row_index = 0
    for signal_label in signal_order:
        for shift in SHIFT_ORDER:
            y = margin + top_margin + row_index * cell_h
            row = cases[(cases["signal_label"] == signal_label) & (cases["shift_semitones"] == shift)]
            draw.text((margin + 8, y + 13), f"{signal_label} {shift:+d}", fill=(35, 35, 35), font=small_font)
            for col_index, dry_mix in enumerate(guard_order):
                x = margin + left_margin + col_index * cell_w
                cell = row[row["dry_mix"] == dry_mix].iloc[0]
                stress = float(cell["composite_stress"])
                color = _color(stress, threshold=threshold)
                draw.rectangle([x, y, x + cell_w - 2, y + cell_h - 2], fill=color, outline=(255, 255, 255))
                text_color = (255, 255, 255) if np.mean(color) < 130 else (20, 20, 20)
                label = "cat" if bool(cell["catastrophic"]) else "ok"
                draw.text((x + 8, y + 8), label, fill=text_color, font=small_font)
                draw.text((x + 8, y + 23), f"{stress:.3f}", fill=text_color, font=small_font)
            row_index += 1

    info_x = margin + left_margin + len(guard_order) * cell_w + 28
    draw.text((info_x, margin + top_margin), "Summary", fill=(35, 35, 35), font=font)
    if not summary.empty:
        info = summary.iloc[0]
        lines = [
            f"best dry mix: {float(info['best_dry_mix']):.2f}",
            f"cat count: {int(info['best_catastrophic_count'])}",
            f"mean stress: {float(info['best_mean_stress']):.3f}",
            f"threshold: {float(info['catastrophic_threshold']):.3f}",
        ]
        for index, line in enumerate(lines):
            draw.text((info_x, margin + top_margin + 28 + index * 22), line, fill=(35, 35, 35), font=small_font)

    legend_y = margin + top_margin + rows * cell_h + 30
    draw.text((margin, legend_y), "Red cells cross the catastrophic threshold.", fill=(35, 35, 35), font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
