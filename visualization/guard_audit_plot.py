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


def _color(value: float) -> tuple[int, int, int]:
    value = min(max(value, 0.0), 1.0)
    low = np.array([41, 110, 137])
    high = np.array([199, 91, 72])
    mixed = (1.0 - value) * low + value * high
    return tuple(int(round(channel)) for channel in mixed)


def save_guard_audit_plot(results: pd.DataFrame, summary: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    dry_mixes = sorted(results["dry_mix"].unique())
    signals = list(dict.fromkeys(results["case_label"].tolist()))
    font = _font(11)
    small_font = _font(10)
    title_font = _font(16)
    cell_w = 84
    cell_h = 42
    left_margin = 210
    margin = 28
    top_margin = 88
    side_w = 300
    rows = len(signals)
    width = margin * 2 + left_margin + len(dry_mixes) * cell_w + side_w
    height = margin * 2 + top_margin + rows * cell_h + 110
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw.text((margin, 18), "Exp19 Guard Audit", fill=(15, 15, 15), font=title_font)
    draw.text(
        (margin, 44),
        "Stress vs. shift-retention sweep for adaptive fallback guards.",
        fill=(60, 60, 60),
        font=font,
    )

    for col_index, dry_mix in enumerate(dry_mixes):
        x = margin + left_margin + col_index * cell_w
        draw.text((x + 9, margin + top_margin - 30), f"dry {dry_mix:.2f}", fill=(35, 35, 35), font=small_font)

    for row_index, case_label in enumerate(signals):
        y = margin + top_margin + row_index * cell_h
        frame = results[results["case_label"] == case_label]
        draw.text((margin + 8, y + 13), case_label, fill=(35, 35, 35), font=small_font)
        for col_index, dry_mix in enumerate(dry_mixes):
            x = margin + left_margin + col_index * cell_w
            cell = frame[frame["dry_mix"] == dry_mix].iloc[0]
            stress = float(cell["composite_stress"])
            retention = float(cell["shift_retention"])
            color = _color(stress)
            draw.rectangle([x, y, x + cell_w - 2, y + cell_h - 2], fill=color, outline=(255, 255, 255))
            text_color = (255, 255, 255) if np.mean(color) < 125 else (25, 25, 25)
            draw.text((x + 7, y + 7), f"s {stress:.2f}", fill=text_color, font=small_font)
            draw.text((x + 7, y + 23), f"r {retention:.2f}", fill=text_color, font=small_font)

    info_x = margin + left_margin + len(dry_mixes) * cell_w + 28
    draw.text((info_x, margin + top_margin), "Best Compromise", fill=(35, 35, 35), font=font)
    if not summary.empty:
        best = summary.iloc[0]
        lines = [
            f"dry mix: {float(best['dry_mix']):.2f}",
            f"cat cases: {int(best['catastrophic_count'])}",
            f"mean stress: {float(best['mean_stress']):.3f}",
            f"shift retention: {float(best['mean_shift_retention']):.3f}",
            f"guard score: {float(best['guard_audit_score']):.3f}",
        ]
        for index, line in enumerate(lines):
            draw.text((info_x, margin + top_margin + 30 + index * 23), line, fill=(35, 35, 35), font=small_font)

    legend_y = margin + top_margin + rows * cell_h + 28
    draw.text((margin, legend_y), "Each cell shows composite stress (s) and retained shift amount (r).", fill=(35, 35, 35), font=small_font)
    draw.text((margin, legend_y + 22), "Best compromise requires zero catastrophes, then balances low stress against retained shift.", fill=(35, 35, 35), font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
