from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


SHIFT_ORDER = [3, 7, 12, -12]


RISK_COLORS = {
    "low": (48, 130, 162),
    "medium": (224, 153, 61),
    "high": (178, 74, 88),
}


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _text_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return (255, 255, 255) if np.mean(color) < 135 else (20, 20, 20)


def save_preflight_risk_plot(predictions: pd.DataFrame, summary: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    signal_order = list(dict.fromkeys(predictions["signal_label"].tolist()))
    font = _font(11)
    small_font = _font(10)
    title_font = _font(15)
    cell_w = 138
    cell_h = 44
    left_margin = 220
    summary_w = 315
    margin = 28
    top_margin = 76
    width = margin * 2 + left_margin + len(SHIFT_ORDER) * cell_w + summary_w
    height = margin * 2 + top_margin + len(signal_order) * cell_h + 120
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw.text((margin, 18), "Exp10 Preflight Failure Prediction", fill=(15, 15, 15), font=title_font)
    draw.text((margin, 42), "Source-only risk before pitch shifting.", fill=(60, 60, 60), font=font)

    for col_index, shift in enumerate(SHIFT_ORDER):
        x = margin + left_margin + col_index * cell_w
        draw.text((x + 10, margin + top_margin - 28), f"{shift:+d} semitones", fill=(35, 35, 35), font=font)

    summary_x = margin + left_margin + len(SHIFT_ORDER) * cell_w + 24
    draw.text((summary_x, margin + top_margin - 28), "Signal Summary", fill=(35, 35, 35), font=font)

    for row_index, signal_label in enumerate(signal_order):
        y = margin + top_margin + row_index * cell_h
        row = predictions[predictions["signal_label"] == signal_label]
        draw.text((margin + 8, y + 14), signal_label, fill=(35, 35, 35), font=font)
        for col_index, shift in enumerate(SHIFT_ORDER):
            x = margin + left_margin + col_index * cell_w
            cell = row[row["shift_semitones"] == shift].iloc[0]
            risk = str(cell["preflight_risk_level"])
            color = RISK_COLORS.get(risk, (210, 210, 210))
            draw.rectangle([x, y, x + cell_w - 2, y + cell_h - 2], fill=color, outline=(255, 255, 255))
            text_color = _text_color(color)
            actual = "cat" if bool(cell["catastrophic_any_fixed"]) else "ok"
            predicted = "flag" if bool(cell["preflight_high_risk"]) else "pass"
            draw.text((x + 8, y + 7), f"{risk} {predicted}", fill=text_color, font=small_font)
            draw.text((x + 8, y + 22), f"{actual} | {float(cell['preflight_risk_score']):.2f}", fill=text_color, font=small_font)

        signal_summary = summary[summary["signal_label"] == signal_label]
        if not signal_summary.empty:
            info = signal_summary.iloc[0]
            summary_text = (
                f"cat {int(info['actual_catastrophic_cases'])} | "
                f"flag {int(info['predicted_high_risk_cases'])} | "
                f"recall {float(info['signal_recall']):.2f}"
            )
            draw.text((summary_x, y + 14), summary_text, fill=(35, 35, 35), font=small_font)

    legend_y = margin + top_margin + len(signal_order) * cell_h + 30
    draw.text((margin, legend_y), "Legend", fill=(35, 35, 35), font=font)
    for index, (risk, color) in enumerate(RISK_COLORS.items()):
        x = margin + index * 180
        y = legend_y + 24
        draw.rectangle([x, y, x + 14, y + 14], fill=color, outline=(90, 90, 90))
        draw.text((x + 22, y - 1), risk, fill=(35, 35, 35), font=small_font)
    draw.text((margin + 560, legend_y + 23), "cat = any fixed algorithm crossed threshold", fill=(35, 35, 35), font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
