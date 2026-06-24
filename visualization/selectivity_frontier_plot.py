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


def _color_for_failures(failures: int) -> tuple[int, int, int]:
    if failures == 0:
        return (48, 130, 162)
    if failures <= 4:
        return (222, 151, 53)
    return (178, 74, 88)


def _scale(value: float, vmin: float, vmax: float, out_min: float, out_max: float) -> float:
    if vmax <= vmin:
        return (out_min + out_max) / 2.0
    ratio = (value - vmin) / (vmax - vmin)
    return out_min + ratio * (out_max - out_min)


def save_selectivity_frontier_plot(
    policies: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    font = _font(11)
    small_font = _font(10)
    title_font = _font(15)
    width = 1120
    height = 720
    margin = 48
    plot_x0 = 92
    plot_y0 = 92
    plot_w = 720
    plot_h = 480
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw.text((margin, 22), "Exp14 Selectivity Recovery Frontier", fill=(15, 15, 15), font=title_font)
    draw.text((margin, 46), "Safety vs. Phase Vocoder reliance across source-only selector policies.", fill=(60, 60, 60), font=font)

    x_values = policies["phase_vocoder_count"].to_numpy(dtype=np.float64)
    y_values = policies["mean_selected_stress"].to_numpy(dtype=np.float64)
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
    y_pad = max((y_max - y_min) * 0.08, 0.01)
    y_min -= y_pad
    y_max += y_pad

    draw.rectangle([plot_x0, plot_y0, plot_x0 + plot_w, plot_y0 + plot_h], outline=(180, 180, 180), width=1)
    for tick in np.linspace(x_min, x_max, 5):
        x = _scale(tick, x_min, x_max, plot_x0, plot_x0 + plot_w)
        draw.line([x, plot_y0 + plot_h, x, plot_y0 + plot_h + 5], fill=(80, 80, 80))
        draw.text((x - 12, plot_y0 + plot_h + 10), f"{tick:.0f}", fill=(50, 50, 50), font=small_font)
    for tick in np.linspace(y_min, y_max, 5):
        y = _scale(tick, y_min, y_max, plot_y0 + plot_h, plot_y0)
        draw.line([plot_x0 - 5, y, plot_x0, y], fill=(80, 80, 80))
        draw.text((plot_x0 - 48, y - 6), f"{tick:.2f}", fill=(50, 50, 50), font=small_font)

    draw.text((plot_x0 + plot_w // 2 - 95, plot_y0 + plot_h + 34), "Phase Vocoder selections", fill=(35, 35, 35), font=font)
    draw.text((20, plot_y0 + plot_h // 2 - 10), "Mean stress", fill=(35, 35, 35), font=font)

    for _, row in policies.iterrows():
        x = _scale(float(row["phase_vocoder_count"]), x_min, x_max, plot_x0, plot_x0 + plot_w)
        y = _scale(float(row["mean_selected_stress"]), y_min, y_max, plot_y0 + plot_h, plot_y0)
        color = _color_for_failures(int(row["adaptive_catastrophic_cases"]))
        radius = 3 if int(row["adaptive_catastrophic_cases"]) else 4
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color, outline=(255, 255, 255))

    if not summary.empty:
        best_id = str(summary.iloc[0]["most_selective_safe_policy_id"])
        best = policies[policies["policy_id"] == best_id]
        if not best.empty:
            row = best.iloc[0]
            x = _scale(float(row["phase_vocoder_count"]), x_min, x_max, plot_x0, plot_x0 + plot_w)
            y = _scale(float(row["mean_selected_stress"]), y_min, y_max, plot_y0 + plot_h, plot_y0)
            draw.ellipse([x - 8, y - 8, x + 8, y + 8], outline=(10, 10, 10), width=2)
            draw.text((x + 10, y - 10), "most selective safe", fill=(15, 15, 15), font=small_font)

    info_x = plot_x0 + plot_w + 36
    draw.text((info_x, plot_y0), "Policy Summary", fill=(35, 35, 35), font=font)
    if not summary.empty:
        row = summary.iloc[0]
        lines = [
            f"policies: {int(row['policy_count'])}",
            f"zero-failure: {int(row['zero_catastrophic_policy_count'])}",
            f"most selective safe: {row['most_selective_safe_policy_id']}",
            f"PV count: {int(row['most_selective_phase_vocoder_count'])}",
            f"mean stress: {float(row['most_selective_mean_stress']):.3f}",
            f"lowest mean safe: {row['lowest_mean_safe_policy_id']}",
            f"lowest mean stress: {float(row['lowest_mean_stress']):.3f}",
        ]
        for index, line in enumerate(lines):
            draw.text((info_x, plot_y0 + 28 + index * 22), line, fill=(35, 35, 35), font=small_font)

    legend_y = plot_y0 + plot_h + 76
    legend = [
        ("0 failures", (48, 130, 162)),
        ("1-4 failures", (222, 151, 53)),
        ("5+ failures", (178, 74, 88)),
    ]
    for index, (label, color) in enumerate(legend):
        x = margin + index * 170
        draw.ellipse([x, legend_y, x + 14, legend_y + 14], fill=color)
        draw.text((x + 22, legend_y - 1), label, fill=(35, 35, 35), font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
