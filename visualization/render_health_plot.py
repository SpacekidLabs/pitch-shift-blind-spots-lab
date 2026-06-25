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


def _color(flagged: bool, listener_silence: bool) -> tuple[int, int, int]:
    if flagged and listener_silence:
        return (188, 75, 88)
    if flagged:
        return (218, 151, 55)
    if listener_silence:
        return (120, 91, 165)
    return (47, 132, 143)


def save_render_health_plot(results: pd.DataFrame, summary: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    frame = results.sort_values(["silence_like_health_flag", "listener_silence_flag", "relative_rms_db"], ascending=[False, False, True])
    font = _font(11)
    small_font = _font(10)
    title_font = _font(16)
    margin = 30
    top = 92
    row_h = 36
    label_w = 150
    bar_w = 300
    side_w = 430
    width = margin * 2 + label_w + bar_w + side_w
    height = margin * 2 + top + len(frame) * row_h + 100
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw.text((margin, 18), "Exp24 Render Health Gate", fill=(15, 15, 15), font=title_font)
    draw.text(
        (margin, 44),
        "Low relative RMS plus sparse active frames catches perceived silence failures.",
        fill=(60, 60, 60),
        font=font,
    )

    for index, (_, row) in enumerate(frame.iterrows()):
        y = top + index * row_h
        color = _color(bool(row["silence_like_health_flag"]), bool(row["listener_silence_flag"]))
        x = margin
        draw.text((x, y + 9), str(row["blind_id"]), fill=(35, 35, 35), font=small_font)
        draw.text((x + 66, y + 9), str(row["strategy_label"]), fill=(35, 35, 35), font=small_font)
        bar_x = margin + label_w
        db = float(row["relative_rms_db"])
        normalized = np.clip((db + 40.0) / 40.0, 0.0, 1.0)
        fill_w = int(round(normalized * bar_w))
        draw.rectangle([bar_x, y + 8, bar_x + bar_w, y + 24], fill=(244, 244, 244), outline=(215, 215, 215))
        draw.rectangle([bar_x, y + 8, bar_x + fill_w, y + 24], fill=color)
        info_x = bar_x + bar_w + 24
        label = "flag" if bool(row["silence_like_health_flag"]) else "ok"
        draw.text(
            (info_x, y + 3),
            f"{label} | rms {db:.1f} dB | active {float(row['active_fraction']):.2f}",
            fill=(35, 35, 35),
            font=small_font,
        )
        draw.text(
            (info_x, y + 19),
            f"listener overall {float(row['overall_rating']):.0f} | silence {bool(row['listener_silence_flag'])}",
            fill=(70, 70, 70),
            font=small_font,
        )

    legend_y = top + len(frame) * row_h + 26
    if not summary.empty:
        row = summary.iloc[0]
        text = (
            f"precision {float(row['silence_precision']):.2f} | recall {float(row['silence_recall']):.2f} | "
            f"false positives {int(row['false_positive_count'])} | false negatives {int(row['false_negative_count'])}"
        )
        draw.text((margin, legend_y), text, fill=(45, 45, 45), font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
