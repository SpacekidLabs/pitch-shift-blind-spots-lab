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


def save_listening_test_plot(metrics: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    frame = metrics.sort_values(["shift_semitones", "blind_id"]).reset_index(drop=True)
    font = _font(11)
    small_font = _font(10)
    title_font = _font(16)
    margin = 30
    row_h = 34
    top = 92
    label_w = 120
    bar_w = 260
    width = 980
    height = margin * 2 + top + len(frame) * row_h + 74
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw.text((margin, 18), "Exp22 Drum Listening Test", fill=(15, 15, 15), font=title_font)
    draw.text(
        (margin, 44),
        "Blinded render pack: lower metric bars are not automatically better, but they flag likely artifacts.",
        fill=(60, 60, 60),
        font=font,
    )

    max_stress = max(float(frame["composite_stress"].max()), 1e-9)
    for index, row in frame.iterrows():
        y = top + index * row_h
        x = margin
        draw.text((x, y + 8), str(row["blind_id"]), fill=(35, 35, 35), font=small_font)
        draw.text((x + 70, y + 8), f"{int(row['shift_semitones']):+d}", fill=(35, 35, 35), font=small_font)
        bar_x = x + label_w
        stress = float(row["composite_stress"])
        fill_w = int(round((stress / max_stress) * bar_w))
        color = (48, 130, 162) if stress < 0.35 * max_stress else (218, 151, 53)
        if stress > 0.70 * max_stress:
            color = (184, 75, 89)
        draw.rectangle([bar_x, y + 7, bar_x + bar_w, y + 23], fill=(244, 244, 244), outline=(215, 215, 215))
        draw.rectangle([bar_x, y + 7, bar_x + fill_w, y + 23], fill=color)
        draw.text((bar_x + bar_w + 20, y + 7), f"stress {stress:.3f}", fill=(45, 45, 45), font=small_font)
        draw.text((bar_x + bar_w + 118, y + 7), f"centroid {float(row['spectral_centroid_difference']):.0f}", fill=(75, 75, 75), font=small_font)
        draw.text((bar_x + bar_w + 240, y + 7), str(row["wav_file"]), fill=(75, 75, 75), font=small_font)

    legend_y = top + len(frame) * row_h + 26
    draw.text((margin, legend_y), "Use the listening sheet for scores; the answer key maps blind IDs to algorithms.", fill=(45, 45, 45), font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
