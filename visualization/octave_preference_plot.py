from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ALGORITHM_ORDER = ["Phase Vocoder", "PSOLA", "WSOLA", "Rubber Band"]
BELIEF_COLORS = {
    "subharmonic_seeking": (75, 166, 124),
    "fundamental_seeking": (27, 109, 146),
    "octave_seeking": (175, 72, 88),
    "upper_partial_seeking": (222, 145, 46),
    "mixed": (130, 130, 130),
    "untracked": (210, 210, 210),
}


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def save_octave_preference_map(beliefs: pd.DataFrame, algorithm_summary: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    case_order = list(dict.fromkeys(beliefs["case_label"].tolist()))
    font = _font(12)
    small_font = _font(11)
    title_font = _font(15)
    cell_w = 138
    cell_h = 36
    left_margin = 230
    top_margin = 72
    summary_w = 260
    margin = 28
    width = left_margin + len(ALGORITHM_ORDER) * cell_w + summary_w + 2 * margin
    height = top_margin + len(case_order) * cell_h + 300
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw.text((margin, 18), "Exp08 Octave Preference Map", fill=(15, 15, 15), font=title_font)
    draw.text((margin, 40), "Observer belief map for octave ambiguity.", fill=(60, 60, 60), font=font)

    x0 = margin
    y0 = margin + 36
    for col_index, algorithm in enumerate(ALGORITHM_ORDER):
        x = x0 + left_margin + col_index * cell_w
        draw.text((x + 8, y0 + 12), algorithm, fill=(35, 35, 35), font=small_font)

    summary_x = x0 + left_margin + len(ALGORITHM_ORDER) * cell_w + 20
    draw.text((summary_x, y0 + 12), "Dominant Prior", fill=(35, 35, 35), font=small_font)

    for row_index, case_label in enumerate(case_order):
        y = y0 + top_margin + row_index * cell_h
        row = beliefs[beliefs["case_label"] == case_label]
        draw.text((x0 + 8, y + 10), case_label, fill=(35, 35, 35), font=font)
        for col_index, algorithm in enumerate(ALGORITHM_ORDER):
            x = x0 + left_margin + col_index * cell_w
            cell = row[row["algorithm_label"] == algorithm]
            if cell.empty:
                belief = "untracked"
                pitch = float("nan")
            else:
                belief = str(cell.iloc[0]["observer_belief"])
                pitch = float(cell.iloc[0]["dominant_inferred_pitch_hz"])
            color = BELIEF_COLORS.get(belief, (210, 210, 210))
            draw.rectangle([x, y, x + cell_w - 2, y + cell_h - 2], fill=color, outline=(255, 255, 255))
            text = "nan" if pd.isna(pitch) else f"{pitch:.0f}"
            text_color = (255, 255, 255) if sum(color) / 3 < 120 else (20, 20, 20)
            draw.text((x + 12, y + 10), text, fill=text_color, font=small_font)

        counts = row["observer_belief"].value_counts()
        dominant = str(counts.idxmax()) if not counts.empty else "untracked"
        draw.text((summary_x, y + 10), dominant, fill=(35, 35, 35), font=small_font)

    legend_y = y0 + top_margin + len(case_order) * cell_h + 28
    draw.text((margin, legend_y), "Belief Legend", fill=(35, 35, 35), font=font)
    for index, (belief, color) in enumerate(BELIEF_COLORS.items()):
        x = margin + (index % 3) * 240
        y = legend_y + 24 + (index // 3) * 22
        draw.rectangle([x, y, x + 14, y + 14], fill=color, outline=(90, 90, 90))
        draw.text((x + 22, y - 1), belief, fill=(35, 35, 35), font=small_font)

    summary_y = legend_y + 78
    draw.text((margin, summary_y), "Algorithm Bias Summary", fill=(35, 35, 35), font=font)
    for row_index, row in algorithm_summary.iterrows():
        text = (
            f"{row['algorithm_label']}: {row['dominant_observer_bias']} "
            f"({int(row['dominant_count'])}/{int(row['case_count'])})"
        )
        draw.text((margin, summary_y + 24 + row_index * 18), text, fill=(35, 35, 35), font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
