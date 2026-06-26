from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _bar_color(score: float) -> tuple[int, int, int]:
    ratio = min(max((score - 1.0) / 4.0, 0.0), 1.0)
    low = (184, 75, 89)
    high = (44, 128, 139)
    return tuple(int(round((1.0 - ratio) * low[i] + ratio * high[i])) for i in range(3))


def _draw_score_bar(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    score: float,
    label: str,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    draw.text((x, y), label[:30], fill=(35, 35, 35), font=font)
    bar_x = x + 240
    fill_w = int(round(((score - 1.0) / 4.0) * width))
    draw.rectangle([bar_x, y - 2, bar_x + width, y + 17], fill=(244, 241, 235), outline=(218, 211, 199))
    draw.rectangle([bar_x, y - 2, bar_x + fill_w, y + 17], fill=_bar_color(score))
    draw.text((bar_x + width + 18, y), f"{score:.2f}", fill=(45, 45, 45), font=small_font)


def save_adaptive_v4_score_plot(
    policy_summary: pd.DataFrame,
    shift_summary: pd.DataFrame,
    duplicate_summary: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    policy_frame = policy_summary.sort_values("mean_overall_rating", ascending=True).reset_index(drop=True)
    shift_frame = shift_summary.sort_values("shift_semitones").reset_index(drop=True)

    font = _font(12)
    small_font = _font(10)
    title_font = _font(18)
    margin = 32
    width = 1040
    height = 760
    image = Image.new("RGB", (width, height), (255, 252, 246))
    draw = ImageDraw.Draw(image)

    draw.text((margin, 24), "Exp27 Adaptive v4 Blind Score Analysis", fill=(18, 22, 22), font=title_font)
    draw.text(
        (margin, 54),
        "Decoded listener ratings for Experiment 26. Higher overall score is better.",
        fill=(83, 89, 86),
        font=font,
    )

    draw.text((margin, 100), "Policy means", fill=(18, 22, 22), font=font)
    y = 134
    for _, row in policy_frame.iterrows():
        _draw_score_bar(
            draw,
            margin,
            y,
            330,
            float(row["mean_overall_rating"]),
            str(row["render_label"]),
            font,
            small_font,
        )
        draw.text(
            (margin + 660, y),
            f"artifact {float(row['mean_artifact_rating']):.2f} | transient {float(row['mean_transient_rating']):.2f} | groove {float(row['mean_groove_rating']):.2f}",
            fill=(83, 89, 86),
            font=small_font,
        )
        y += 42

    draw.text((margin, 376), "Shift means", fill=(18, 22, 22), font=font)
    y = 410
    for _, row in shift_frame.iterrows():
        label = f"{int(row['shift_semitones']):+d} semitones"
        _draw_score_bar(draw, margin, y, 330, float(row["mean_overall_rating"]), label, font, small_font)
        draw.text(
            (margin + 660, y),
            f"stress {float(row['mean_composite_stress']):.3f} | range {float(row['overall_range']):.1f}",
            fill=(83, 89, 86),
            font=small_font,
        )
        y += 42

    draw.text((margin, 570), "Equivalent-render spread", fill=(18, 22, 22), font=font)
    if duplicate_summary.empty:
        draw.text((margin, 604), "No duplicate-equivalent render groups found.", fill=(83, 89, 86), font=small_font)
    else:
        y = 604
        for _, row in duplicate_summary.sort_values("overall_range", ascending=False).iterrows():
            draw.text(
                (margin, y),
                f"{int(row['shift_semitones']):+d}: overall {row['overall_scores']} | range {float(row['overall_range']):.1f}",
                fill=(45, 45, 45),
                font=small_font,
            )
            y += 24

    draw.text(
        (margin, height - 54),
        "Caution: all Experiment 26 final renders resolved to Phase Vocoder, so policy differences are preference/context effects.",
        fill=(110, 88, 62),
        font=small_font,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
