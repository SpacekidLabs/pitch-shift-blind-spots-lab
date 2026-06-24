from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


METRIC_COLUMNS = ["rms_error", "spectral_distance", "spectral_centroid_difference"]


def add_composite_stress(results: pd.DataFrame, metric_columns: Sequence[str] = METRIC_COLUMNS) -> pd.DataFrame:
    scored = results.copy()
    normalized_columns = []
    for column in metric_columns:
        normalized = f"{column}_normalized"
        values = scored[column].to_numpy(dtype=np.float64)
        min_value = float(np.min(values))
        max_value = float(np.max(values))
        if max_value > min_value:
            scored[normalized] = (values - min_value) / (max_value - min_value)
        else:
            scored[normalized] = 0.0
        normalized_columns.append(normalized)

    scored["composite_stress"] = scored[normalized_columns].mean(axis=1)
    return scored


def build_disagreement_landscape(results: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    landscape = (
        results.groupby(list(group_columns), as_index=False)
        .agg(
            algorithm_disagreement=("composite_stress", lambda values: float(np.var(values, ddof=0))),
            mean_composite_stress=("composite_stress", "mean"),
            min_composite_stress=("composite_stress", "min"),
            max_composite_stress=("composite_stress", "max"),
        )
        .reset_index(drop=True)
    )
    landscape["algorithm_spread"] = landscape["max_composite_stress"] - landscape["min_composite_stress"]
    return landscape


def add_disagreement_levels(summary: pd.DataFrame, score_column: str = "mean_algorithm_disagreement") -> pd.DataFrame:
    ranked = summary.sort_values(score_column, ascending=False).reset_index(drop=True).copy()
    ranked["disagreement_rank"] = np.arange(1, len(ranked) + 1)
    if len(ranked) >= 3:
        ranked["disagreement_level"] = pd.qcut(
            ranked[score_column].rank(method="first"),
            q=3,
            labels=["Low", "Medium", "High"],
        ).astype(str)
    else:
        ranked["disagreement_level"] = "High"
    return ranked

