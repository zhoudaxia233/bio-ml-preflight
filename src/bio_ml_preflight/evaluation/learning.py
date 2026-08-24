from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def independent_group_learning_curve(
    frame: pd.DataFrame,
    *,
    group_column: str,
    fractions: list[float],
    seed: int,
    evaluate: Callable[[pd.DataFrame], float],
) -> list[dict[str, float | int]]:
    """Evaluate increasing deterministic subsets of whole independent groups."""
    if group_column not in frame:
        raise ValueError(f"Learning-curve group {group_column!r} is missing")
    groups = frame[group_column].dropna().unique()
    if len(groups) < 3:
        raise ValueError("At least three independent groups are required for a learning curve")
    rng = np.random.default_rng(seed)
    order = rng.permutation(groups)
    points = []
    for fraction in sorted(set(fractions)):
        if not 0 < fraction <= 1:
            raise ValueError("Learning-curve fractions must be in (0, 1]")
        count = max(2, round(fraction * len(order)))
        subset = frame[frame[group_column].isin(order[:count])]
        points.append(
            {
                "group_count": int(count),
                "row_count": int(len(subset)),
                "metric": float(evaluate(subset)),
            }
        )
    return points
