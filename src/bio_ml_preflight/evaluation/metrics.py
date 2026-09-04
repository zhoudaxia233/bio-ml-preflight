from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    ndcg_score,
    roc_auc_score,
)


def metric_higher_is_better(metric: str) -> bool:
    return metric not in {"mae", "rmse", "log_loss"}


def _correlation(
    function: Callable[..., Any],
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
) -> float:
    if len(y_true) < 2 or np.ptp(y_true) <= 1e-12 or np.ptp(y_pred) <= 1e-12:
        return float("nan")
    return float(function(y_true, y_pred).statistic)


def _pearson(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
    if len(y_true) < 2 or np.ptp(y_true) <= 1e-12 or np.ptp(y_pred) <= 1e-12:
        return float("nan")
    left = y_true - np.mean(y_true)
    right = y_pred - np.mean(y_pred)
    left /= np.max(np.abs(left))
    right /= np.max(np.abs(right))
    denominator = np.sqrt(np.sum(left * left) * np.sum(right * right))
    return float(np.sum(left * right) / denominator)


def compute_metrics(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    task_kind: str,
    *,
    classification_threshold: float = 0.5,
) -> dict[str, float | None]:
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[finite], y_pred[finite]
    if not len(y_true):
        return {"applicable": None}
    if task_kind == "binary_classification":
        predicted_class = (y_pred >= classification_threshold).astype(int)
        both_classes = len(np.unique(y_true)) == 2
        return {
            "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted_class)),
            "roc_auc": float(roc_auc_score(y_true, y_pred)) if both_classes else None,
            "average_precision": float(average_precision_score(y_true, y_pred))
            if both_classes
            else None,
            "log_loss": float(log_loss(y_true, np.clip(y_pred, 1e-7, 1 - 1e-7))),
        }
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "pearson": _pearson(y_true, y_pred),
        "spearman": _correlation(spearmanr, y_true, y_pred),
        "kendall": _correlation(kendalltau, y_true, y_pred),
    }


def empirical_permutation_summary(
    observed: float, null_values: Iterable[float], *, higher_is_better: bool = True
) -> dict[str, float | int | None]:
    null = np.asarray(list(null_values), dtype=float)
    null = null[np.isfinite(null)]
    if not np.isfinite(observed) or not len(null):
        return {"draws": int(len(null)), "median": None, "q05": None, "q95": None, "p_value": None}
    at_least_as_good = null >= observed if higher_is_better else null <= observed
    return {
        "draws": int(len(null)),
        "median": float(np.median(null)),
        "q05": float(np.quantile(null, 0.05)),
        "q95": float(np.quantile(null, 0.95)),
        "p_value": float((1 + np.count_nonzero(at_least_as_good)) / (len(null) + 1)),
    }


def per_group_ranking_metrics(
    predictions: pd.DataFrame,
    *,
    group_column: str,
    truth_column: str = "y_true",
    prediction_column: str = "y_pred",
    k: int = 5,
) -> dict[str, float | None]:
    correlations, ndcgs = [], []
    for _, group in predictions.groupby(group_column):
        if len(group) < 2:
            continue
        correlations.append(
            _correlation(
                spearmanr, group[truth_column].to_numpy(), group[prediction_column].to_numpy()
            )
        )
        relevance = group[truth_column].to_numpy(dtype=float)
        relevance = relevance - np.nanmin(relevance)
        if np.any(relevance > 0):
            ndcgs.append(
                float(ndcg_score([relevance], [group[prediction_column]], k=min(k, len(group))))
            )
    finite_correlations = [value for value in correlations if np.isfinite(value)]
    return {
        "per_group_spearman_median": (
            float(np.median(finite_correlations)) if finite_correlations else None
        ),
        "ndcg_at_k_median": float(np.nanmedian(ndcgs)) if ndcgs else None,
        "groups_evaluated": float(len(correlations)),
    }


def group_respecting_permutation(
    values: npt.NDArray[Any], groups: npt.NDArray[Any] | None, seed: int
) -> npt.NDArray[Any]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    if groups is None:
        return rng.permutation(values)
    groups = np.asarray(groups)
    output = values.copy()
    grouped_positions = [np.flatnonzero(groups == group) for group in pd.unique(groups)]
    by_size: dict[int, list[npt.NDArray[np.int64]]] = {}
    for positions in grouped_positions:
        by_size.setdefault(len(positions), []).append(positions)
    for blocks in by_size.values():
        sources = rng.permutation(len(blocks))
        if len(blocks) == 1:
            output[blocks[0]] = rng.permutation(values[blocks[0]])
        else:
            for destination, source in zip(blocks, sources, strict=True):
                output[destination] = rng.permutation(values[blocks[source]])
    return output
