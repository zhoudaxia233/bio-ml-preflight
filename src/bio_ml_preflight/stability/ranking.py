from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def ranking_stability(
    predictions: pd.DataFrame,
    *,
    group_column: str,
    candidate_column: str,
    k_values: list[int],
    higher_is_better: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    required = {"run_id", "y_pred", group_column, candidate_column}
    if predictions.empty or not required.issubset(predictions.columns):
        return {
            "status": "NOT_ASSESSABLE",
            "reason": "ranking columns are unavailable",
        }, pd.DataFrame()
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    run_ids = sorted(predictions["run_id"].unique())
    for k in k_values:
        overlaps, memberships = [], []
        top_sets: dict[tuple[str, str], set[str]] = {}
        for run_id, run in predictions.groupby("run_id"):
            for group, values in run.groupby(group_column):
                ranked = values.sort_values("y_pred", ascending=not higher_is_better).head(k)
                top_sets[(str(run_id), str(group))] = set(ranked[candidate_column].astype(str))
        groups = sorted({group for _, group in top_sets})
        for group in groups:
            available = [
                (run, top_sets[(run, group)]) for run in run_ids if (run, group) in top_sets
            ]
            for (_, left), (_, right) in combinations(available, 2):
                overlaps.append(jaccard(left, right))
            candidates = (
                sorted(set().union(*(values for _, values in available))) if available else []
            )
            for candidate in candidates:
                probability = np.mean([candidate in values for _, values in available])
                memberships.append(probability)
                rows.append(
                    {
                        "group": group,
                        "candidate": candidate,
                        "k": k,
                        "selection_probability": float(probability),
                        "runs": len(available),
                    }
                )
        summaries[str(k)] = {
            "average_pairwise_jaccard": float(np.mean(overlaps)) if overlaps else None,
            "unstable_candidate_fraction": (
                float(
                    np.mean(
                        [(probability > 0) and (probability < 1) for probability in memberships]
                    )
                )
                if memberships
                else None
            ),
            "comparisons": len(overlaps),
        }
    rank_frame = predictions.copy()
    rank_frame["rank"] = rank_frame.groupby(["run_id", group_column])["y_pred"].rank(
        ascending=not higher_is_better, method="average"
    )
    rank_std = rank_frame.groupby([group_column, candidate_column])["rank"].std().dropna()
    return {
        "status": "ASSESSED"
        if any(value["comparisons"] for value in summaries.values())
        else "NOT_ASSESSABLE",
        "top_k": summaries,
        "rank_standard_deviation_median": float(rank_std.median()) if len(rank_std) else None,
        "scope": "Development predictions; confirmatory holdout labels were not accessed.",
    }, pd.DataFrame(rows)


def stability_decomposition(experiments: pd.DataFrame, primary_metric: str) -> dict[str, Any]:
    usable = experiments[
        experiments["model"].ne("dummy")
        & experiments["permuted"].eq(False)
        & experiments[primary_metric].notna()
    ]
    if usable.empty:
        return {
            key: {"status": "NOT_ASSESSABLE", "reason": "no finite experiment metrics"}
            for key in [
                "training_initialization",
                "train_validation_split",
                "model_family",
                "preprocessing",
                "molecular_representation",
                "deployment_scenario",
            ]
        }
    representation_columns = ["representation"] if "representation" in usable else []
    split_std = (
        usable.groupby(["scenario", *representation_columns, "model"])[primary_metric]
        .std()
        .dropna()
    )
    model_std = (
        usable.groupby(["scenario", *representation_columns, "seed"])[primary_metric].std().dropna()
    )
    representation_std = (
        usable.groupby(["scenario", "seed", "model"])[primary_metric].std().dropna()
        if representation_columns
        else pd.Series(dtype=float)
    )
    scenario_medians = usable.groupby("scenario")[primary_metric].median()
    return {
        "training_initialization": {
            "status": "NOT_ASSESSABLE",
            "reason": "v0.1 smoke runs do not cross training seeds with fixed split manifests",
        },
        "train_validation_split": {
            "status": "ASSESSED",
            "median_standard_deviation": float(split_std.median()) if len(split_std) else None,
        },
        "model_family": {
            "status": "ASSESSED",
            "median_standard_deviation": float(model_std.median()) if len(model_std) else None,
        },
        "preprocessing": {
            "status": "NOT_ASSESSABLE",
            "reason": "one train-fold-only preprocessing policy was fixed per representation",
        },
        "molecular_representation": {
            "status": "ASSESSED" if len(representation_std) else "NOT_ASSESSABLE",
            "median_standard_deviation": (
                float(representation_std.median()) if len(representation_std) else None
            ),
        },
        "deployment_scenario": {
            "status": "ASSESSED" if len(scenario_medians) > 1 else "NOT_ASSESSABLE",
            "standard_deviation": float(scenario_medians.std())
            if len(scenario_medians) > 1
            else None,
        },
    }
