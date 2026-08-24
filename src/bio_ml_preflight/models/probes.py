from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _preprocessor(features: pd.DataFrame, *, scale: bool) -> ColumnTransformer:
    numeric = list(features.select_dtypes(include=[np.number, "bool"]).columns)
    categorical = [column for column in features if column not in numeric]
    numeric_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False, max_categories=200
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def build_probe_suite(
    features: pd.DataFrame, task_kind: str, seed: int, budget: str
) -> dict[str, Pipeline]:
    if task_kind == "binary_classification":
        models: dict[str, Any] = {
            "dummy": DummyClassifier(strategy="prior"),
            "logistic": LogisticRegression(C=1.0, max_iter=1000, random_state=seed),
            "extra_trees": ExtraTreesClassifier(
                n_estimators=80, min_samples_leaf=3, random_state=seed
            ),
            "hist_gradient_boosting": HistGradientBoostingClassifier(
                max_iter=80, random_state=seed
            ),
            "nearest_neighbour": KNeighborsClassifier(n_neighbors=7),
        }
    else:
        models = {
            "dummy": DummyRegressor(strategy="mean"),
            "elastic_net": ElasticNet(alpha=0.01, l1_ratio=0.1, max_iter=5000, random_state=seed),
            "extra_trees": ExtraTreesRegressor(
                n_estimators=80 if budget == "smoke" else 200,
                min_samples_leaf=3,
                n_jobs=1,
                random_state=seed,
            ),
            "hist_gradient_boosting": __import__(
                "sklearn.ensemble", fromlist=["HistGradientBoostingRegressor"]
            ).HistGradientBoostingRegressor(max_iter=80, random_state=seed),
            "nearest_neighbour": KNeighborsRegressor(n_neighbors=7, weights="distance"),
        }
    if budget == "smoke":
        keep = {"dummy", "elastic_net", "logistic", "extra_trees"}
        models = {name: model for name, model in models.items() if name in keep}
    return {
        name: Pipeline(
            [
                (
                    "preprocess",
                    _preprocessor(
                        features,
                        scale=name in {"elastic_net", "logistic", "nearest_neighbour"},
                    ),
                ),
                ("model", model),
            ]
        )
        for name, model in models.items()
    }


def entity_mean_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target: str,
    left: str,
    right: str,
) -> dict[str, npt.NDArray[np.float64]]:
    global_mean = float(train[target].mean())
    left_means = train.groupby(left)[target].mean()
    right_means = train.groupby(right)[target].mean()
    left_prediction = test[left].map(left_means)
    right_prediction = test[right].map(right_means)
    additive = left_prediction + right_prediction - global_mean
    return {
        "global_mean": np.full(len(test), global_mean),
        "seen_left_mean": left_prediction.to_numpy(dtype=float),
        "seen_right_mean": right_prediction.to_numpy(dtype=float),
        "additive_entity_mean": additive.to_numpy(dtype=float),
    }
