from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from bio_ml_preflight.models.probes import build_probe_suite


def test_logistic_probe_uses_stable_solver_and_returns_finite_probabilities() -> None:
    rng = np.random.default_rng(11)
    features = pd.DataFrame(rng.normal(size=(200, 32)))
    target = np.asarray([0] * 50 + [1] * 150)
    model = build_probe_suite(features, "binary_classification", 11, "smoke")["logistic"]
    assert model.named_steps["model"].solver == "liblinear"

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        model.fit(features, target)

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        probabilities = model.predict_proba(features)
    assert np.isfinite(probabilities).all()


def test_preprocessing_statistics_are_fit_from_the_training_fold_only() -> None:
    features = pd.DataFrame({"x": [0.0, 2.0, 100.0]})
    target = np.asarray([0.0, 1.0, 2.0])
    model = build_probe_suite(features, "regression", 11, "smoke")["elastic_net"]

    model.fit(features.iloc[:2], target[:2])

    numeric = model.named_steps["preprocess"].named_transformers_["numeric"]
    assert numeric.named_steps["scale"].mean_.tolist() == [1.0]
