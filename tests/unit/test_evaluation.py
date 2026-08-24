import numpy as np
import pandas as pd

from bio_ml_preflight.evaluation.metrics import bootstrap_interval, group_respecting_permutation
from bio_ml_preflight.stability.ranking import jaccard, ranking_stability


def test_group_permutation_preserves_equal_size_label_blocks() -> None:
    values = np.array([1, 2, 3, 4, 5, 6])
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    permuted = group_respecting_permutation(values, groups, 11)
    original_blocks = {tuple(sorted(values[groups == group])) for group in np.unique(groups)}
    permuted_blocks = {tuple(sorted(permuted[groups == group])) for group in np.unique(groups)}
    assert permuted_blocks == original_blocks
    assert not np.array_equal(values, permuted)


def test_bootstrap_resamples_independent_units() -> None:
    frame = pd.DataFrame({"patient": np.repeat(["a", "b", "c"], 4), "value": np.arange(12.0)})
    result = bootstrap_interval(
        frame,
        unit_column="patient",
        statistic=lambda values: float(values["value"].mean()),
        seed=7,
        draws=100,
    )
    assert result["unit_count"] == 3
    assert result["lower"] < result["estimate"] < result["upper"]


def test_top_k_overlap_and_rank_stability() -> None:
    rows = []
    for run_id, scores in {"r1": [4, 3, 2, 1], "r2": [4, 2, 3, 1]}.items():
        for candidate, score in zip("abcd", scores, strict=True):
            rows.append({"run_id": run_id, "group": "g", "candidate": candidate, "y_pred": score})
    summary, memberships = ranking_stability(
        pd.DataFrame(rows),
        group_column="group",
        candidate_column="candidate",
        k_values=[2],
        higher_is_better=True,
    )
    assert jaccard({"a", "b"}, {"a", "c"}) == 1 / 3
    assert summary["top_k"]["2"]["average_pairwise_jaccard"] == 1 / 3
    assert set(memberships["selection_probability"]) == {0.5, 1.0}
