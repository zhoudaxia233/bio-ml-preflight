"""Keep mechanical evidence distinct from biological interpretation."""

import numpy as np
import pandas as pd
import pytest

from bio_ml_preflight.audits import audit_overlap
from bio_ml_preflight.cli import synthetic_case
from bio_ml_preflight.contracts.case import EntitySpec, ScenarioSpec
from bio_ml_preflight.evaluation.capability import capability_matrix
from bio_ml_preflight.runner import _test_target_counts
from bio_ml_preflight.splits.core import SplitManifest
from bio_ml_preflight.stability.ranking import stability_decomposition


def controlled_runs(partitions=("fixed", "fixed"), method="row", unit=None):
    return pd.DataFrame(
        [
            {
                "scenario": "comparison",
                "strategy": "supplied",
                "model": "linear",
                "seed": seed,
                "permuted": draw >= 0,
                "permutation_draw": draw if draw >= 0 else None,
                "permutation_method": method if draw >= 0 else None,
                "permutation_unit": unit if draw >= 0 else None,
                "partition_fingerprint": partition,
                "spearman": 0.0 if draw >= 0 else 0.9,
            }
            for seed, partition in zip([11, 23], partitions, strict=True)
            for draw in range(-1, 9)
        ]
    )


def test_patient_support_can_include_both_visit_outcomes(tmp_path):
    case = synthetic_case("no_signal", tmp_path / "unused")
    case.task.prediction_unit = "visit"
    case.task.kind = "binary_classification"
    case.evaluation.bootstrap_unit = "patient_id"
    frame = pd.DataFrame({"patient_id": ["a", "a", "a", "b"], "y": [0, 1, 1, 0]})

    counts, unit = _test_target_counts(frame, np.arange(4), case)

    assert counts == {"0": 2, "1": 1}
    assert unit == "patient_id"


@pytest.mark.parametrize(
    "boundary,expected",
    [
        ("context", "SUPPORTED"),
        ("random_context", "SUPPORTED"),
        ("exact_duplicate", "SUPPORTED_WITH_LIMITS"),
        ("held_out_cell", "SUPPORTED_WITH_LIMITS"),
        ("unseen_compound", "NOT_ASSESSABLE"),
        ("unrelated_entity", "SUPPORTED_WITH_LIMITS"),
        ("legacy_pair_columns", "SUPPORTED_WITH_LIMITS"),
    ],
)
def test_compound_cell_pairs_across_plates_follow_declared_boundary(tmp_path, boundary, expected):
    case = synthetic_case("no_signal", tmp_path / "unused")
    case.entities = {
        "compound": EntitySpec(id_column="compound"),
        "cell": EntitySpec(id_column="cell"),
    }
    scenario = ScenarioSpec(
        name="comparison",
        strategy="random" if boundary == "random_context" else "supplied",
        split_column="partition",
        group_column="cell" if boundary == "held_out_cell" else None,
        split_claim={
            "kind": "unseen_entity"
            if boundary == "unseen_compound"
            else "same_entity_across_context",
            "entity_column": "donor" if boundary == "unrelated_entity" else "compound",
            "context_column": "plate",
        },
    )
    case.generalization_scenarios = [scenario]
    case.evaluation.bootstrap_unit = "compound"
    case.data.fingerprint_columns = ["compound", "cell", "plate"]
    frame = pd.DataFrame(
        {
            "compound": ["a", "b", "a", "b"],
            "cell": ["u2os"] * 4,
            "plate": [1, 1, 2, 2],
            "donor": ["d"] * 4,
        }
    )
    audit = audit_overlap(frame, np.array([0, 1]), np.array([2, 3]), case, scenario)
    if boundary == "exact_duplicate":
        audit["exact_duplicate_overlap"] = 1
    if boundary == "legacy_pair_columns":
        audit.pop("pair_columns")
    runs = controlled_runs()
    runs["strategy"] = scenario.strategy
    verdict = capability_matrix(runs, case, {}, overlap_results={"comparison:11": audit})[0]

    assert audit["pair_overlap"] == 2  # Keep the raw observation in every case.
    assert verdict["status"] == expected
    if boundary in {"context", "random_context"}:
        assert verdict["numbers"]["expected_pair_overlap"] == 2
        assert "overlap-free" not in verdict["cheapest_next_evidence"]
        assert not any(
            "independence boundary has overlap" in s for s in verdict["unmet_assumptions"]
        )
    elif boundary in {
        "exact_duplicate",
        "held_out_cell",
        "unrelated_entity",
        "legacy_pair_columns",
    }:
        assert "overlap-free" in verdict["cheapest_next_evidence"]


def test_partition_identity_ignores_seed_and_order_but_preserves_roles():
    first = SplitManifest("a", "supplied", 11, [0, 1], [2, 3], [4])
    repeat = SplitManifest("b", "supplied", 23, [1, 0], [3, 2], [4])
    swapped = SplitManifest("a", "supplied", 11, [2, 3], [0, 1], [4])

    assert first.fingerprint() != repeat.fingerprint()  # Existing lock identity stays intact.
    assert first.membership_fingerprint() == repeat.membership_fingerprint()
    assert first.membership_fingerprint() != swapped.membership_fingerprint()


@pytest.mark.parametrize("partitions", [("fixed", "fixed"), (None, None), ("a", None)])
def test_repeated_or_unrecorded_partitions_do_not_estimate_split_uncertainty(tmp_path, partitions):
    runs = controlled_runs(partitions)
    case = synthetic_case("no_signal", tmp_path / "unused")
    case.generalization_scenarios = [
        ScenarioSpec(name="comparison", strategy="supplied", split_column="partition")
    ]

    decomposition = stability_decomposition(runs, "spearman")["train_validation_split"]
    verdict = capability_matrix(runs, case, {})[0]

    assert decomposition["status"] == "NOT_ASSESSABLE"
    assert decomposition["median_standard_deviation"] is None
    assert "Across-split standard deviation=0.000" not in verdict["uncertainty"]
    assert verdict["numbers"]["split_variation"]["status"] == "NOT_ASSESSABLE"


def test_distinct_partitions_use_one_summary_per_partition():
    runs = controlled_runs(("a", "b"))
    runs.loc[runs.seed.eq(23) & runs.permuted.eq(False), "spearman"] = 0.5
    # Unequal repeated runs must not give partition a more weight.
    runs = pd.concat([runs, runs[runs.seed.eq(11)]], ignore_index=True)
    result = stability_decomposition(runs, "spearman")["train_validation_split"]

    assert result["status"] == "ASSESSED"
    assert result["median_standard_deviation"] == pytest.approx(np.std([0.9, 0.5], ddof=1))
    assert "initialization" in result["scope"]  # Descriptive variation, not a causal decomposition.


def test_single_run_and_legacy_partitions_have_no_estimated_variation(tmp_path):
    case = synthetic_case("no_signal", tmp_path / "unused")
    case.generalization_scenarios = [
        ScenarioSpec(name="comparison", strategy="supplied", split_column="partition")
    ]
    runs = controlled_runs().query("seed == 11").drop(columns="partition_fingerprint")
    verdict = capability_matrix(runs, case, {})[0]

    assert verdict["numbers"]["dispersion"] is None
    assert "one finite run" in verdict["uncertainty"]
    assert verdict["numbers"]["split_variation"]["distinct_partitions"] is None


def test_different_models_do_not_supply_each_others_missing_partitions():
    runs = controlled_runs(("a", "b"))
    runs.loc[runs.seed.eq(23), "model"] = "other_model"
    result = stability_decomposition(runs, "spearman")["train_validation_split"]

    assert result["status"] == "NOT_ASSESSABLE"
    assert result["median_standard_deviation"] is None


def test_undefined_metric_does_not_hide_missing_partition_metadata():
    runs = controlled_runs(("a", "b"))
    incomplete = runs[runs.permuted.eq(False)].iloc[[0]].copy()
    incomplete["seed"] = 47
    incomplete["partition_fingerprint"] = None
    incomplete["spearman"] = np.nan
    runs = pd.concat([runs, incomplete], ignore_index=True)

    result = stability_decomposition(runs, "spearman")["train_validation_split"]

    assert result["status"] == "NOT_ASSESSABLE"
    assert result["median_standard_deviation"] is None
    assert "not recorded for every run" in result["reason"]


@pytest.mark.parametrize(
    "method,unit,label",
    [
        ("row", None, "row permutation"),
        ("equal_size_group_blocks_within_group_shuffle", "patient_id", "patient_id"),
        (None, None, "unrecorded"),
    ],
)
def test_permutation_description_uses_executed_method_not_case_guess(tmp_path, method, unit, label):
    case = synthetic_case("no_signal", tmp_path / "unused")
    case.generalization_scenarios = [
        ScenarioSpec(name="comparison", strategy="supplied", split_column="partition")
    ]
    case.evaluation.bootstrap_unit = "unrelated_case_default"
    runs = controlled_runs(method=method, unit=unit)
    if method is None:
        runs = runs.drop(columns=["permutation_method", "permutation_unit"])

    verdict = capability_matrix(runs, case, {})[0]
    evidence = " ".join(verdict["evidence_supporting"] + verdict["evidence_against"])

    assert label in evidence
    assert "grouped-permutation" not in evidence
    assert verdict["numbers"]["permutation_design"]["exchangeability"] == "NOT_ASSESSABLE"
