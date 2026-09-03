import pandas as pd
import pytest

from bio_ml_preflight.cli import synthetic_case
from bio_ml_preflight.contracts.case import EntitySpec, ScenarioSpec
from bio_ml_preflight.evaluation.capability import capability_matrix


@pytest.mark.parametrize("metric", ["rmse", "mae", "log_loss"])
def test_error_metrics_select_lower_error_and_compare_the_lower_null_tail(tmp_path, metric) -> None:
    case = synthetic_case("no_signal", tmp_path / "unused.csv")
    case.evaluation.primary_metric = metric
    case.generalization_scenarios = [ScenarioSpec(name="random", strategy="random")]
    case.thresholds.supported_metric = 0.75
    case.thresholds.limited_metric = 1.25
    rows = []
    for model, error in [("lower_error", 0.5), ("higher_error", 2.0)]:
        rows.append(
            {
                "scenario": "random",
                "strategy": "random",
                "model": model,
                "permuted": False,
                "permutation_draw": None,
                metric: error,
            }
        )
        rows.extend(
            {
                "scenario": "random",
                "strategy": "random",
                "model": model,
                "permuted": True,
                "permutation_draw": draw,
                metric: 3.0,
            }
            for draw in range(9)
        )

    verdict = capability_matrix(pd.DataFrame(rows), case, {})[0]

    assert verdict["numbers"]["best_model"] == "lower_error"
    assert verdict["numbers"]["median"] == 0.5
    assert verdict["numbers"]["permutation_delta"] == 2.5
    assert verdict["numbers"]["permutation_p_value"] == 0.1
    assert verdict["numbers"]["primary_metric_higher_is_better"] is False
    assert verdict["numbers"]["permutation_q05"] == 3.0
    assert verdict["status"] == "SUPPORTED"


@pytest.mark.parametrize("metric", ["rmse", "spearman"])
@pytest.mark.parametrize(
    "error,null_error,random_error,expected",
    [
        (0.2, 0.9, None, "SUPPORTED"),
        (0.3, 0.9, None, "SUPPORTED"),
        (0.4, 0.9, None, "SUPPORTED_WITH_LIMITS"),
        (0.5, 0.9, None, "SUPPORTED_WITH_LIMITS"),
        (0.6, 0.9, None, "INSUFFICIENT_EVIDENCE"),
        (0.2, 0.1, None, "INSUFFICIENT_EVIDENCE"),
        (0.6, 0.9, 0.2, "CONTRADICTED"),
        (0.4, 0.9, 0.6, "SUPPORTED_WITH_LIMITS"),
    ],
)
def test_metric_direction_preserves_threshold_and_transfer_verdicts(
    tmp_path, metric, error, null_error, random_error, expected
) -> None:
    case = synthetic_case("no_signal", tmp_path / "unused.csv")
    case.evaluation.primary_metric = metric
    case.task.higher_is_better = False  # Outcome ranking must not reverse metric quality.
    case.thresholds.supported_metric = 0.3 if metric == "rmse" else 0.7
    case.thresholds.limited_metric = 0.5
    case.generalization_scenarios = [
        ScenarioSpec(name="held_out", strategy="group", group_column="entity_id")
    ]
    scenarios = [("held_out", "group", error)]
    if random_error is not None:
        case.generalization_scenarios.append(ScenarioSpec(name="random", strategy="random"))
        scenarios.append(("random", "random", random_error))
    rows = []
    for name, strategy, current_error in scenarios:
        for draw in range(-1, 9):
            value = current_error if draw == -1 else null_error
            rows.append(
                {
                    "scenario": name,
                    "strategy": strategy,
                    "model": "ridge",
                    "permuted": draw != -1,
                    "permutation_draw": None if draw == -1 else draw,
                    metric: value if metric == "rmse" else 1 - value,
                }
            )

    assert capability_matrix(pd.DataFrame(rows), case, {})[0]["status"] == expected


def _supported_experiments(scenario: str, strategy: str) -> pd.DataFrame:
    rows = [
        {
            "scenario": scenario,
            "strategy": strategy,
            "model": "ridge",
            "permuted": False,
            "permutation_draw": None,
            "spearman": 0.9,
        }
    ]
    rows.extend(
        {
            "scenario": scenario,
            "strategy": strategy,
            "model": "ridge",
            "permuted": True,
            "permutation_draw": draw,
            "spearman": 0.0,
        }
        for draw in range(9)
    )
    return pd.DataFrame(rows)


def test_random_success_and_cold_failure_is_contradicted(tmp_path) -> None:
    case = synthetic_case("leakage", tmp_path / "x.parquet")
    experiments = pd.DataFrame(
        [
            {
                "scenario": "random_row",
                "strategy": "random",
                "model": "ridge",
                "permuted": False,
                "spearman": 0.95,
            },
            {
                "scenario": "random_row",
                "strategy": "random",
                "model": "ridge",
                "permuted": True,
                "spearman": 0.0,
            },
            {
                "scenario": "cold_entity",
                "strategy": "group",
                "model": "ridge",
                "permuted": False,
                "spearman": 0.02,
            },
            {
                "scenario": "cold_entity",
                "strategy": "group",
                "model": "ridge",
                "permuted": True,
                "spearman": 0.01,
            },
        ]
    )
    matrix = capability_matrix(experiments, case, {})
    assert (
        next(row for row in matrix if row["claim_or_scenario"] == "cold_entity")["status"]
        == "CONTRADICTED"
    )


def test_entity_conflicts_cap_support_and_audit_boundaries_get_rows(tmp_path) -> None:
    case = synthetic_case("no_signal", tmp_path / "x.parquet")
    case.generalization_scenarios = [ScenarioSpec(name="random_row", strategy="random")]
    audits = {
        "independence": {
            "entities": {
                "compound": {
                    "conflicting_target_entities": 5,
                    "inconsistent_representation_entities": 9,
                }
            },
            "pair_structure": {"status": "NOT_ASSESSABLE"},
        },
        "measurement": {
            "status": "NOT_ASSESSABLE",
            "reason": "Repeated class labels do not establish measurement reliability.",
            "unmet_assumption": "The replicate protocol was not declared.",
            "cheapest_next_evidence": "Declare the replicate protocol.",
            "label_consistency_assessed": True,
        },
        "missing_high_value_metadata": [
            "batch_id: batch confounding is not assessable",
            "replicate identifiers: measurement reliability is limited",
        ],
    }

    matrix = capability_matrix(
        _supported_experiments("random_row", "random"),
        case,
        {},
        audits=audits,
        overlap_results={},
    )
    prediction = next(row for row in matrix if row["claim_or_scenario"] == "random_row")
    measurement = next(
        row for row in matrix if row["claim_or_scenario"] == "measurement reliability"
    )
    batch = next(row for row in matrix if row["claim_or_scenario"] == "batch confounding")
    assert prediction["status"] == "SUPPORTED_WITH_LIMITS"
    assert prediction["numbers"]["conflicting_target_entities"] == 5
    assert prediction["numbers"]["inconsistent_representation_entities"] == 9
    assert measurement["status"] == "NOT_ASSESSABLE"
    assert measurement["unmet_assumptions"] == ["The replicate protocol was not declared."]
    assert measurement["cheapest_next_evidence"] == "Declare the replicate protocol."
    assert measurement["numbers"]["label_consistency_assessed"] is True
    assert batch["status"] == "NOT_ASSESSABLE"


def test_protected_entity_overlap_caps_support(tmp_path) -> None:
    case = synthetic_case("stable", tmp_path / "x.parquet")
    case.entities["compound"] = EntitySpec(id_column="compound_id")
    case.generalization_scenarios = [
        ScenarioSpec(name="cold_target", strategy="group", group_column="target_id")
    ]
    overlap = {
        "cold_target:11": {
            "exact_duplicate_overlap": 0,
            "pair_overlap": 0,
            "entity_overlap": {
                "target": {"count": 1, "test_fraction": 0.1},
                "compound": {"count": 20, "test_fraction": 1.0},
            },
        }
    }
    matrix = capability_matrix(
        _supported_experiments("cold_target", "group"),
        case,
        {"top_k": {"5": {"average_pairwise_jaccard": 0.9}}},
        audits={},
        overlap_results=overlap,
    )
    row = next(value for value in matrix if value["claim_or_scenario"] == "cold_target")
    ranking = next(
        value for value in matrix if value["claim_or_scenario"] == "top-5 selection within target"
    )
    assert row["status"] == "SUPPORTED_WITH_LIMITS"
    assert row["numbers"]["protected_entity_overlap_count"] == 1
    assert row["numbers"]["protected_entity_overlap_fraction"] == 0.1
    assert ranking["status"] == "SUPPORTED_WITH_LIMITS"
    assert ranking["numbers"]["protected_entity_overlap_count"] == 1


@pytest.mark.parametrize(
    "conflict,overlap_kind",
    [
        ("conflicting_target_entities", None),
        ("inconsistent_representation_entities", None),
        (None, "exact_duplicate_overlap"),
        (None, "pair_overlap"),
        (None, "entity_overlap"),
        ("conflicting_target_entities", "entity_overlap"),
    ],
)
def test_audit_advice_separates_conflicts_from_overlap(tmp_path, conflict, overlap_kind) -> None:
    case = synthetic_case("stable", tmp_path / "unused.parquet")
    case.generalization_scenarios = [
        ScenarioSpec(name="cold_target", strategy="group", group_column="target_id")
    ]
    experiments = _supported_experiments("cold_target", "group")
    ranking = {"top_k": {"5": {"average_pairwise_jaccard": 0.9}}}
    original = capability_matrix(experiments, case, ranking)
    audits = {"independence": {"entities": {"target": {conflict: 1} if conflict else {}}}}
    overlap = {}
    if overlap_kind:
        overlap["cold_target:11"] = {
            overlap_kind: {"target": {"count": 1, "test_fraction": 0.1}}
            if overlap_kind == "entity_overlap"
            else 1
        }

    rows = capability_matrix(experiments, case, ranking, audits=audits, overlap_results=overlap)

    assert {row["claim_or_scenario"] for row in original} == {
        "cold_target",
        "top-5 selection within target",
    }
    for before in original:
        row = next(row for row in rows if row["claim_or_scenario"] == before["claim_or_scenario"])
        assert row["status"] == "SUPPORTED_WITH_LIMITS"
        advice = row["cheapest_next_evidence"]
        assert ("overlap-free split manifests" in advice) is bool(overlap_kind)
        assert ("conflicting entity records" in advice) is bool(conflict)
        assert before["cheapest_next_evidence"] in advice
        if conflict:
            assert row["numbers"][conflict] == 1
            assert "declared identity policy" in advice
        if not overlap_kind:
            assert not any("split isolation" in value for value in row["unmet_assumptions"])


def test_small_external_minority_class_blocks_confirmation(tmp_path) -> None:
    case = synthetic_case("no_signal", tmp_path / "x.parquet")
    case.task.kind = "binary_classification"
    case.evaluation.primary_metric = "balanced_accuracy"
    case.thresholds.supported_metric = 0.65
    case.thresholds.limited_metric = 0.55
    case.thresholds.minimum_test_class_count = 20
    case.generalization_scenarios = [
        ScenarioSpec(
            name="external",
            strategy="supplied",
            split_column="validation_split",
            group_column="group_id",
        )
    ]
    experiments = pd.DataFrame(
        [
            {
                "scenario": "external",
                "strategy": "supplied",
                "model": "logistic",
                "permuted": False,
                "permutation_draw": None,
                "balanced_accuracy": 0.8,
            },
            *[
                {
                    "scenario": "external",
                    "strategy": "supplied",
                    "model": "logistic",
                    "permuted": True,
                    "permutation_draw": draw,
                    "balanced_accuracy": 0.5,
                }
                for draw in range(9)
            ],
        ]
    )
    overlap = {
        "external:11": {
            "test_target_counts": {"0": 4, "1": 171},
            "test_target_count_unit": "compound_id",
            "exact_duplicate_overlap": 0,
            "pair_overlap": 0,
            "entity_overlap": {"group": {"count": 0, "test_fraction": 0.0}},
        }
    }

    row = capability_matrix(experiments, case, {}, audits={}, overlap_results=overlap)[0]

    assert row["status"] == "INSUFFICIENT_EVIDENCE"
    assert row["numbers"]["test_class_counts"] == {"0": 4, "1": 171}
    assert row["numbers"]["test_class_count_unit"] == "compound_id"
    assert row["numbers"]["minimum_test_class_count_required"] == 20
    assert "at least 16" in row["cheapest_next_evidence"]


@pytest.mark.parametrize("target_conflicts", [0, 1])
def test_adequately_supported_holdout_with_weak_permutation_targets_new_boundary(
    tmp_path,
    target_conflicts,
) -> None:
    case = synthetic_case("no_signal", tmp_path / "x.parquet")
    case.task.kind = "binary_classification"
    case.evaluation.primary_metric = "balanced_accuracy"
    case.thresholds.supported_metric = 0.65
    case.thresholds.limited_metric = 0.55
    case.thresholds.minimum_test_class_count = 20
    case.holdout.enabled = True
    case.generalization_scenarios = [
        ScenarioSpec(
            name="external",
            strategy="supplied",
            split_column="validation_split",
            group_column="compound_id",
        )
    ]
    experiments = pd.DataFrame(
        [
            {
                "scenario": "external",
                "strategy": "supplied",
                "model": "logistic",
                "permuted": False,
                "permutation_draw": None,
                "balanced_accuracy": 0.572,
            },
            *[
                {
                    "scenario": "external",
                    "strategy": "supplied",
                    "model": "logistic",
                    "permuted": True,
                    "permutation_draw": draw,
                    "balanced_accuracy": score,
                }
                for draw, score in enumerate(
                    [0.543, 0.563, 0.515, 0.438, 0.593, 0.584, 0.425, 0.469, 0.458]
                )
            ],
        ]
    )
    overlap = {
        "external:11": {
            "test_target_counts": {"0": 37, "1": 500},
            "test_target_count_unit": "compound_id",
            "exact_duplicate_overlap": 0,
            "pair_overlap": 0,
            "entity_overlap": {"compound": {"count": 0, "test_fraction": 0.0}},
        }
    }

    audits = {
        "independence": {
            "entities": {"compound": {"conflicting_target_entities": target_conflicts}}
        }
    }
    row = capability_matrix(experiments, case, {}, audits=audits, overlap_results=overlap)[0]

    assert row["status"] == "INSUFFICIENT_EVIDENCE"
    assert row["numbers"]["permutation_p_value"] == 0.3
    assert row["numbers"]["permutation_delta"] > 0.05
    assert any("Delta over" in value for value in row["evidence_supporting"])
    assert any("p=0.300" in value for value in row["evidence_against"])
    assert not any("p=0.300" in value for value in row["evidence_supporting"])
    assert any("Both holdout classes meet" in value for value in row["evidence_supporting"])
    assert "target and measurement boundary" in row["cheapest_next_evidence"]
    assert "new untouched confirmation set" in row["cheapest_next_evidence"]
    assert "repeat the fixed validation" not in row["cheapest_next_evidence"]
    assert "Add independent units" not in row["cheapest_next_evidence"]


@pytest.mark.parametrize("target_conflicts", [0, 1])
def test_low_unstable_baseline_and_weak_permutation_are_all_opposing(
    tmp_path, target_conflicts
) -> None:
    case = synthetic_case("no_signal", tmp_path / "x.parquet")
    case.thresholds.limited_metric = 0.2
    case.thresholds.maximum_dispersion = 0.15
    case.thresholds.minimum_permutation_delta = 0.05
    case.thresholds.maximum_permutation_p_value = 0.1
    case.generalization_scenarios = [
        ScenarioSpec(name="unseen_participant", strategy="group", group_column="subject_id")
    ]
    experiments = pd.DataFrame(
        [
            {
                "scenario": "unseen_participant",
                "strategy": "group",
                "model": "extra_trees",
                "permuted": False,
                "permutation_draw": None,
                "spearman": score,
            }
            for score in [0.307, 0.083]
        ]
        + [
            {
                "scenario": "unseen_participant",
                "strategy": "group",
                "model": "extra_trees",
                "permuted": True,
                "permutation_draw": draw,
                "spearman": score,
            }
            for draw, score in enumerate([0.05, 0.08, 0.10, 0.12, 0.125, 0.15, 0.18, 0.20, 0.25])
        ]
    )

    audits = {
        "independence": {
            "entities": {"participant": {"conflicting_target_entities": target_conflicts}}
        }
    }
    row = capability_matrix(experiments, case, {}, audits=audits, overlap_results={})[0]

    assert row["status"] == "INSUFFICIENT_EVIDENCE"
    assert row["numbers"]["permutation_p_value"] == 0.3
    assert any("median spearman=0.195" in value for value in row["evidence_against"])
    assert not any("median spearman=0.195" in value for value in row["evidence_supporting"])
    assert "Permutation separation is insufficient." in row["unmet_assumptions"]
    assert (
        "The controlled baseline does not meet the configured usefulness threshold."
        in row["unmet_assumptions"]
    )
    assert "Performance is unstable across the evaluated splits." in row["unmet_assumptions"]
    assert "Localize the weak target or split boundary" in row["cheapest_next_evidence"]


@pytest.mark.parametrize("target_conflicts", [0, 1])
def test_random_split_repeating_bootstrap_units_caps_support(tmp_path, target_conflicts) -> None:
    case = synthetic_case("no_signal", tmp_path / "x.parquet")
    case.entities = {"participant": EntitySpec(id_column="subject_id")}
    case.evaluation.bootstrap_unit = "subject_id"
    case.generalization_scenarios = [ScenarioSpec(name="random_record", strategy="random")]
    overlap = {
        "random_record:11": {
            "exact_duplicate_overlap": 0,
            "pair_overlap": 0,
            "entity_overlap": {"participant": {"count": 42, "test_fraction": 1.0}},
        }
    }

    audits = {
        "independence": {
            "entities": {"participant": {"conflicting_target_entities": target_conflicts}}
        }
    }
    row = capability_matrix(
        _supported_experiments("random_record", "random"),
        case,
        {},
        audits=audits,
        overlap_results=overlap,
    )[0]

    assert row["status"] == "SUPPORTED_WITH_LIMITS"
    assert row["numbers"]["bootstrap_unit_overlap_count"] == 42
    assert any("repeats 42 subject_id" in value for value in row["evidence_against"])
    assert "subject_id held out" in row["cheapest_next_evidence"]
