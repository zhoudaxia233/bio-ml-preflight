import pandas as pd

from bio_ml_preflight.cli import synthetic_case
from bio_ml_preflight.contracts.case import EntitySpec, ScenarioSpec
from bio_ml_preflight.evaluation.capability import capability_matrix


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


def test_entity_conflicts_cap_support_and_missing_metadata_gets_rows(tmp_path) -> None:
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
            "reason": "No replicate metadata or repeated entity-pair measurements.",
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


def test_adequately_supported_holdout_with_weak_permutation_targets_new_boundary(
    tmp_path,
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

    row = capability_matrix(experiments, case, {}, audits={}, overlap_results=overlap)[0]

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


def test_low_unstable_baseline_and_weak_permutation_are_all_opposing(tmp_path) -> None:
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

    row = capability_matrix(experiments, case, {}, audits={}, overlap_results={})[0]

    assert row["status"] == "INSUFFICIENT_EVIDENCE"
    assert row["numbers"]["permutation_p_value"] == 0.3
    assert any("median spearman=0.195" in value for value in row["evidence_against"])
    assert not any("median spearman=0.195" in value for value in row["evidence_supporting"])
    assert "Permutation separation is insufficient." in row["unmet_assumptions"]
    assert (
        "The controlled baseline is below the configured useful metric." in row["unmet_assumptions"]
    )
    assert "Performance is unstable across the evaluated splits." in row["unmet_assumptions"]


def test_random_split_repeating_bootstrap_units_caps_support(tmp_path) -> None:
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

    row = capability_matrix(
        _supported_experiments("random_record", "random"),
        case,
        {},
        audits={},
        overlap_results=overlap,
    )[0]

    assert row["status"] == "SUPPORTED_WITH_LIMITS"
    assert row["numbers"]["bootstrap_unit_overlap_count"] == 42
    assert any("repeats 42 subject_id" in value for value in row["evidence_against"])
    assert "subject_id held out" in row["cheapest_next_evidence"]
