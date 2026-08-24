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
