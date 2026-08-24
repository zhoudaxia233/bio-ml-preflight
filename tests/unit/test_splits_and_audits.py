import numpy as np
import pandas as pd
import pytest

from bio_ml_preflight.audits import (
    apply_identity_conflict_policies,
    audit_dataset,
    audit_overlap,
)
from bio_ml_preflight.cli import synthetic_case
from bio_ml_preflight.contracts.case import EntitySpec, ScenarioSpec
from bio_ml_preflight.data.synthetic import generate_synthetic
from bio_ml_preflight.runner import run_case
from bio_ml_preflight.splits import create_split


def test_group_split_is_deterministic_and_has_no_group_overlap(tmp_path) -> None:
    data_path = tmp_path / "stable.parquet"
    frame = generate_synthetic("stable", data_path)
    scenario = ScenarioSpec(name="cold", strategy="group", group_column="target_id")
    first = create_split(frame, scenario, 11)
    second = create_split(frame, scenario, 11)
    assert first == second
    train_groups = set(frame.iloc[first.train_indices]["target_id"])
    test_groups = set(frame.iloc[first.test_indices]["target_id"])
    assert not train_groups & test_groups


def test_duplicate_overlap_is_detected(tmp_path) -> None:
    frame = pd.DataFrame({"sample_id": ["a", "a", "b"], "x": [1, 1, 2], "y": [3, 3, 4]})
    path = tmp_path / "table.parquet"
    frame.to_parquet(path)
    case = synthetic_case("no_signal", path)
    case.task.target_column = "y"
    case.entities = {}
    case.data.fingerprint_columns = ["sample_id", "x", "y"]
    result = audit_overlap(frame, np.array([0, 2]), np.array([1]), case)
    assert result["exact_duplicate_overlap"] == 1


def test_entity_target_and_representation_conflicts_are_reported(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "compound_id": ["a", "a", "b", "b"],
            "smiles": ["CC", "CO", "CN", "CN"],
            "y": [0, 1, 1, 1],
        }
    )
    case = synthetic_case("no_signal", tmp_path / "unused.parquet")
    case.task.target_column = "y"
    case.task.prediction_unit = "compound"
    case.entities = {
        "compound": EntitySpec(id_column="compound_id", representation_column="smiles")
    }

    result = audit_dataset(frame, case)["independence"]
    compound = result["entities"]["compound"]
    assert compound["conflicting_target_entities"] == 1
    assert compound["inconsistent_representation_entities"] == 1
    assert len(result["warnings"]) == 3


def test_partial_entity_does_not_define_prediction_unit_label_consistency(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "target_id": ["x", "x", "y", "y"],
            "candidate": ["a", "b", "a", "b"],
            "y": [0.1, 0.9, 0.2, 0.8],
        }
    )
    case = synthetic_case("stable", tmp_path / "unused.parquet")

    target = audit_dataset(frame, case)["independence"]["entities"]["target"]
    assert "conflicting_target_entities" not in target


def test_pairwise_target_variation_is_not_an_entity_label_conflict(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "compound_id": ["a", "a", "b", "b"],
            "target_id": ["x", "y", "x", "y"],
            "y": [0.1, 0.9, 0.2, 0.8],
        }
    )
    case = synthetic_case("stable", tmp_path / "unused.parquet")
    case.task.target_column = "y"
    case.entities = {
        "compound": EntitySpec(id_column="compound_id"),
        "target": EntitySpec(id_column="target_id"),
    }

    result = audit_dataset(frame, case)["independence"]
    assert "conflicting_target_entities" not in result["entities"]["compound"]
    assert "conflicting_target_entities" not in result["entities"]["target"]
    assert result["pair_structure"]["conflicting_label_pairs"] == 0


def test_identity_conflicts_require_an_explicit_policy(tmp_path) -> None:
    frame = pd.DataFrame(
        {"compound_id": ["a", "a", "b"], "smiles": ["CC", "CO", "CN"], "y": [0, 1, 1]}
    )
    case = synthetic_case("no_signal", tmp_path / "unused.parquet")
    case.task.target_column = "y"
    case.task.prediction_unit = "compound"
    case.features.include = ["smiles"]
    case.entities = {
        "compound": EntitySpec(id_column="compound_id", representation_column="smiles")
    }

    with pytest.raises(ValueError, match="identity_conflict_policy"):
        apply_identity_conflict_policies(frame, case)


def test_exclude_policy_removes_every_conflicting_identity(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "compound_id": ["a", "a", "b", "b", "c"],
            "smiles": ["CC", "CO", "CN", "CN", "CF"],
            "y": [0, 0, 0, 1, 1],
        }
    )
    case = synthetic_case("no_signal", tmp_path / "unused.parquet")
    case.task.target_column = "y"
    case.task.prediction_unit = "compound"
    case.entities = {
        "compound": EntitySpec(
            id_column="compound_id",
            representation_column="smiles",
            identity_conflict_policy="exclude",
        )
    }

    resolved, result = apply_identity_conflict_policies(frame, case)
    assert resolved["compound_id"].tolist() == ["c"]
    assert result["status"] == "RESOLVED"
    assert result["entities"]["compound"] == {
        "policy": "exclude",
        "conflicting_target_entities": 1,
        "inconsistent_representation_entities": 1,
        "affected_entities": 2,
        "excluded_rows": 4,
    }


def test_aggregate_policy_averages_regression_targets(tmp_path) -> None:
    frame = pd.DataFrame(
        {"compound_id": ["a", "a", "b"], "smiles": ["CC", "CC", "CN"], "y": [1.0, 3.0, 4.0]}
    )
    case = synthetic_case("no_signal", tmp_path / "unused.parquet")
    case.task.target_column = "y"
    case.task.prediction_unit = "compound"
    case.features.include = ["smiles"]
    case.entities = {
        "compound": EntitySpec(
            id_column="compound_id",
            representation_column="smiles",
            identity_conflict_policy="aggregate",
        )
    }

    resolved, result = apply_identity_conflict_policies(frame, case)
    assert resolved.to_dict("records") == [
        {"compound_id": "a", "smiles": "CC", "y": 2.0},
        {"compound_id": "b", "smiles": "CN", "y": 4.0},
    ]
    assert result["entities"]["compound"]["aggregated_rows"] == 1


def test_grouping_by_representation_cannot_split_one_identity_across_folds(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "compound_id": ["a", "a", "b", "c", "d"],
            "smiles": ["CC", "CCC", "CO", "CN", "CF"],
            "y": [0, 0, 1, 0, 1],
        }
    )
    path = tmp_path / "compounds.parquet"
    frame.to_parquet(path)
    case = synthetic_case("no_signal", path)
    case.task.kind = "binary_classification"
    case.task.prediction_unit = "compound"
    case.features.include = ["smiles"]
    case.entities = {
        "compound": EntitySpec(
            id_column="compound_id",
            representation_column="smiles",
            identity_conflict_policy="keep",
        )
    }
    case.generalization_scenarios = [
        ScenarioSpec(name="heldout", strategy="group", group_column="smiles")
    ]
    case.evaluation.seeds = [23]

    with pytest.raises(ValueError, match="identifiers cross train and test"):
        run_case(case, tmp_path / "report", model_allowlist={"dummy"})
