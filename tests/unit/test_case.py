import hashlib
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from bio_ml_preflight.contracts import CaseSpec, load_case


def test_case_validation_and_relative_path(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "case_id": "x",
        "data": {"path": "data.csv"},
        "task": {"kind": "regression", "prediction_unit": "sample", "target_column": "y"},
        "generalization_scenarios": [{"name": "g", "strategy": "group"}],
    }
    with pytest.raises(ValidationError, match="group_column"):
        CaseSpec.model_validate(payload)
    payload["generalization_scenarios"][0]["group_column"] = "patient_id"
    path = tmp_path / "case.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    assert Path(load_case(path).data.path) == tmp_path / "data.csv"


def test_molecular_representation_ladder_requires_an_explicit_smiles_column() -> None:
    payload = {
        "schema_version": 1,
        "case_id": "molecule",
        "data": {"path": "data.csv"},
        "task": {
            "kind": "binary_classification",
            "prediction_unit": "compound",
            "target_column": "y",
        },
        "features": {"include": ["smiles"], "molecular_representations": ["morgan"]},
        "generalization_scenarios": [{"name": "random", "strategy": "random"}],
    }
    with pytest.raises(ValidationError, match="smiles_column"):
        CaseSpec.model_validate(payload)


def test_minimum_test_class_count_requires_an_independent_unit() -> None:
    payload = {
        "schema_version": 1,
        "case_id": "external",
        "data": {"path": "data.csv"},
        "task": {
            "kind": "binary_classification",
            "prediction_unit": "compound",
            "target_column": "y",
        },
        "generalization_scenarios": [
            {"name": "external", "strategy": "supplied", "split_column": "split"}
        ],
        "thresholds": {"minimum_test_class_count": 20},
    }

    with pytest.raises(ValidationError, match="evaluation.bootstrap_unit"):
        CaseSpec.model_validate(payload)


def test_graph_readiness_requires_a_declared_nonrandom_scenario() -> None:
    payload = {
        "schema_version": 1,
        "case_id": "molecule",
        "data": {"path": "data.csv"},
        "task": {
            "kind": "binary_classification",
            "prediction_unit": "compound",
            "target_column": "y",
        },
        "entities": {
            "compound": {
                "id_column": "compound_id",
                "representation_column": "smiles",
                "identity_conflict_policy": "exclude",
            }
        },
        "features": {"include": ["smiles"], "smiles_column": "smiles"},
        "generalization_scenarios": [{"name": "random", "strategy": "random"}],
        "evaluation": {"bootstrap_unit": "compound_id"},
        "graph_readiness": {
            "structure_column": "smiles",
            "independent_unit_column": "compound_id",
            "node_features": ["atomic_number"],
            "edge_features": ["bond_type"],
            "evaluation_scenarios": ["random"],
            "minimum_independent_units_per_class": 1,
        },
    }

    with pytest.raises(ValidationError, match="random diagnostic"):
        CaseSpec.model_validate(payload)


def test_optional_graph_contract_does_not_change_a_locked_case_fingerprint() -> None:
    root = Path(__file__).resolve().parents[2]
    case = load_case(root / "examples" / "petbd_external" / "case.yaml")
    legacy_payload = case.model_dump_json(exclude={"graph_readiness"}, exclude_none=False)
    legacy_fingerprint = hashlib.sha256(legacy_payload.encode()).hexdigest()

    assert case.graph_readiness is None
    assert case.fingerprint() == legacy_fingerprint
