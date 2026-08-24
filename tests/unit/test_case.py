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
