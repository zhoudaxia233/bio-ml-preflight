import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from bio_ml_preflight.cli import _project_root, app
from bio_ml_preflight.contracts import CaseSpec, save_case
from bio_ml_preflight.contracts.case import DataSpec, FeatureSpec, ScenarioSpec, TaskSpec
from bio_ml_preflight.discovery.cards import infer_roles


def test_project_root_uses_checkout_from_current_directory(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "examples").mkdir()
    monkeypatch.chdir(tmp_path)
    assert _project_root() == tmp_path


def test_discovery_treats_two_value_numeric_columns_as_classification_candidates() -> None:
    roles = infer_roles(pd.DataFrame({"measurement": [0.1, 0.2, 0.3], "outcome": [0, 1, 0]}))
    integer_named = infer_roles(pd.DataFrame({0: [0, 1, 0]}))

    regression_targets = {item["column"] for item in roles["candidate_numeric_targets"]}
    classification_targets = {item["column"] for item in roles["candidate_categorical_targets"]}
    assert "outcome" not in regression_targets
    assert "outcome" in classification_targets
    assert integer_named["candidate_categorical_targets"][0]["column"] == "0"


def test_discovery_keeps_physical_categorical_targets_ahead_of_numeric_flags() -> None:
    frame = pd.DataFrame(
        {
            **{f"flag_{index}": [0, 1, 0] for index in range(5)},
            "known_label": ["case", "control", "case"],
        }
    )

    candidates = infer_roles(frame)["candidate_categorical_targets"][:5]

    assert "known_label" in {item["column"] for item in candidates}


def test_validate_case_reports_missing_required_role_confirmations(tmp_path: Path) -> None:
    case = CaseSpec(
        case_id="missing-confirmations",
        data=DataSpec(path="unused.csv"),
        task=TaskSpec(kind="regression", prediction_unit="sample", target_column="y"),
        features=FeatureSpec(include=["x"]),
        generalization_scenarios=[ScenarioSpec(name="random", strategy="random")],
        role_confirmation={"target": True, "batch": False},
    )
    case_path = tmp_path / "case.yaml"
    save_case(case, case_path)

    result = CliRunner().invoke(app, ["validate-case", str(case_path)])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "NOT_ASSESSABLE"
    assert payload["unconfirmed_roles"] == [
        "batch",
        "entities",
        "features",
        "prediction_unit",
    ]
