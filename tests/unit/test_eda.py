import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from bio_ml_preflight.cli import app
from bio_ml_preflight.contracts import CaseSpec, save_case
from bio_ml_preflight.contracts.case import (
    DataSpec,
    EntitySpec,
    EvaluationSpec,
    FeatureSpec,
    HoldoutSpec,
    MetadataSpec,
    ScenarioSpec,
    TaskSpec,
)
from bio_ml_preflight.eda import run_eda


def test_eda_command_persists_quality_evidence_without_modeling(tmp_path: Path) -> None:
    data_path = tmp_path / "observations.parquet"
    pd.DataFrame(
        {
            "patient_id": ["a", "a", "b", "b", "c"],
            "x": [1.0, None, 2.0, 2.0, 3.0],
            "constant": [1, 1, 1, 1, 1],
            "after_y": [0, 1, 2, 2, 3],
            "y": [0, 1, 1, 1, 0],
        }
    ).to_parquet(data_path, index=False)
    case = CaseSpec(
        case_id="quality-check",
        data=DataSpec(path=str(data_path)),
        task=TaskSpec(kind="binary_classification", prediction_unit="patient", target_column="y"),
        entities={"patient": EntitySpec(id_column="patient_id")},
        features=FeatureSpec(include=["x", "constant"], post_outcome=["after_y"]),
        generalization_scenarios=[
            ScenarioSpec(name="unseen_patient", strategy="group", group_column="patient_id")
        ],
        evaluation=EvaluationSpec(bootstrap_unit="patient_id"),
        role_confirmation={
            "target": True,
            "prediction_unit": True,
            "features": True,
            "entities": True,
        },
    )
    case_path = tmp_path / "case.yaml"
    save_case(case, case_path)
    output = tmp_path / "eda"

    result = CliRunner().invoke(app, ["eda", str(case_path), "--output", str(output)])

    assert result.exit_code == 0, result.output
    payload = json.loads((output / "eda.json").read_text(encoding="utf-8"))
    assert payload["scope"] == {
        "holdout_labels_accessed": False,
        "model_trained": False,
        "non_training_rows_excluded": 0,
        "rows_loaded": 5,
        "rows_profiled": 5,
    }
    assert payload["audits"]["inventory"]["missing"] == {"x": 1}
    assert payload["audits"]["inventory"]["duplicate_rows"] == 1
    checks = {row["check"]: row for row in payload["findings"]}
    assert checks["identity consistency patient"]["status"] == "ACTION_REQUIRED"
    assert checks["uninformative modeled features"]["status"] == "WARNING"
    assert checks["declared post-outcome fields"]["status"] == "INFO"
    profile = pd.read_parquet(output / "column_profile.parquet").set_index("column")
    assert profile.loc["y", "roles"] == "target"
    assert profile.loc["x", "missing_fraction"] == pytest.approx(0.2)
    assert profile.loc["after_y", "roles"] == "post_outcome"
    for relative in [
        "eda.md",
        "eda.html",
        "figures/missingness.png",
        "figures/target_distribution.png",
        "figures/entity_repetition.png",
    ]:
        assert (output / relative).is_file()


def test_eda_profiles_only_training_rows_for_a_supplied_split(tmp_path: Path) -> None:
    data_path = tmp_path / "split.csv"
    pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "x": [1.0, 2.0, 999.0],
            "y": [0.1, 0.2, 999.0],
            "split": ["train", "train", "test"],
        }
    ).to_csv(data_path, index=False)
    case = CaseSpec(
        case_id="development-only",
        data=DataSpec(path=str(data_path)),
        task=TaskSpec(kind="regression", prediction_unit="sample", target_column="y"),
        entities={"sample": EntitySpec(id_column="sample_id")},
        features=FeatureSpec(include=["x"]),
        generalization_scenarios=[
            ScenarioSpec(name="supplied", strategy="supplied", split_column="split")
        ],
    )

    payload = run_eda(case, tmp_path / "eda")
    profile = pd.read_parquet(tmp_path / "eda" / "column_profile.parquet").set_index("column")

    assert payload["scope"]["rows_profiled"] == 2
    assert payload["scope"]["non_training_rows_excluded"] == 1
    assert profile.loc["y", "maximum"] == pytest.approx(0.2)


def test_eda_refuses_a_holdout_enabled_case_before_reading_data(tmp_path: Path) -> None:
    case = CaseSpec(
        case_id="locked",
        data=DataSpec(path=str(tmp_path / "does-not-exist.csv")),
        task=TaskSpec(kind="regression", prediction_unit="sample", target_column="y"),
        features=FeatureSpec(include=["x"]),
        generalization_scenarios=[
            ScenarioSpec(name="locked", strategy="supplied", split_column="split")
        ],
        holdout=HoldoutSpec(enabled=True),
    )

    with pytest.raises(ValueError, match="refuses holdout-enabled cases"):
        run_eda(case, tmp_path / "eda")


def test_eda_handles_categorical_targets_with_replicate_metadata(tmp_path: Path) -> None:
    data_path = tmp_path / "categorical.tsv"
    pd.DataFrame(
        {
            "sample_id": ["a", "a", "b", "b"],
            "replicate_id": ["a-1", "a-1", "b-1", "b-1"],
            "x": [1.0, 1.1, 2.0, 2.1],
            "label": ["case", "case", "control", "control"],
        }
    ).to_csv(data_path, sep="\t", index=False)
    case = CaseSpec(
        case_id="categorical-replicates",
        data=DataSpec(path=str(data_path)),
        task=TaskSpec(
            kind="binary_classification", prediction_unit="sample", target_column="label"
        ),
        entities={"sample": EntitySpec(id_column="sample_id")},
        features=FeatureSpec(include=["x"]),
        metadata=MetadataSpec(replicate_id="replicate_id"),
        generalization_scenarios=[
            ScenarioSpec(name="sample", strategy="group", group_column="sample_id")
        ],
    )

    payload = run_eda(case, tmp_path / "eda")

    assert payload["audits"]["measurement"]["status"] == "ASSESSED"
    assert payload["audits"]["measurement"]["within_group_dispersion_median"] is None
    assert "rank_consistency_proxy" not in payload["audits"]["measurement"]
