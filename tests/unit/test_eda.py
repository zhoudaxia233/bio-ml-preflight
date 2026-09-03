import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from bio_ml_preflight.audits import audit_dataset
from bio_ml_preflight.cli import app
from bio_ml_preflight.contracts import CaseSpec, save_case
from bio_ml_preflight.contracts.case import (
    DataSpec,
    DecisionSpec,
    EntitySpec,
    EvaluationSpec,
    FeatureSpec,
    HoldoutSpec,
    MetadataSpec,
    ScenarioSpec,
    TaskSpec,
)
from bio_ml_preflight.eda import (
    assess_readiness,
    development_rows,
    readiness_findings,
    run_eda,
)
from bio_ml_preflight.features import model_feature_columns


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
    assert payload["readiness"]["status"] == "BLOCKED"
    assert payload["readiness"]["model_fitting_allowed"] is False
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
    assert "Pre-model readiness: `BLOCKED`" in (output / "eda.md").read_text(encoding="utf-8")
    assert 'Pre-model readiness: <strong class="BLOCKED">BLOCKED</strong>' in (
        output / "eda.html"
    ).read_text(encoding="utf-8")


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

    case.generalization_scenarios.append(ScenarioSpec(name="random", strategy="random"))
    payload = run_eda(case, tmp_path / "mixed-eda")
    profile = pd.read_parquet(tmp_path / "mixed-eda" / "column_profile.parquet").set_index("column")
    assert payload["scope"]["rows_profiled"] == 3
    assert payload["scope"]["non_training_rows_excluded"] == 0
    assert profile.loc["y", "maximum"] == pytest.approx(999.0)


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


def test_categorical_label_repeats_do_not_claim_measurement_reliability(tmp_path: Path) -> None:
    data_path = tmp_path / "categorical.tsv"
    pd.DataFrame(
        {
            "sample_id": ["a", "a", "b"],
            "replicate_id": ["a-1", "a-1", "b-1"],
            "x": [1.0, 1.1, 2.0],
            "label": [0, 1, 1],
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

    measurement = payload["audits"]["measurement"]
    assert measurement["status"] == "NOT_ASSESSABLE"
    assert measurement["label_consistency_assessed"] is True
    assert measurement["conflicting_label_rate"] == 1.0
    assert "class labels" in measurement["reason"]


def test_eda_blocks_when_required_role_confirmations_are_missing(tmp_path: Path) -> None:
    data_path = tmp_path / "observations.csv"
    pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0.0, 1.0, 2.0]}).to_csv(data_path, index=False)
    case = CaseSpec(
        case_id="missing-confirmations",
        data=DataSpec(path=str(data_path)),
        task=TaskSpec(kind="regression", prediction_unit="sample", target_column="y"),
        features=FeatureSpec(include=["x"]),
        generalization_scenarios=[ScenarioSpec(name="random", strategy="random")],
        role_confirmation={"target": True},
    )

    payload = run_eda(case, tmp_path / "eda")

    assert payload["readiness"]["status"] == "BLOCKED"
    finding = next(row for row in payload["findings"] if row["code"] == "unconfirmed_roles")
    assert finding["status"] == "ACTION_REQUIRED"
    assert "entities, features, prediction_unit" in finding["evidence"]


@pytest.mark.parametrize(
    ("statuses", "expected", "allowed"),
    [
        ([], "READY", True),
        (["INFO"], "READY", True),
        (["WARNING"], "READY_WITH_LIMITS", True),
        (["NOT_ASSESSABLE"], "READY_WITH_LIMITS", True),
        (["INFO", "ACTION_REQUIRED"], "BLOCKED", False),
    ],
)
def test_readiness_is_a_transparent_gate_not_a_quality_score(
    statuses: list[str], expected: str, allowed: bool
) -> None:
    findings = [
        {"code": f"check_{index}", "status": status} for index, status in enumerate(statuses)
    ]

    result = assess_readiness(findings)

    assert result["status"] == expected
    assert result["model_fitting_allowed"] is allowed
    assert "score" not in result


def test_readiness_distinguishes_missing_values_from_non_finite_modeled_values() -> None:
    case = CaseSpec(
        case_id="numeric-boundary",
        data=DataSpec(path="unused.csv"),
        task=TaskSpec(kind="regression", prediction_unit="sample", target_column="y"),
        entities={"sample": EntitySpec(id_column="sample_id")},
        features=FeatureSpec(include=["x"], post_outcome=["context"]),
        generalization_scenarios=[ScenarioSpec(name="random", strategy="random")],
        role_confirmation={
            "target": True,
            "prediction_unit": True,
            "features": True,
            "entities": True,
        },
    )
    missing_feature = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "x": [1.0, np.nan, 3.0],
            "context": [1.0, np.inf, 3.0],
            "y": [0.0, 1.0, 2.0],
        }
    )

    audits = audit_dataset(missing_feature, case)
    findings = readiness_findings(audits, case)

    assert audits["inventory"]["invalid_numeric_values"] == {"context": 1}
    assert next(row for row in findings if row["code"] == "non_finite_context")["status"] == (
        "WARNING"
    )
    assert assess_readiness(findings)["status"] == "READY_WITH_LIMITS"

    non_finite_feature = missing_feature.assign(x=[1.0, np.inf, 3.0], context=[1.0, 2.0, 3.0])
    findings = readiness_findings(audit_dataset(non_finite_feature, case), case)
    assert (
        next(row for row in findings if row["code"] == "non_finite_modeled_features")["status"]
        == "ACTION_REQUIRED"
    )
    assert assess_readiness(findings)["status"] == "BLOCKED"

    mixed_non_finite = missing_feature.assign(
        x=pd.Series([1.0, np.inf, "unknown"], dtype=object),
        context=[1.0, 2.0, 3.0],
    )
    findings = readiness_findings(audit_dataset(mixed_non_finite, case), case)
    assert "non_finite_modeled_features" in assess_readiness(findings)["blocking_checks"]

    categorical_literal = missing_feature.assign(
        x=pd.Series(["inf", "alpha", "beta"], dtype=object),
        context=[1.0, 2.0, 3.0],
    )
    findings = readiness_findings(audit_dataset(categorical_literal, case), case)
    assert "non_finite_modeled_features" not in assess_readiness(findings)["blocking_checks"]

    all_missing_feature = missing_feature.assign(x=[np.nan, np.nan, np.nan], context=[1, 2, 3])
    findings = readiness_findings(audit_dataset(all_missing_feature, case), case)
    assert "no_observed_modeled_features" in assess_readiness(findings)["blocking_checks"]


def test_one_class_binary_target_and_conflicting_entity_pairs_block_readiness() -> None:
    case = CaseSpec(
        case_id="target-boundary",
        data=DataSpec(path="unused.csv"),
        task=TaskSpec(
            kind="binary_classification",
            prediction_unit="compound-target pair",
            target_column="y",
        ),
        entities={
            "compound": EntitySpec(id_column="compound_id"),
            "target": EntitySpec(id_column="target_id"),
        },
        features=FeatureSpec(include=["x"]),
        generalization_scenarios=[ScenarioSpec(name="random", strategy="random")],
        role_confirmation={
            "target": True,
            "prediction_unit": True,
            "features": True,
            "entities": True,
        },
    )
    one_class = pd.DataFrame(
        {
            "compound_id": ["a", "b", "c"],
            "target_id": ["u", "v", "w"],
            "x": [1.0, 2.0, 3.0],
            "y": [1, 1, 1],
        }
    )
    findings = readiness_findings(audit_dataset(one_class, case), case)
    assert "invalid_binary_target_support" in assess_readiness(findings)["blocking_checks"]

    conflicting_pair = pd.DataFrame(
        {
            "compound_id": ["a", "a", "b", "c"],
            "target_id": ["u", "u", "v", "w"],
            "x": [1.0, 1.0, 2.0, 3.0],
            "y": [0, 1, 0, 1],
        }
    )
    findings = readiness_findings(audit_dataset(conflicting_pair, case), case)
    assert "pair_target_conflict" in assess_readiness(findings)["blocking_checks"]

    case.task.prediction_unit = "pair"
    findings = readiness_findings(audit_dataset(conflicting_pair, case), case)
    assert "pair_target_conflict" in assess_readiness(findings)["blocking_checks"]

    case.task.prediction_unit = "measurement"
    findings = readiness_findings(audit_dataset(conflicting_pair, case), case)
    assert "pair_target_conflict" not in assess_readiness(findings)["blocking_checks"]


@pytest.mark.parametrize("labels", [["negative", "positive", "negative"], [1, 2, 1]])
def test_binary_target_requires_numeric_zero_one_encoding(labels: list[object]) -> None:
    case = CaseSpec(
        case_id="binary-encoding",
        data=DataSpec(path="unused.csv"),
        task=TaskSpec(
            kind="binary_classification",
            prediction_unit="sample",
            target_column="y",
        ),
        features=FeatureSpec(include=["x"]),
        generalization_scenarios=[ScenarioSpec(name="random", strategy="random")],
        role_confirmation={
            "target": True,
            "prediction_unit": True,
            "features": True,
            "entities": True,
        },
    )
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": labels})

    readiness = assess_readiness(readiness_findings(audit_dataset(frame, case), case))

    assert "invalid_binary_target_encoding" in readiness["blocking_checks"]


def test_pending_identity_policy_is_only_acknowledged_before_runner_applies_it() -> None:
    case = CaseSpec(
        case_id="pending-policy",
        data=DataSpec(path="unused.csv"),
        task=TaskSpec(kind="regression", prediction_unit="sample", target_column="y"),
        entities={"sample": EntitySpec(id_column="sample_id", identity_conflict_policy="exclude")},
        features=FeatureSpec(include=["x"]),
        generalization_scenarios=[ScenarioSpec(name="random", strategy="random")],
    )
    frame = pd.DataFrame({"sample_id": ["a", "a", "b"], "x": [1.0, 2.0, 3.0], "y": [0.0, 1.0, 2.0]})
    audits = audit_dataset(frame, case)

    pending = readiness_findings(audits, case)
    residual = readiness_findings(audits, case, allow_pending_identity_policies=False)

    assert (
        next(row for row in pending if row["code"] == "identity_conflict:sample")["status"]
        == "WARNING"
    )
    assert (
        next(row for row in residual if row["code"] == "identity_conflict:sample")["status"]
        == "ACTION_REQUIRED"
    )


@pytest.mark.parametrize("kind", ["regression", "ranking"])
def test_continuous_targets_must_be_numeric_and_nonconstant(kind: str) -> None:
    case = CaseSpec(
        case_id="continuous-target",
        data=DataSpec(path="unused.csv"),
        task=TaskSpec(kind=kind, prediction_unit="sample", target_column="y"),
        features=FeatureSpec(include=["x"]),
        generalization_scenarios=[ScenarioSpec(name="random", strategy="random")],
        role_confirmation={
            "target": True,
            "prediction_unit": True,
            "features": True,
            "entities": True,
        },
    )

    non_numeric = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": ["low", "high", "low"]})
    readiness = assess_readiness(readiness_findings(audit_dataset(non_numeric, case), case))
    assert "non_numeric_target" in readiness["blocking_checks"]

    constant = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 1.0, 1.0]})
    readiness = assess_readiness(readiness_findings(audit_dataset(constant, case), case))
    assert "invalid_continuous_target_support" in readiness["blocking_checks"]


def test_automatic_feature_selection_excludes_controls_and_explicit_leakage_blocks() -> None:
    columns = [
        "sample_id",
        "bootstrap_id",
        "ranking_group",
        "treatment",
        "x",
        "after_y",
        "split",
        "y",
    ]
    case = CaseSpec(
        case_id="feature-boundary",
        data=DataSpec(path="unused.csv"),
        task=TaskSpec(kind="regression", prediction_unit="sample", target_column="y"),
        entities={"sample": EntitySpec(id_column="sample_id")},
        features=FeatureSpec(post_outcome=["after_y"]),
        metadata=MetadataSpec(treatment_column="treatment", control_value="x"),
        decision=DecisionSpec(kind="top_k_per_group", group_entity="ranking_group"),
        generalization_scenarios=[
            ScenarioSpec(name="supplied", strategy="supplied", split_column="split")
        ],
        evaluation=EvaluationSpec(bootstrap_unit="bootstrap_id"),
        role_confirmation={
            "target": True,
            "prediction_unit": True,
            "features": True,
            "entities": True,
        },
    )

    assert model_feature_columns(columns, case) == ["x"]

    case.features.include = ["x", "after_y", "y"]
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "bootstrap_id": ["i", "j", "k"],
            "ranking_group": ["g", "g", "h"],
            "treatment": ["control", "treated", "control"],
            "x": [1.0, 2.0, 3.0],
            "after_y": [0.0, 1.0, 2.0],
            "split": ["train", "train", "test"],
            "y": [0.0, 1.0, 2.0],
        }
    )
    findings = readiness_findings(audit_dataset(frame, case), case)
    blocking_codes = set(assess_readiness(findings)["blocking_checks"])
    assert {"target_modeled_as_feature", "post_outcome_modeled_as_feature"} <= blocking_codes

    case.features.include = []
    no_predictors = frame.drop(columns="x")
    findings = readiness_findings(audit_dataset(no_predictors, case), case)
    assert "no_modeled_features" in assess_readiness(findings)["blocking_checks"]


def test_holdout_requires_consistent_training_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0],
            "y": [0, 1],
            "split_a": ["train", "holdout"],
            "split_b": ["holdout", "train"],
        }
    )
    case = CaseSpec(
        case_id="inconsistent-holdout",
        data=DataSpec(path="unused.csv"),
        task=TaskSpec(kind="binary_classification", prediction_unit="row", target_column="y"),
        features=FeatureSpec(include=["x"]),
        generalization_scenarios=[
            ScenarioSpec(name="a", strategy="supplied", split_column="split_a"),
            ScenarioSpec(name="b", strategy="supplied", split_column="split_b"),
        ],
        holdout=HoldoutSpec(enabled=True),
    )

    with pytest.raises(ValueError, match="consistent training boundary"):
        development_rows(frame, case)
