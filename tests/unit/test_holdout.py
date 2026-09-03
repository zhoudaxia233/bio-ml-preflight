from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import bio_ml_preflight.runner as runner_module
from bio_ml_preflight.cli import synthetic_case
from bio_ml_preflight.contracts.case import EntitySpec, ScenarioSpec
from bio_ml_preflight.provenance import HoldoutLedger
from bio_ml_preflight.runner import run_case


def test_holdout_access_requires_reason_after_limit(tmp_path: Path) -> None:
    ledger = HoldoutLedger(tmp_path / "holdout.jsonl", maximum_accesses=1)
    ledger.record_access(actor="tester", purpose="final confirmation")
    with pytest.raises(PermissionError, match="explicit override reason"):
        ledger.record_access(actor="tester", purpose="another look")
    ledger.record_access(actor="tester", purpose="audited correction", override_reason="label fix")
    assert ledger.entries()[-1]["override"] is True


def test_enabled_holdout_case_can_only_run_once_across_outputs(tmp_path: Path) -> None:
    data_path = tmp_path / "holdout.parquet"
    pd.DataFrame(
        {
            "group_id": ["a", "b", "c", "d", "e", "f"],
            "x": [0.0, 1.0, 0.2, 0.8, 0.1, 0.9],
            "y": [0, 1, 0, 1, 0, 1],
            "split": ["train", "train", "train", "train", "holdout", "holdout"],
        }
    ).to_parquet(data_path)
    case = synthetic_case("no_signal", data_path)
    case.task.kind = "binary_classification"
    case.features.include = ["x"]
    case.generalization_scenarios = [
        ScenarioSpec(
            name="external",
            strategy="supplied",
            split_column="split",
            group_column="group_id",
        )
    ]
    case.evaluation.seeds = [11]
    case.evaluation.primary_metric = "balanced_accuracy"
    case.holdout.enabled = True
    first_output = tmp_path / "report-a"
    second_output = tmp_path / "report-b"

    result = run_case(case, first_output, model_allowlist={"dummy"})

    assert result["holdout_access"][0]["access_number"] == 1
    first_fingerprint = result["holdout_access"][0]["case_fingerprint"]
    case.evaluation.classification_threshold = 0.7
    assert case.fingerprint() != first_fingerprint
    with pytest.raises(PermissionError, match="Holdout access limit"):
        run_case(case, second_output, model_allowlist={"dummy"})

    overridden = run_case(
        case,
        second_output,
        model_allowlist={"dummy"},
        holdout_override_reason="audited test override",
    )

    assert overridden["holdout_access"][-1]["access_number"] == 2
    assert overridden["holdout_access"][-1]["override"] is True
    assert overridden["holdout_access"][-1]["override_reason"] == "audited test override"
    assert overridden["holdout_access"][-1]["case_fingerprint"] == case.fingerprint()


def test_declaration_block_precedes_holdout_access_and_table_read(
    tmp_path: Path, monkeypatch
) -> None:
    case = synthetic_case("no_signal", tmp_path / "does-not-exist.parquet")
    case.role_confirmation["features"] = False
    case.generalization_scenarios = [
        ScenarioSpec(name="locked", strategy="supplied", split_column="split")
    ]
    case.holdout.enabled = True

    def unexpected_access(*_args, **_kwargs):
        pytest.fail("holdout data was accessed before the declaration gate")

    monkeypatch.setattr(runner_module, "file_checksum", unexpected_access)
    monkeypatch.setattr(runner_module, "read_table", unexpected_access)

    with pytest.raises(ValueError, match="Pre-model readiness gate BLOCKED"):
        runner_module.run_case(case, tmp_path / "report")


def test_holdout_label_conflict_cannot_remove_protected_identity(
    tmp_path: Path, monkeypatch
) -> None:
    data_path = tmp_path / "identity-overlap.parquet"
    pd.DataFrame(
        {
            "compound_id": ["a", "a", "b", "c", "d", "e"],
            "representation": ["A", "A", "B", "C", "D", "E"],
            "x": [0.0, 1.0, 0.2, 0.8, 0.1, 0.9],
            "y": [0, 1, 1, 0, 1, 0],
            "split": ["train", "holdout", "train", "train", "holdout", "holdout"],
        }
    ).to_parquet(data_path, index=False)
    case = synthetic_case("no_signal", data_path)
    case.task.kind = "binary_classification"
    case.task.prediction_unit = "compound"
    case.features.include = ["x"]
    case.entities = {
        "compound": EntitySpec(
            id_column="compound_id",
            representation_column="representation",
            identity_conflict_policy="exclude",
        )
    }
    case.generalization_scenarios = [
        ScenarioSpec(
            name="external",
            strategy="supplied",
            split_column="split",
        )
    ]
    case.evaluation.seeds = [11]
    case.evaluation.primary_metric = "balanced_accuracy"
    case.evaluation.bootstrap_unit = "compound_id"
    case.holdout.enabled = True

    def unexpected_feature_construction(*_args, **_kwargs):
        pytest.fail("feature construction ran before protected identity isolation")

    monkeypatch.setattr(runner_module, "build_feature_frames", unexpected_feature_construction)

    with pytest.raises(ValueError, match="identifiers cross train and test"):
        runner_module.run_case(case, tmp_path / "report", model_allowlist={"dummy"})


@pytest.mark.parametrize(
    ("test_targets", "test_features", "blocking_code"),
    [
        ([1, 2], [0.1, 0.9], "invalid_binary_target_encoding"),
        ([1, np.inf], [0.1, 0.9], "non_finite_target"),
        ([1, 1], [0.1, 0.9], "invalid_binary_target_support"),
        ([0, 1], [0.1, np.inf], "non_finite_modeled_features"),
    ],
)
def test_invalid_holdout_contract_blocks_before_features(
    tmp_path: Path,
    monkeypatch,
    test_targets: list[float],
    test_features: list[float],
    blocking_code: str,
) -> None:
    data_path = tmp_path / "invalid-holdout.parquet"
    pd.DataFrame(
        {
            "group_id": ["a", "b", "c", "d", "e", "f"],
            "x": [0.0, 1.0, 0.2, 0.8, *test_features],
            "y": [0, 1, 0, 1, *test_targets],
            "split": ["train", "train", "train", "train", "holdout", "holdout"],
        }
    ).to_parquet(data_path, index=False)
    case = synthetic_case("no_signal", data_path)
    case.task.kind = "binary_classification"
    case.features.include = ["x"]
    case.generalization_scenarios = [
        ScenarioSpec(
            name="external",
            strategy="supplied",
            split_column="split",
            group_column="group_id",
        )
    ]
    case.evaluation.seeds = [11]
    case.evaluation.primary_metric = "balanced_accuracy"
    case.holdout.enabled = True

    def unexpected_feature_construction(*_args, **_kwargs):
        pytest.fail("feature construction ran with an invalid holdout contract")

    monkeypatch.setattr(runner_module, "build_feature_frames", unexpected_feature_construction)

    with pytest.raises(ValueError, match=blocking_code):
        runner_module.run_case(case, tmp_path / "report", model_allowlist={"dummy"})

    assert not (tmp_path / "report").exists()


def test_unapplied_mutating_policy_cannot_downgrade_holdout_identity_conflict(
    tmp_path: Path, monkeypatch
) -> None:
    data_path = tmp_path / "conflicting-holdout-identity.parquet"
    pd.DataFrame(
        {
            "compound_id": ["a", "b", "c", "d", "q", "q"],
            "row_id": ["r0", "r1", "r2", "r3", "r4", "r5"],
            "x": [0.0, 1.0, 0.2, 0.8, 0.1, 0.9],
            "y": [0, 1, 0, 1, 0, 1],
            "split": ["train", "train", "train", "train", "holdout", "holdout"],
        }
    ).to_parquet(data_path, index=False)
    case = synthetic_case("no_signal", data_path)
    case.task.kind = "binary_classification"
    case.task.prediction_unit = "compound"
    case.features.include = ["x"]
    case.entities = {
        "compound": EntitySpec(
            id_column="compound_id",
            identity_conflict_policy="exclude",
        )
    }
    case.generalization_scenarios = [
        ScenarioSpec(
            name="external",
            strategy="supplied",
            split_column="split",
            group_column="compound_id",
        )
    ]
    case.evaluation.seeds = [11]
    case.evaluation.primary_metric = "balanced_accuracy"
    case.evaluation.bootstrap_unit = "row_id"
    case.holdout.enabled = True

    def unexpected_feature_construction(*_args, **_kwargs):
        pytest.fail("feature construction ran with an unresolved holdout identity conflict")

    monkeypatch.setattr(runner_module, "build_feature_frames", unexpected_feature_construction)

    with pytest.raises(
        ValueError,
        match="evaluation_partition:external:seed_11:identity_conflict:compound",
    ):
        runner_module.run_case(case, tmp_path / "report", model_allowlist={"dummy"})
