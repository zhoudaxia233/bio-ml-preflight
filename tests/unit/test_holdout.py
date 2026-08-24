from pathlib import Path

import pandas as pd
import pytest

from bio_ml_preflight.cli import synthetic_case
from bio_ml_preflight.contracts.case import ScenarioSpec
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
