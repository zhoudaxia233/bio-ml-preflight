"""A synthetic biological structure verifies wiring, not clinical performance."""

import numpy as np
import pandas as pd
import pytest

from bio_ml_preflight.contracts import CaseSpec
from bio_ml_preflight.runner import run_case


@pytest.mark.parametrize("unit", ["patient_id", None])
def test_visit_outcomes_record_actual_split_and_null_design(tmp_path, unit):
    rng = np.random.default_rng(16)
    patients = np.repeat(np.arange(20), 6)
    outcomes = np.tile([0, 0, 0, 1, 1, 1], 20)
    frame = pd.DataFrame(
        {
            "patient_id": patients,
            "partition": np.where(patients < 14, "train", "test"),
            "measurement": outcomes + rng.normal(size=len(patients)),
            "outcome": outcomes,
        }
    )
    path = tmp_path / "visits.parquet"
    frame.to_parquet(path, index=False)
    case = CaseSpec.model_validate(
        {
            "case_id": "synthetic-patient-visits",
            "data": {"path": str(path)},
            "task": {
                "kind": "binary_classification",
                "prediction_unit": "visit",
                "target_column": "outcome",
            },
            "entities": {"patient": {"id_column": "patient_id"}},
            "features": {"include": ["measurement"]},
            "generalization_scenarios": [
                {
                    "name": "new_patients",
                    "strategy": "supplied",
                    "split_column": "partition",
                    "group_column": "patient_id",
                    "split_claim": {"kind": "unseen_entity", "entity_column": "patient_id"},
                }
            ],
            "evaluation": {
                "seeds": [11, 23],
                "primary_metric": "balanced_accuracy",
                "bootstrap_unit": unit,
                "model_allowlist": ["dummy", "logistic"],
            },
            "role_confirmation": dict.fromkeys(
                ["target", "prediction_unit", "features", "entities"], True
            ),
        }
    )

    report = run_case(case, tmp_path / "report", budget="smoke")
    runs = pd.read_parquet(tmp_path / "report" / "aggregate_experiments.parquet")
    assert runs.partition_fingerprint.nunique() == 1
    controls = runs[runs.permuted]
    method = "equal_size_group_blocks_within_group_shuffle" if unit else "row"
    assert set(controls.permutation_method) == {method}
    if unit:
        assert set(controls.permutation_unit) == {unit}
    else:
        assert controls.permutation_unit.isna().all()
    overlap = report["split_overlap"]["new_patients:11"]
    assert overlap["test_target_counts"] == ({"0": 6, "1": 6} if unit else {"0": 18, "1": 18})
    assert overlap["entity_overlap"]["patient"]["count"] == 0
    verdict = report["capability_matrix"][0]
    assert verdict["numbers"]["permutation_design"]["method"] == method
    assert verdict["numbers"]["split_variation"]["status"] == "NOT_ASSESSABLE"
    assert "Across-split standard deviation" not in (tmp_path / "report" / "report.md").read_text()
