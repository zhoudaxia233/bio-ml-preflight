import numpy as np
import pandas as pd

from bio_ml_preflight.audits import audit_overlap
from bio_ml_preflight.cli import synthetic_case
from bio_ml_preflight.contracts.case import ScenarioSpec
from bio_ml_preflight.data.synthetic import generate_synthetic
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
