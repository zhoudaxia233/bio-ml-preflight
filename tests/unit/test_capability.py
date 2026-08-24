import pandas as pd

from bio_ml_preflight.cli import synthetic_case
from bio_ml_preflight.evaluation.capability import capability_matrix


def test_random_success_and_cold_failure_is_contradicted(tmp_path) -> None:
    case = synthetic_case("leakage", tmp_path / "x.parquet")
    experiments = pd.DataFrame(
        [
            {
                "scenario": "random_row",
                "strategy": "random",
                "model": "ridge",
                "permuted": False,
                "spearman": 0.95,
            },
            {
                "scenario": "random_row",
                "strategy": "random",
                "model": "ridge",
                "permuted": True,
                "spearman": 0.0,
            },
            {
                "scenario": "cold_entity",
                "strategy": "group",
                "model": "ridge",
                "permuted": False,
                "spearman": 0.02,
            },
            {
                "scenario": "cold_entity",
                "strategy": "group",
                "model": "ridge",
                "permuted": True,
                "spearman": 0.01,
            },
        ]
    )
    matrix = capability_matrix(experiments, case, {})
    assert (
        next(row for row in matrix if row["claim_or_scenario"] == "cold_entity")["status"]
        == "CONTRADICTED"
    )
