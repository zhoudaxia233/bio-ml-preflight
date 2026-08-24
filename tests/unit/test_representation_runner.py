from pathlib import Path

import pandas as pd

import bio_ml_preflight.runner as runner_module
from bio_ml_preflight.cli import synthetic_case
from bio_ml_preflight.contracts.case import EvaluationSpec, ScenarioSpec
from bio_ml_preflight.data.synthetic import generate_synthetic


def test_representations_reuse_one_split_manifest_and_model_suite(
    tmp_path: Path, monkeypatch
) -> None:
    data_path = tmp_path / "no-signal.parquet"
    frame = generate_synthetic("no_signal", data_path)
    case = synthetic_case("no_signal", data_path)
    case.generalization_scenarios = [ScenarioSpec(name="random", strategy="random")]
    case.evaluation = EvaluationSpec(seeds=[11], primary_metric="spearman", permutation_draws=1)
    monkeypatch.setattr(
        runner_module,
        "build_feature_frames",
        lambda _frame, _case: {
            "first": frame[["x1"]].reset_index(drop=True),
            "second": frame[["x2"]].reset_index(drop=True),
        },
    )

    output = tmp_path / "report"
    result = runner_module.run_case(case, output, model_allowlist={"elastic_net"})

    assert len(list((output / "split_manifests").glob("*.json"))) == 1
    predictions = [pd.read_parquet(path) for path in (output / "predictions").glob("*.parquet")]
    assert {table["representation"].iat[0] for table in predictions} == {"first", "second"}
    test_rows = [set(table.loc[table["is_test"], "_row_id"]) for table in predictions]
    assert test_rows[0] == test_rows[1]
    experiments = pd.read_parquet(output / "aggregate_experiments.parquet")
    assert set(experiments["representation"]) == {"first", "second"}
    suites = experiments.groupby("representation")["model"].apply(set)
    assert all(models == {"elastic_net"} for models in suites)
    sensitivity = result["representation_sensitivity"][0]
    assert set(sensitivity["matched_model_medians"]["elastic_net"]) == {"first", "second"}
