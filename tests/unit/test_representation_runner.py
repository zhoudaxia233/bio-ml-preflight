from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import bio_ml_preflight.runner as runner_module
from bio_ml_preflight.cli import synthetic_case
from bio_ml_preflight.contracts.case import EntitySpec, EvaluationSpec, ScenarioSpec
from bio_ml_preflight.data.synthetic import generate_synthetic


def test_representations_reuse_one_split_manifest_and_model_suite(
    tmp_path: Path, monkeypatch
) -> None:
    data_path = tmp_path / "no-signal.parquet"
    frame = generate_synthetic("no_signal", data_path)
    case = synthetic_case("no_signal", data_path)
    case.generalization_scenarios = [ScenarioSpec(name="random", strategy="random")]
    case.evaluation = EvaluationSpec(
        seeds=[11],
        primary_metric="spearman",
        permutation_draws=1,
        model_allowlist=["elastic_net"],
    )
    split_calls = []
    create_split = runner_module.create_split

    def counted_split(split_frame, scenario, seed):
        split_calls.append((scenario.name, seed))
        return create_split(split_frame, scenario, seed)

    monkeypatch.setattr(runner_module, "create_split", counted_split)
    monkeypatch.setattr(
        runner_module,
        "build_feature_frames",
        lambda _frame, _case: {
            "first": frame[["x1"]].reset_index(drop=True),
            "second": frame[["x2"]].reset_index(drop=True),
        },
    )

    output = tmp_path / "report"
    result = runner_module.run_case(case, output)

    assert split_calls == [("random", 11)]
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
    assert result["pre_model_readiness"]["model_fitting_allowed"] is True
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "Pre-model readiness" in report


def test_blocked_readiness_stops_before_feature_construction(tmp_path: Path, monkeypatch) -> None:
    data_path = tmp_path / "missing-identity.parquet"
    pd.DataFrame(
        {
            "group_id": ["a", "b", None, None, "e", "f"],
            "x1": [0.0, 1.0, 0.2, 0.8, 0.1, 0.9],
            "x2": [1.0, 0.0, 0.8, 0.2, 0.9, 0.1],
            "y": [0.0, 1.0, 0.0, 1.0, 0.1, 0.9],
        }
    ).to_parquet(data_path, index=False)
    case = synthetic_case("no_signal", data_path)
    case.task.prediction_unit = "group"
    case.entities["group"].identity_conflict_policy = "exclude"

    def unexpected_feature_construction(*_args, **_kwargs):
        pytest.fail("feature construction ran after a blocked readiness decision")

    monkeypatch.setattr(runner_module, "build_feature_frames", unexpected_feature_construction)
    monkeypatch.setattr(runner_module, "build_probe_suite", unexpected_feature_construction)
    output = tmp_path / "report"

    with pytest.raises(ValueError, match="Pre-model readiness gate BLOCKED"):
        runner_module.run_case(case, output)

    assert not output.exists()


def test_every_row_used_by_any_supplied_split_is_audited_before_features(
    tmp_path: Path, monkeypatch
) -> None:
    data_path = tmp_path / "multiple-splits.parquet"
    pd.DataFrame(
        {
            "group_id": ["a", "b", "c", "d", "e", "f"],
            "x1": [np.inf, 1.0, 0.2, 0.8, 0.1, 0.9],
            "x2": [1.0, 0.0, 0.8, 0.2, 0.9, 0.1],
            "y": [0.0, 1.0, 0.2, 0.8, 0.1, 0.9],
            "split_a": ["train", "train", "train", "test", "test", "test"],
            "split_b": ["test", "train", "test", "train", "train", "test"],
        }
    ).to_parquet(data_path, index=False)
    case = synthetic_case("no_signal", data_path)
    case.generalization_scenarios = [
        ScenarioSpec(name="a", strategy="supplied", split_column="split_a"),
        ScenarioSpec(name="b", strategy="supplied", split_column="split_b"),
    ]

    def unexpected_model_setup(*_args, **_kwargs):
        pytest.fail("model setup ran after a blocked readiness decision")

    monkeypatch.setattr(runner_module, "build_feature_frames", unexpected_model_setup)
    monkeypatch.setattr(runner_module, "build_probe_suite", unexpected_model_setup)

    with pytest.raises(ValueError, match="non_finite_modeled_features"):
        runner_module.run_case(case, tmp_path / "report")


def test_supplied_split_exclusions_are_not_feature_built_or_predicted(
    tmp_path: Path, monkeypatch
) -> None:
    build_feature_frames = runner_module.build_feature_frames

    def checked_feature_construction(modeling_frame, model_case):
        assert list(modeling_frame.index) == [0, 1, 3, 4, 5, 6]
        return build_feature_frames(modeling_frame, model_case)

    monkeypatch.setattr(runner_module, "build_feature_frames", checked_feature_construction)

    def run(ignored_target: float, name: str):
        data_path = tmp_path / f"{name}.parquet"
        pd.DataFrame(
            {
                "group_id": ["a", "b", "a", "c", "d", "e", "f"],
                "x1": [0.0, 0.2, np.inf, 0.6, 0.8, 1.0, 1.2],
                "x2": [1.2, 1.0, np.inf, 0.6, 0.4, 0.2, 0.0],
                "y": [0.0, 0.2, ignored_target, 0.6, 0.8, 1.0, 1.2],
                "split": ["train", "train", "ignore", "train", "train", "test", "test"],
            }
        ).to_parquet(data_path, index=False)
        case = synthetic_case("no_signal", data_path)
        case.task.prediction_unit = "group"
        case.generalization_scenarios = [
            ScenarioSpec(name="supplied", strategy="supplied", split_column="split")
        ]
        case.evaluation = EvaluationSpec(
            seeds=[11],
            primary_metric="spearman",
            bootstrap_unit="group_id",
            permutation_draws=1,
        )
        output = tmp_path / name
        result = runner_module.run_case(case, output, model_allowlist={"elastic_net"})
        prediction = pd.read_parquet(
            output / "predictions" / "supplied__declared_features__elastic_net__seed-11.parquet"
        )
        assert set(prediction["_row_id"]) == {0, 1, 3, 4, 5, 6}
        return result

    matching = run(0.0, "matching-ignored-target")
    conflicting = run(99.0, "conflicting-ignored-target")

    assert matching["capability_matrix"] == conflicting["capability_matrix"]
    assert conflicting["audits"]["inventory"]["source_rows_after_identity_policy"] == 7
    assert conflicting["audits"]["inventory"]["manifest_excluded_rows"] == 1
    assert (
        conflicting["audits"]["independence"]["entities"]["group"]["conflicting_target_entities"]
        == 0
    )


def test_mutating_identity_policy_requires_one_supplied_training_boundary(
    tmp_path: Path, monkeypatch
) -> None:
    data_path = tmp_path / "conflicting-supplied-boundaries.parquet"
    pd.DataFrame(
        {
            "group_id": ["q", "q", "r", "s"],
            "x1": [0.0, 1.0, 0.2, 0.8],
            "x2": [1.0, 0.0, 0.8, 0.2],
            "y": [0.0, 1.0, 0.2, 0.8],
            "split_a": ["train", "test", "train", "test"],
            "split_b": ["test", "train", "test", "train"],
        }
    ).to_parquet(data_path, index=False)
    case = synthetic_case("no_signal", data_path)
    case.task.prediction_unit = "group"
    case.entities["group"].identity_conflict_policy = "exclude"
    case.generalization_scenarios = [
        ScenarioSpec(name="a", strategy="supplied", split_column="split_a"),
        ScenarioSpec(name="b", strategy="supplied", split_column="split_b"),
    ]

    def unexpected_feature_construction(*_args, **_kwargs):
        pytest.fail("feature construction ran with ambiguous identity-policy scope")

    monkeypatch.setattr(runner_module, "build_feature_frames", unexpected_feature_construction)

    with pytest.raises(ValueError, match="consistent training boundary"):
        runner_module.run_case(case, tmp_path / "report")


def test_each_training_partition_must_be_ready_before_features(tmp_path: Path, monkeypatch) -> None:
    data_path = tmp_path / "one-class-splits.parquet"
    pd.DataFrame(
        {
            "group_id": ["a", "b", "c", "d", "e", "f"],
            "x1": [0.0, 0.1, 0.2, 0.8, 0.9, 1.0],
            "x2": [1.0, 0.9, 0.8, 0.2, 0.1, 0.0],
            "y": [0, 0, 0, 1, 1, 1],
            "split_a": ["train", "train", "train", "test", "test", "test"],
            "split_b": ["test", "test", "test", "train", "train", "train"],
        }
    ).to_parquet(data_path, index=False)
    case = synthetic_case("no_signal", data_path)
    case.task.kind = "binary_classification"
    case.generalization_scenarios = [
        ScenarioSpec(name="a", strategy="supplied", split_column="split_a"),
        ScenarioSpec(name="b", strategy="supplied", split_column="split_b"),
    ]
    case.evaluation.seeds = [11]

    def unexpected_model_setup(*_args, **_kwargs):
        pytest.fail("model setup ran after a blocked training partition")

    monkeypatch.setattr(runner_module, "build_feature_frames", unexpected_model_setup)
    monkeypatch.setattr(runner_module, "build_probe_suite", unexpected_model_setup)

    with pytest.raises(
        ValueError,
        match="training_partition:a:seed_11:invalid_binary_target_support",
    ):
        runner_module.run_case(case, tmp_path / "report")


def test_readiness_is_recomputed_after_identity_exclusion(tmp_path: Path, monkeypatch) -> None:
    data_path = tmp_path / "identity-exclusion.parquet"
    pd.DataFrame(
        {
            "group_id": ["a", "a", "b", "c"],
            "x1": [1.0, 2.0, np.nan, np.nan],
            "y": [0, 1, 0, 0],
        }
    ).to_parquet(data_path, index=False)
    case = synthetic_case("no_signal", data_path)
    case.task.kind = "binary_classification"
    case.task.prediction_unit = "group"
    case.features.include = ["x1"]
    case.entities = {"group": EntitySpec(id_column="group_id", identity_conflict_policy="exclude")}
    case.generalization_scenarios = [ScenarioSpec(name="random", strategy="random")]
    case.evaluation = EvaluationSpec(seeds=[11], bootstrap_unit="group_id")

    def unexpected_model_setup(*_args, **_kwargs):
        pytest.fail("model setup ran after identity exclusion invalidated readiness")

    monkeypatch.setattr(runner_module, "build_feature_frames", unexpected_model_setup)
    monkeypatch.setattr(runner_module, "build_probe_suite", unexpected_model_setup)

    with pytest.raises(ValueError, match="no_observed_modeled_features"):
        runner_module.run_case(case, tmp_path / "report")


def test_identity_exclusion_can_remove_a_blocking_non_finite_feature(tmp_path: Path) -> None:
    data_path = tmp_path / "resolvable-identity.parquet"
    pd.DataFrame(
        {
            "group_id": ["a", "a", "b", "c", "d", "e", "f", "g", "h", "i"],
            "x": [np.inf, np.inf, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4],
            "y": [0.0, 1.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
        }
    ).to_parquet(data_path, index=False)
    case = synthetic_case("no_signal", data_path)
    case.task.prediction_unit = "group"
    case.features.include = ["x"]
    case.entities = {"group": EntitySpec(id_column="group_id", identity_conflict_policy="exclude")}
    case.generalization_scenarios = [ScenarioSpec(name="random", strategy="random")]
    case.evaluation = EvaluationSpec(
        seeds=[11],
        primary_metric="spearman",
        bootstrap_unit="group_id",
        permutation_draws=1,
    )

    result = runner_module.run_case(case, tmp_path / "report", model_allowlist={"dummy"})

    assert result["audits"]["identity_consistency"]["status"] == "RESOLVED"
    assert result["audits"]["identity_consistency"]["model_rows"] == 8
    assert result["pre_model_readiness"]["model_fitting_allowed"] is True
    assert not any(
        finding["code"].endswith("non_finite_modeled_features")
        for finding in result["pre_model_findings"]
    )


def test_partition_limits_are_preserved_in_run_readiness(tmp_path: Path) -> None:
    data_path = tmp_path / "partition-limits.parquet"
    pd.DataFrame(
        {
            "x": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            "y": [0.0, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 1.0],
            "replicate_id": ["r1", "r1", "r2", "r2", "r3", "r3", "r4", "r4"],
            "batch_id": ["b1", "b2", "b1", "b2", "b1", "b2", "b1", "b2"],
            "split_a": ["train"] * 4 + ["test"] * 4,
            "split_b": ["test"] * 4 + ["train"] * 4,
        }
    ).to_parquet(data_path, index=False)
    case = synthetic_case("no_signal", data_path)
    case.entities = {}
    case.features.include = ["x"]
    case.metadata.replicate_id = "replicate_id"
    case.metadata.batch_id = "batch_id"
    case.generalization_scenarios = [
        ScenarioSpec(name="a", strategy="supplied", split_column="split_a"),
        ScenarioSpec(name="b", strategy="supplied", split_column="split_b"),
    ]
    case.evaluation = EvaluationSpec(
        seeds=[11],
        primary_metric="spearman",
        bootstrap_unit="replicate_id",
        permutation_draws=1,
    )

    result = runner_module.run_case(case, tmp_path / "report", model_allowlist={"dummy"})

    assert result["pre_model_readiness"]["status"] == "READY_WITH_LIMITS"
    assert {
        "training_partition:a:seed_11:uninformative_modeled_features",
        "training_partition:b:seed_11:uninformative_modeled_features",
    } <= set(result["pre_model_readiness"]["limiting_checks"])


def test_learning_curve_skips_a_one_class_training_subset(tmp_path: Path) -> None:
    groups = np.array([f"g{index}" for index in range(8)])
    order = np.random.default_rng(11).permutation(groups)
    labels = {group: int(index >= 3) for index, group in enumerate(order)}
    frame = pd.DataFrame(
        {
            "group_id": [*groups, "test-0", "test-1"],
            "x": np.arange(10, dtype=float),
            "y": [*[labels[group] for group in groups], 0, 1],
        }
    )
    case = synthetic_case("no_signal", tmp_path / "unused.parquet")
    case.task.kind = "binary_classification"
    case.task.prediction_unit = "group"
    case.features.include = ["x"]
    case.evaluation.primary_metric = "balanced_accuracy"
    train_indices = np.arange(8, dtype=np.int64)
    test_indices = np.array([8, 9], dtype=np.int64)

    rows = runner_module._learning_curve(
        frame,
        frame[["x"]],
        frame["y"].to_numpy(),
        train_indices,
        test_indices,
        "group_id",
        "grouped",
        "declared_features",
        case,
        "smoke",
        11,
        {"logistic"},
    )

    assert [row["group_count"] for row in rows] == [5, 8]


def test_declared_bootstrap_unit_must_exist_before_feature_construction(
    tmp_path: Path, monkeypatch
) -> None:
    data_path = tmp_path / "missing-bootstrap.parquet"
    generate_synthetic("no_signal", data_path)
    case = synthetic_case("no_signal", data_path)
    case.evaluation.bootstrap_unit = "missing_unit"

    def unexpected_feature_construction(*_args, **_kwargs):
        pytest.fail("feature construction ran with a missing bootstrap unit")

    monkeypatch.setattr(runner_module, "build_feature_frames", unexpected_feature_construction)

    with pytest.raises(ValueError, match="missing_unit"):
        runner_module.run_case(case, tmp_path / "report")


def test_declared_metadata_must_exist_before_feature_construction(
    tmp_path: Path, monkeypatch
) -> None:
    data_path = tmp_path / "missing-metadata.parquet"
    generate_synthetic("no_signal", data_path)
    case = synthetic_case("no_signal", data_path)
    case.metadata.batch_id = "missing_batch"

    def unexpected_feature_construction(*_args, **_kwargs):
        pytest.fail("feature construction ran with missing declared metadata")

    monkeypatch.setattr(runner_module, "build_feature_frames", unexpected_feature_construction)

    with pytest.raises(ValueError, match="missing_batch"):
        runner_module.run_case(case, tmp_path / "report")
