import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

import bio_ml_preflight.autoprobe.core as autoprobe
import bio_ml_preflight.runner as runner
from bio_ml_preflight.cli import synthetic_case
from bio_ml_preflight.contracts import save_case
from bio_ml_preflight.contracts.case import ScenarioSpec
from bio_ml_preflight.data.synthetic import generate_synthetic


def _prepare(tmp_path, metric="spearman"):
    case = synthetic_case("no_signal", tmp_path / "unused.parquet")
    case.evaluation.primary_metric = metric
    if metric == "log_loss":
        case.task.kind = "binary_classification"
    case.task.higher_is_better = False  # Outcome ranking must not reverse metric quality.
    case.generalization_scenarios = [
        ScenarioSpec(name="scenario-0", strategy="random"),
        ScenarioSpec(name="scenario-1", strategy="group", group_column="group_id"),
    ]
    source = tmp_path / "source.yaml"
    save_case(case, source)
    return autoprobe.prepare_run(source, tmp_path / "probe")


def _mock_reports(monkeypatch, reports):
    reports = iter(reports)

    def run_case(case, output, *, budget, model_allowlist):
        model = next(iter(model_allowlist))
        medians = next(reports)
        report = {
            "experiment_summary": [
                {
                    "scenario": f"scenario-{index}",
                    "representation": "tabular",
                    "model": model,
                    "permuted": False,
                    "standard_deviation": 0.01,
                    "runs": 2,
                    "finite_runs": 2,
                    **({"median": value} if value != "missing" else {}),
                }
                for index, value in enumerate(medians)
            ],
            "ranking_stability": {"top_k": {}},
            "audits": {"leakage": {"suspicious_identifier_features": []}},
            "provenance": {"runtime_seconds": 0.1},
        }
        output.mkdir(parents=True)
        (output / "report.json").write_text(json.dumps(report), encoding="utf-8")
        return report

    monkeypatch.setattr(autoprobe, "run_case", run_case)


def _evaluate(candidate_path, candidate_id):
    candidate = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
    candidate["candidate_id"] = candidate_id
    candidate_path.write_text(yaml.safe_dump(candidate), encoding="utf-8")
    return autoprobe.evaluate_candidate(candidate_path)


@pytest.mark.parametrize(
    "metric,values",
    [
        ("rmse", [1.5, 1.0, 2.0, 1.2]),
        ("mae", [1.5, 1.0, 2.0, 1.2]),
        ("log_loss", [1.5, 1.0, 2.0, 1.2]),
        ("spearman", [0.3, 0.6, 0.2, 0.4]),
    ],
)
def test_candidate_quality_uses_metric_direction_and_latest_accepted_baseline(
    tmp_path, monkeypatch, metric, values
) -> None:
    candidate_path = _prepare(tmp_path, metric)
    _mock_reports(monkeypatch, [[value, value] for value in values])

    vectors = [_evaluate(candidate_path, f"candidate-{i}") for i in range(len(values))]

    assert [vector["decision"] for vector in vectors] == ["KEEP", "KEEP", "DISCARD", "DISCARD"]
    assert [vector["median_primary_development_metric"] for vector in vectors] == values
    assert [vector["worst_scenario_metric"] for vector in vectors] == values
    persisted = [
        json.loads(line)
        for line in (candidate_path.parent / "experiments.jsonl").read_text().splitlines()
    ]
    assert persisted == vectors
    for index, vector in enumerate(vectors):
        result_dir = candidate_path.parent / "results" / f"candidate-{index}"
        assert json.loads((result_dir / "evaluation-vector.json").read_text()) == vector
        report = json.loads((result_dir / "report.json").read_text())
        assert report["experiment_summary"][0]["median"] == values[index]


@pytest.mark.parametrize(
    "metric,baseline,challenger,raw_worst",
    [
        ("rmse", [1.0, 1.0], [0.5, 1.1], 1.1),
        ("mae", [1.0, 1.0], [0.5, 1.1], 1.1),
        ("log_loss", [1.0, 1.0], [0.5, 1.1], 1.1),
        ("spearman", [0.5, 0.5], [0.8, 0.4], 0.4),
    ],
)
def test_better_median_cannot_hide_worse_scenario(
    tmp_path, monkeypatch, metric, baseline, challenger, raw_worst
) -> None:
    candidate_path = _prepare(tmp_path, metric)
    _mock_reports(monkeypatch, [baseline, challenger])

    assert _evaluate(candidate_path, "baseline")["decision"] == "KEEP"
    vector = _evaluate(candidate_path, "challenger")

    assert vector["decision"] == "DISCARD"
    assert vector["worst_scenario_metric"] == raw_worst
    assert any("worst" in reason.lower() for reason in vector["decision_reasons"])


@pytest.mark.parametrize(
    "medians",
    [
        [],
        [None, None],
        [float("nan"), float("nan")],
        [float("inf"), float("inf")],
        [-float("inf"), -float("inf")],
        [0.5, None],
        [0.5, float("nan")],
        [0.5, "missing"],
        [0.5],
    ],
    ids=[
        "absent",
        "null",
        "nan",
        "inf",
        "negative-inf",
        "partial-null",
        "partial-nan",
        "missing-median",
        "missing-scenario",
    ],
)
def test_invalid_primary_evidence_cannot_establish_baseline(tmp_path, monkeypatch, medians) -> None:
    candidate_path = _prepare(tmp_path)
    _mock_reports(monkeypatch, [medians, [0.4, 0.4]])

    invalid = _evaluate(candidate_path, "invalid")
    valid = _evaluate(candidate_path, "valid")

    assert invalid["decision"] == "DISCARD"
    assert invalid["decision_reasons"]
    assert valid["decision"] == "KEEP"


@pytest.mark.parametrize("last_metric", [None, float("inf"), "absent", 0.6])
def test_every_scheduled_seed_needs_a_finite_primary_metric(
    tmp_path, monkeypatch, last_metric
) -> None:
    candidate_path = _prepare(tmp_path)
    candidate = yaml.safe_load(candidate_path.read_text())
    candidate["budget"] = "standard"
    candidate_path.write_text(yaml.safe_dump(candidate))

    def run_case(case, output, *, budget, model_allowlist):
        model = next(iter(model_allowlist))
        rows = [
            {
                "scenario": scenario.name,
                "representation": "declared_features",
                "model": model,
                "permuted": False,
                "seed": seed,
                "spearman": metric,
            }
            for scenario in case.generalization_scenarios
            for seed, metric in zip(case.evaluation.seeds, [0.4, 0.5, last_metric], strict=True)
            if metric != "absent"
        ]
        output.mkdir(parents=True)
        return {
            "experiment_summary": runner._experiment_summary(pd.DataFrame(rows), "spearman"),
            "ranking_stability": {},
            "audits": {"leakage": {"suspicious_identifier_features": []}},
            "provenance": {"runtime_seconds": 0.1},
            "holdout_access": [],
        }

    monkeypatch.setattr(autoprobe, "run_case", run_case)
    result = autoprobe.evaluate_candidate(candidate_path)

    assert result["decision"] == ("KEEP" if last_metric == 0.6 else "DISCARD")


def _block_data_access(monkeypatch):
    def unexpected_access(*args, **kwargs):
        pytest.fail("Autoprobe reached data access or evaluation before the protocol guard")

    monkeypatch.setattr(autoprobe, "run_case", unexpected_access)
    monkeypatch.setattr(runner, "file_checksum", unexpected_access)
    monkeypatch.setattr(runner, "read_table", unexpected_access)


def _output_snapshot(output):
    return {path: path.read_bytes() if path.is_file() else None for path in output.rglob("*")}


@pytest.mark.parametrize("existing_output", [False, True])
def test_prepare_rejects_holdout_before_output_changes(
    tmp_path, monkeypatch, existing_output
) -> None:
    case = synthetic_case("no_signal", tmp_path / "never-read.parquet")
    case.holdout.enabled = True
    source = tmp_path / "holdout.yaml"
    save_case(case, source)
    output = tmp_path / "probe"
    if existing_output:
        output.mkdir()
        (output / "case.yaml").write_text("preserve existing run", encoding="utf-8")
    before = _output_snapshot(output)
    _block_data_access(monkeypatch)

    with pytest.raises(PermissionError, match="(?i)holdout"):
        autoprobe.prepare_run(source, output)

    assert output.exists() == existing_output
    assert _output_snapshot(output) == before


@pytest.mark.parametrize("redirect_candidate", [False, True])
def test_evaluate_rechecks_holdout_before_output_changes(
    tmp_path, monkeypatch, redirect_candidate
) -> None:
    candidate_path = _prepare(tmp_path)
    candidate = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
    case_path = Path(candidate["case_path"])
    case = autoprobe.load_case(case_path)
    case.holdout.enabled = True
    if redirect_candidate:
        case_path = tmp_path / "other-holdout.yaml"
        candidate["case_path"] = str(case_path)
        candidate_path.write_text(yaml.safe_dump(candidate), encoding="utf-8")
    save_case(case, case_path)
    output = candidate_path.parent
    before = _output_snapshot(output)
    _block_data_access(monkeypatch)

    with pytest.raises(PermissionError, match="(?i)holdout"):
        autoprobe.evaluate_candidate(candidate_path)

    assert _output_snapshot(output) == before


@pytest.mark.parametrize("edit", ["metric", "threshold"])
def test_frozen_case_edits_rejected_before_evaluation(tmp_path, monkeypatch, edit) -> None:
    candidate_path = _prepare(tmp_path)
    case_path = candidate_path.parent / "case.yaml"
    case = autoprobe.load_case(case_path)
    if edit == "metric":
        case.evaluation.primary_metric = "rmse"
    else:
        case.thresholds.supported_metric += 0.1
    save_case(case, case_path)
    before = _output_snapshot(candidate_path.parent)
    _block_data_access(monkeypatch)

    with pytest.raises(ValueError, match="(?i)case|protocol|frozen|locked"):
        autoprobe.evaluate_candidate(candidate_path)

    assert _output_snapshot(candidate_path.parent) == before


def test_candidate_budget_change_rejected_before_evaluation(tmp_path, monkeypatch) -> None:
    candidate_path = _prepare(tmp_path)
    _mock_reports(monkeypatch, [[0.5, 0.5]])
    assert _evaluate(candidate_path, "baseline")["decision"] == "KEEP"
    candidate = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
    candidate.update(candidate_id="more-seeds", budget="standard")
    candidate_path.write_text(yaml.safe_dump(candidate), encoding="utf-8")
    before = _output_snapshot(candidate_path.parent)
    _block_data_access(monkeypatch)

    with pytest.raises(ValueError, match="(?i)budget|protocol|frozen|locked"):
        autoprobe.evaluate_candidate(candidate_path)

    assert _output_snapshot(candidate_path.parent) == before


@pytest.mark.parametrize(
    "field",
    [
        "minimum_delta",
        "maximum_runtime_seconds",
        "maximum_worst_scenario_regression",
        "maximum_stability_regression",
    ],
)
@pytest.mark.parametrize("value", [-0.01, float("nan"), float("inf"), -float("inf")])
def test_invalid_candidate_tolerances_rejected_before_evaluation(
    tmp_path, monkeypatch, field, value
) -> None:
    candidate_path = _prepare(tmp_path)
    candidate = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
    candidate[field] = value
    candidate_path.write_text(yaml.safe_dump(candidate), encoding="utf-8")
    before = _output_snapshot(candidate_path.parent)
    _block_data_access(monkeypatch)

    with pytest.raises(ValidationError, match=field):
        autoprobe.evaluate_candidate(candidate_path)

    assert _output_snapshot(candidate_path.parent) == before


def test_real_development_workflow_writes_evidence_without_holdout(tmp_path) -> None:
    data_path = tmp_path / "development.parquet"
    generate_synthetic("no_signal", data_path, seed=94013)
    case = synthetic_case("no_signal", data_path)
    case.evaluation.primary_metric = "rmse"
    case.evaluation.seeds = [613]
    case.evaluation.permutation_draws = 1
    case.generalization_scenarios = [case.generalization_scenarios[1]]
    source = tmp_path / "source.yaml"
    save_case(case, source)
    candidate_path = autoprobe.prepare_run(source, tmp_path / "probe")

    vector = autoprobe.evaluate_candidate(candidate_path)

    assert vector["decision"] == "KEEP"
    assert vector["holdout_accessed"] is False
    assert vector["median_primary_development_metric"] > 0
    result_dir = candidate_path.parent / "results" / vector["candidate_id"]
    report = json.loads((result_dir / "report.json").read_text())
    assert not report["holdout_access"]
    assert list((result_dir / "split_manifests").glob("*.json"))
    assert list((result_dir / "predictions").glob("*.parquet"))
