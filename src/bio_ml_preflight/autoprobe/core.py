from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from bio_ml_preflight.contracts import load_case, save_case
from bio_ml_preflight.runner import run_case


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    candidate_id: str
    case_path: str
    model: Literal["dummy", "elastic_net", "extra_trees", "logistic"]
    budget: Literal["smoke", "standard"] = "smoke"
    complexity: dict[str, float | int | str] = Field(default_factory=dict)
    minimum_delta: float = 0.01
    maximum_runtime_seconds: float = 300.0
    maximum_worst_scenario_regression: float = 0.02
    maximum_stability_regression: float = 0.05


def prepare_run(case_path: Path, run_dir: Path) -> Path:
    case = load_case(case_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    locked_case = run_dir / "case.yaml"
    save_case(case, locked_case, relative_data_path=case.data.path)
    candidate_path = run_dir / "candidate.yaml"
    candidate = Candidate(
        candidate_id="baseline-elastic-net",
        case_path=str(locked_case.resolve()),
        model="logistic" if case.task.kind == "binary_classification" else "elastic_net",
        complexity={"family": "linear", "tunable_parameters": 1},
    )
    candidate_path.write_text(
        yaml.safe_dump(candidate.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )
    policy = {
        "immutable": ["case.yaml", "split_manifests", "evaluator", "metrics", "holdout policy"],
        "mutable": ["candidate.yaml"],
        "final_holdout_access": False,
        "selection": "guardrails, primary delta, worst scenario, stability, runtime, simplicity",
    }
    (run_dir / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    return candidate_path


def evaluate_candidate(candidate_path: Path) -> dict[str, Any]:
    candidate = Candidate.model_validate(yaml.safe_load(candidate_path.read_text(encoding="utf-8")))
    case = load_case(Path(candidate.case_path))
    run_dir = candidate_path.parent
    result_dir = run_dir / "results" / candidate.candidate_id
    report = run_case(
        case,
        result_dir,
        budget=candidate.budget,
        model_allowlist={candidate.model},
    )
    summaries = [
        row
        for row in report["experiment_summary"]
        if row["model"] == candidate.model and row["permuted"] is False
    ]
    medians = [row["median"] for row in summaries if row["median"] is not None]
    dispersions = [
        row["standard_deviation"] for row in summaries if row["standard_deviation"] is not None
    ]
    top_k = report["ranking_stability"].get("top_k", {})
    first_k = top_k.get(str(min(case.decision.k)), {}) if case.decision.k else {}
    guardrails = []
    suspicious = report["audits"]["leakage"]["suspicious_identifier_features"]
    if suspicious:
        guardrails.append(f"identifier-like configured features: {suspicious}")
    vector = {
        "candidate_id": candidate.candidate_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "median_primary_development_metric": _median(medians),
        "worst_scenario_metric": min(medians) if medians else None,
        "seed_or_split_dispersion": max(dispersions) if dispersions else None,
        "top_k_stability": first_k.get("average_pairwise_jaccard"),
        "runtime_seconds": report["provenance"]["runtime_seconds"],
        "complexity": candidate.complexity,
        "guardrail_failures": guardrails,
        "holdout_accessed": False,
    }
    history_path = run_dir / "experiments.jsonl"
    history = (
        [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line]
        if history_path.exists()
        else []
    )
    baseline = next((entry for entry in history if entry.get("decision") == "KEEP"), None)
    decision, reasons = _select(vector, baseline, candidate)
    vector["decision"] = decision
    vector["decision_reasons"] = reasons
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(vector, sort_keys=True) + "\n")
    (result_dir / "evaluation-vector.json").write_text(
        json.dumps(vector, indent=2, sort_keys=True), encoding="utf-8"
    )
    return vector


def _select(
    vector: dict[str, Any], baseline: dict[str, Any] | None, candidate: Candidate
) -> tuple[str, list[str]]:
    if vector["guardrail_failures"]:
        return "DISCARD", ["validity/leakage guardrail failed"]
    if vector["runtime_seconds"] > candidate.maximum_runtime_seconds:
        return "DISCARD", ["runtime budget exceeded"]
    if baseline is None:
        return "KEEP", ["first valid candidate establishes the baseline"]
    current = vector["median_primary_development_metric"]
    previous = baseline["median_primary_development_metric"]
    if current is None or previous is None or current < previous + candidate.minimum_delta:
        return "DISCARD", ["primary metric did not improve by the configured minimum delta"]
    worst = vector["worst_scenario_metric"]
    previous_worst = baseline["worst_scenario_metric"]
    if (
        worst is not None
        and previous_worst is not None
        and worst < previous_worst - candidate.maximum_worst_scenario_regression
    ):
        return "DISCARD", ["worst-scenario performance materially regressed"]
    stability = vector["top_k_stability"]
    previous_stability = baseline["top_k_stability"]
    if (
        stability is not None
        and previous_stability is not None
        and stability < previous_stability - candidate.maximum_stability_regression
    ):
        return "DISCARD", ["ranking stability materially regressed"]
    return "KEEP", ["lexicographic scientific guardrails and improvement rules passed"]


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def copy_candidate(source: Path, destination: Path) -> None:
    shutil.copy2(source, destination)
