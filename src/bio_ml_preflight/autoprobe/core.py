from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from bio_ml_preflight.contracts import CaseSpec, load_case, save_case
from bio_ml_preflight.evaluation.metrics import metric_higher_is_better
from bio_ml_preflight.runner import run_case


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    candidate_id: str
    case_path: str
    model: Literal["dummy", "elastic_net", "extra_trees", "logistic"]
    budget: Literal["smoke", "standard"] = "smoke"
    complexity: dict[str, float | int | str] = Field(default_factory=dict)
    minimum_delta: float = Field(default=0.01, ge=0, allow_inf_nan=False)
    maximum_runtime_seconds: float = Field(default=300.0, ge=0, allow_inf_nan=False)
    maximum_worst_scenario_regression: float = Field(default=0.02, ge=0, allow_inf_nan=False)
    maximum_stability_regression: float = Field(default=0.05, ge=0, allow_inf_nan=False)


def _load_development_case(path: Path) -> CaseSpec:
    case = load_case(path)
    if case.holdout.enabled:
        raise PermissionError("Autoprobe cannot access holdouts; use a development-only case.")
    return case


def prepare_run(case_path: Path, run_dir: Path) -> Path:
    case = _load_development_case(case_path)
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
        "case_fingerprint": case.fingerprint(),
        "immutable": ["case.yaml", "split_manifests", "evaluator", "metrics", "holdout policy"],
        "mutable": ["candidate.yaml"],
        "final_holdout_access": False,
        "selection": "validity, runtime, primary delta, worst scenario, ranking stability",
    }
    (run_dir / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    return candidate_path


def evaluate_candidate(candidate_path: Path) -> dict[str, Any]:
    candidate = Candidate.model_validate(yaml.safe_load(candidate_path.read_text(encoding="utf-8")))
    case = _load_development_case(Path(candidate.case_path))
    run_dir = candidate_path.parent
    policy = json.loads((run_dir / "policy.json").read_text(encoding="utf-8"))
    case_fingerprint = case.fingerprint()
    if policy.get("case_fingerprint") != case_fingerprint:
        raise ValueError("Autoprobe requires the frozen case; prepare a new run directory.")
    history_path = run_dir / "experiments.jsonl"
    history = (
        [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line]
        if history_path.exists()
        else []
    )
    if any(
        entry.get("case_fingerprint") != case_fingerprint or entry.get("budget") != candidate.budget
        for entry in history
    ):
        raise ValueError("Autoprobe history must use the same case and budget; start a new run.")
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
    medians = [
        row["median"]
        for row in summaries
        if row.get("median") is not None and math.isfinite(row["median"])
    ]
    dispersions = [
        row["standard_deviation"] for row in summaries if row["standard_deviation"] is not None
    ]
    top_k = report["ranking_stability"].get("top_k", {})
    first_k = top_k.get(str(min(case.decision.k)), {}) if case.decision.k else {}
    guardrails = []
    expected_runs = len(
        case.evaluation.seeds[:2] if candidate.budget == "smoke" else case.evaluation.seeds
    )
    if (
        not medians
        or len(medians) != len(summaries)
        or any(
            row.get("runs") != expected_runs or row.get("finite_runs") != expected_runs
            for row in summaries
        )
        or {row["scenario"] for row in summaries}
        != {scenario.name for scenario in case.generalization_scenarios}
    ):
        guardrails.append(
            "finite primary metrics are required for every declared scenario and scheduled seed"
        )
    suspicious = report["audits"]["leakage"]["suspicious_identifier_features"]
    if suspicious:
        guardrails.append(f"identifier-like configured features: {suspicious}")
    primary = case.evaluation.primary_metric
    higher_is_better = metric_higher_is_better(primary)
    worst_metric = min if higher_is_better else max
    vector = {
        "candidate_id": candidate.candidate_id,
        "case_fingerprint": case_fingerprint,
        "budget": candidate.budget,
        "timestamp": datetime.now(UTC).isoformat(),
        "primary_metric": primary,
        "primary_metric_higher_is_better": higher_is_better,
        "median_primary_development_metric": median(medians) if medians else None,
        "worst_scenario_metric": worst_metric(medians) if medians else None,
        "seed_or_split_dispersion": max(dispersions) if dispersions else None,
        "top_k_stability": first_k.get("average_pairwise_jaccard"),
        "runtime_seconds": report["provenance"]["runtime_seconds"],
        "complexity": candidate.complexity,
        "guardrail_failures": guardrails,
        "holdout_accessed": bool(report.get("holdout_access")),
    }
    baseline = next((entry for entry in reversed(history) if entry.get("decision") == "KEEP"), None)
    decision, reasons = _select(vector, baseline, candidate, higher_is_better=higher_is_better)
    vector["decision"] = decision
    vector["decision_reasons"] = reasons
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(vector, sort_keys=True) + "\n")
    (result_dir / "evaluation-vector.json").write_text(
        json.dumps(vector, indent=2, sort_keys=True), encoding="utf-8"
    )
    return vector


def _select(
    vector: dict[str, Any],
    baseline: dict[str, Any] | None,
    candidate: Candidate,
    *,
    higher_is_better: bool,
) -> tuple[str, list[str]]:
    if vector["guardrail_failures"]:
        return "DISCARD", ["validity/leakage guardrail failed"]
    if vector["runtime_seconds"] > candidate.maximum_runtime_seconds:
        return "DISCARD", ["runtime budget exceeded"]
    if baseline is None:
        return "KEEP", ["first valid candidate establishes the baseline"]
    current = vector["median_primary_development_metric"]
    previous = baseline["median_primary_development_metric"]
    direction = 1 if higher_is_better else -1
    if (
        current is None
        or previous is None
        or direction * (current - previous) < candidate.minimum_delta
    ):
        return "DISCARD", ["primary metric did not improve by the configured minimum delta"]
    worst = vector["worst_scenario_metric"]
    previous_worst = baseline["worst_scenario_metric"]
    if (
        worst is not None
        and previous_worst is not None
        and direction * (worst - previous_worst) < -candidate.maximum_worst_scenario_regression
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
