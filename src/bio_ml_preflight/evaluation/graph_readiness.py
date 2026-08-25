from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd


def assess_graph_readiness(
    development_report: Path, external_report: Path | None = None
) -> dict[str, Any]:
    """Assess whether existing evidence justifies one bounded graph-model probe."""
    structured = _read_report(development_report)
    contract = structured.get("audits", {}).get("graph_readiness_contract")
    if not isinstance(contract, dict) or contract.get("status") == "NOT_DECLARED":
        return _verdict(
            structured,
            "NOT_ASSESSABLE",
            [],
            ["No machine-readable graph construction contract was assessed."],
            ["Graph identity, features, support, and a non-random scenario are required."],
            "Declare and run a graph-readiness contract before choosing a graph model.",
            {},
        )
    if contract.get("status") != "VALID_CONTRACT":
        return _verdict(
            structured,
            "NOT_READY_DATA_LIMITED",
            [],
            [f"Graph construction contract status={contract.get('status')}."],
            ["Resolve invalid structures, identity conflicts, or independent support limits."],
            "Repair the unmet graph-contract requirement and rerun the deterministic audit.",
            {"graph_contract": contract},
        )

    scenarios = [str(row["name"]) for row in contract["evaluation_scenarios"]]
    capability = pd.read_parquet(development_report / "capability_matrix.parquet")
    selected = capability[capability["claim_or_scenario"].isin(scenarios)].copy()
    if selected.empty:
        return _verdict(
            structured,
            "NOT_ASSESSABLE",
            [],
            ["No capability rows match the graph contract's evaluation scenarios."],
            ["The declared deployment-relevant split must produce finite baseline evidence."],
            "Repair the selected split or metric before considering a graph model.",
            {"graph_contract": contract},
        )

    graph_overlap = _graph_overlap(
        structured,
        scenarios,
        minimum_class_count=int(contract["support"]["minimum_independent_units_per_class"]),
        independent_unit=str(contract["support"]["unit"]),
    )
    if graph_overlap["status"] != "ASSESSED":
        return _verdict(
            structured,
            "NOT_ASSESSABLE",
            ["The 2D graph construction contract is valid."],
            ["Canonical-graph overlap was not assessed on every selected split."],
            ["Graph identities must be isolated across the selected train/test manifests."],
            "Regenerate the development report with canonical-graph overlap auditing.",
            {"graph_contract": contract, "graph_overlap": graph_overlap},
        )
    if graph_overlap["maximum_count"]:
        return _verdict(
            structured,
            "NOT_READY_DATA_LIMITED",
            ["The 2D graph construction contract is valid."],
            [
                f"Up to {graph_overlap['maximum_count']} canonical graphs cross a selected "
                "train/test boundary."
            ],
            ["Exact graph identities must not cross a claimed unseen-graph boundary."],
            "Group identical canonical graphs and regenerate the split manifests.",
            {"graph_contract": contract, "graph_overlap": graph_overlap},
        )
    if not graph_overlap["class_support_meets_minimum"]:
        return _verdict(
            structured,
            "NOT_READY_DATA_LIMITED",
            ["The 2D graph construction contract is valid."],
            ["At least one selected split does not meet the declared per-class support floor."],
            ["Both test classes need adequate independent support on every selected split."],
            "Add independent minority-class units and regenerate the fixed split manifests.",
            {"graph_contract": contract, "graph_overlap": graph_overlap},
        )

    sensitivity = pd.read_parquet(development_report / "representation_sensitivity.parquet")
    sensitivity = sensitivity[sensitivity["claim_or_scenario"].isin(scenarios)]
    learning = pd.read_parquet(development_report / "learning_curve.parquet")
    learning = learning[learning["scenario"].isin(scenarios)]
    development = _development_summary(selected, sensitivity, learning)
    external = _external_summary(external_report) if external_report else None

    supporting = [
        (
            f"All {contract['rows_converted']} rows convert deterministically to "
            f"{contract['unique_canonical_graphs']} canonical 2D graphs."
        ),
        (
            "Independent class support meets the declared floor: "
            f"{contract['support']['class_counts']}."
        ),
        "Selected development manifests have zero canonical-graph overlap.",
    ]
    opposing = []
    uncertainty = []
    if not development["representation_verdict_changes"]:
        opposing.append(
            "Character and Morgan representations do not change the capability verdict."
        )
    if development["learning_curve_status"] != "ASSESSED":
        uncertainty.append("No learning curve is available for the selected graph scenario.")
    missing_metadata = [
        key for key, available in contract["metadata_available"].items() if not available
    ]
    if missing_metadata:
        uncertainty.append(f"Unavailable metadata remains limiting: {missing_metadata}.")

    external_failure = bool(external and external["adequately_supported_null_failure"])
    if external_failure:
        opposing.append(
            "The supplied external confirmation has adequate class support but does not "
            "separate from its permutation control."
        )
        status = "NOT_JUSTIFIED_BY_CURRENT_EVIDENCE"
        next_step = (
            "Align development data to the intended external target and measurement boundary, "
            "then reserve a new untouched confirmation set; do not tune a graph model on the "
            "consumed external outcomes."
        )
        unmet = [
            "Current evidence does not identify fixed molecular representation as the limiting "
            "boundary."
        ]
    elif development["all_selected_statuses_at_least_limited"]:
        status = "FEASIBLE_BUT_LOW_PRIORITY"
        next_step = (
            "Keep the transparent fixed-representation baseline; predeclare a bounded graph "
            "probe only if a decision-relevant representation gap appears."
        )
        unmet = ["No decision-level capability gap currently requires a learned graph model."]
    elif (
        development["representation_verdict_changes"]
        and development["has_bounded_representation_gap"]
    ):
        status = "READY_FOR_BOUNDED_PROBE"
        next_step = (
            "Predeclare one small development-only graph model on the existing manifests and "
            "stop unless it changes the weakest capability verdict."
        )
        unmet = ["Any selected graph candidate still requires a new untouched confirmation set."]
    else:
        status = "NOT_JUSTIFIED_BY_CURRENT_EVIDENCE"
        next_step = (
            "Localize the weak split or target boundary with existing baselines before adding "
            "model capacity."
        )
        unmet = ["Weak evidence has not been attributed to fixed representation."]

    result = _verdict(
        structured,
        status,
        supporting,
        opposing,
        [*uncertainty, *unmet],
        next_step,
        {
            "graph_contract": contract,
            "graph_overlap": graph_overlap,
            "development": development,
            "external_confirmation": external,
        },
    )
    result["governance"] = {
        "development_only_probe": True,
        "external_outcomes_may_not_select_architecture_or_threshold": True,
        "new_untouched_confirmation_required_after_selection": True,
        "predictive_evidence_is_not_causal": True,
    }
    result["input_provenance"] = {
        "development": _provenance_summary(structured),
        "external_confirmation": (external["provenance"] if external is not None else None),
    }
    return result


def write_graph_readiness(result: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "graph_readiness.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    row = {
        key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        for key, value in result.items()
    }
    pd.DataFrame([row]).to_parquet(output / "graph_readiness.parquet", index=False)


def _read_report(path: Path) -> dict[str, Any]:
    report = path / "report.json"
    if not report.is_file():
        raise ValueError(f"Missing structured report: {report}")
    return cast(dict[str, Any], json.loads(report.read_text(encoding="utf-8")))


def _graph_overlap(
    structured: dict[str, Any],
    scenarios: list[str],
    *,
    minimum_class_count: int,
    independent_unit: str,
) -> dict[str, Any]:
    rows = []
    for key, overlap in structured.get("split_overlap", {}).items():
        if key.rsplit(":", 1)[0] not in scenarios:
            continue
        graph = overlap.get("canonical_graph_overlap")
        if not isinstance(graph, dict) or "count" not in graph:
            return {"status": "NOT_ASSESSABLE", "missing_split": key}
        counts = overlap.get("test_target_counts")
        count_unit = overlap.get("test_target_count_unit")
        if not isinstance(counts, dict) or count_unit != independent_unit:
            return {"status": "NOT_ASSESSABLE", "missing_class_support_split": key}
        rows.append(
            {
                "split": key,
                "count": int(graph["count"]),
                "test_class_counts": {str(label): int(count) for label, count in counts.items()},
            }
        )
    if not rows:
        return {"status": "NOT_ASSESSABLE", "reason": "No selected split overlap rows."}
    return {
        "status": "ASSESSED",
        "maximum_count": max(row["count"] for row in rows),
        "minimum_class_count_required": minimum_class_count,
        "class_support_meets_minimum": all(
            len(row["test_class_counts"]) == 2
            and min(row["test_class_counts"].values()) >= minimum_class_count
            for row in rows
        ),
        "splits": rows,
    }


def _development_summary(
    capability: pd.DataFrame, sensitivity: pd.DataFrame, learning: pd.DataFrame
) -> dict[str, Any]:
    rows = []
    for _, row in capability.iterrows():
        numbers = row["numbers"]
        rows.append(
            {
                "scenario": row["claim_or_scenario"],
                "representation": row.get("representation"),
                "status": row["status"],
                "median": numbers.get("median"),
                "best_model": numbers.get("best_model"),
                "permutation_delta": numbers.get("permutation_delta"),
                "permutation_p_value": numbers.get("permutation_p_value"),
            }
        )
    statuses = {str(row["status"]) for row in rows}
    return {
        "scenario_results": rows,
        "all_selected_statuses_at_least_limited": bool(statuses)
        and statuses
        <= {
            "SUPPORTED",
            "SUPPORTED_WITH_LIMITS",
        },
        "representation_verdict_changes": bool(
            not sensitivity.empty and sensitivity["verdict_changes"].any()
        ),
        "has_bounded_representation_gap": bool(
            statuses & {"INSUFFICIENT_EVIDENCE", "SUPPORTED_WITH_LIMITS"}
            and not statuses & {"CONTRADICTED", "NOT_ASSESSABLE"}
        ),
        "maximum_matched_model_range": (
            float(sensitivity["maximum_matched_model_range"].max())
            if sensitivity["maximum_matched_model_range"].notna().any()
            else None
        ),
        "learning_curve_status": "ASSESSED" if not learning.empty else "NOT_ASSESSABLE",
        "learning_curve_rows": int(len(learning)),
    }


def _external_summary(path: Path) -> dict[str, Any]:
    structured = _read_report(path)
    capability = pd.read_parquet(path / "capability_matrix.parquet")
    scenario_names = {
        str(row["name"]) for row in structured.get("case", {}).get("generalization_scenarios", [])
    }
    rows = capability[capability["claim_or_scenario"].isin(scenario_names)]
    summaries = []
    adequately_supported_null_failure = False
    maximum_p = float(structured["case"]["thresholds"]["maximum_permutation_p_value"])
    for _, row in rows.iterrows():
        numbers = row["numbers"]
        counts = {
            str(label): int(value)
            for label, value in (numbers.get("test_class_counts") or {}).items()
        }
        required = numbers.get("minimum_test_class_count_required")
        adequate = bool(
            required is not None
            and len(counts) == 2
            and min(int(value) for value in counts.values()) >= int(required)
        )
        p_value = numbers.get("permutation_p_value")
        null_failure = bool(
            row["status"] == "INSUFFICIENT_EVIDENCE"
            and adequate
            and p_value is not None
            and float(p_value) > maximum_p
        )
        adequately_supported_null_failure |= null_failure
        summaries.append(
            {
                "scenario": row["claim_or_scenario"],
                "representation": row.get("representation"),
                "status": row["status"],
                "median": numbers.get("median"),
                "permutation_q95": numbers.get("permutation_q95"),
                "permutation_p_value": p_value,
                "test_class_counts": counts,
                "adequate_class_support": adequate,
            }
        )
    return {
        "case_id": structured["case"]["case_id"],
        "scenario_results": summaries,
        "adequately_supported_null_failure": adequately_supported_null_failure,
        "provenance": _provenance_summary(structured),
    }


def _provenance_summary(structured: dict[str, Any]) -> dict[str, Any]:
    provenance = structured.get("provenance", {})
    return {
        key: provenance.get(key)
        for key in [
            "case_spec_hash",
            "dataset_fingerprint",
            "git_commit",
            "git_diff_sha256",
            "git_worktree_dirty",
            "split_manifest_hashes",
        ]
    }


def _verdict(
    structured: dict[str, Any],
    status: str,
    supporting: list[str],
    opposing: list[str],
    unmet: list[str],
    next_evidence: str,
    numbers: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "case_id": structured.get("case", {}).get("case_id"),
        "status": status,
        "evidence_supporting": supporting,
        "evidence_opposing": opposing,
        "uncertainty_and_unmet_assumptions": unmet,
        "cheapest_next_evidence": next_evidence,
        "numbers": numbers,
    }
