from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bio_ml_preflight.contracts import CaseSpec
from bio_ml_preflight.contracts.case import ScenarioSpec
from bio_ml_preflight.evaluation.metrics import empirical_permutation_summary

Status = str


def capability_matrix(
    experiments: pd.DataFrame,
    case: CaseSpec,
    ranking: dict[str, Any],
    *,
    audits: dict[str, Any] | None = None,
    overlap_results: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    primary = case.evaluation.primary_metric
    usable = experiments[experiments["model"].ne("dummy")]
    random_values = usable[
        usable["strategy"].isin(["random", "random_pair"]) & usable["permuted"].eq(False)
    ][primary]
    random_median = float(random_values.median()) if random_values.notna().any() else None
    rows: list[dict[str, Any]] = []
    audit_summary = _audit_conflicts(audits or {})
    scenario_overlap_summaries: list[dict[str, float | int]] = []
    for scenario in case.generalization_scenarios:
        scenario_overlap = _overlap_summary(overlap_results or {}, case, scenario)
        scenario_overlap_summaries.append(scenario_overlap)
        current = usable[usable["scenario"].eq(scenario.name)]
        real = current[current["permuted"].eq(False)]
        control = current[current["permuted"].eq(True)]
        controlled_models = set(control["model"])
        by_model = (
            real[real["model"].isin(controlled_models)].groupby("model")[primary].median().dropna()
        )
        if by_model.empty:
            rows.append(
                _verdict(
                    scenario.name,
                    "NOT_ASSESSABLE",
                    [],
                    ["No applicable model produced a finite primary metric."],
                    "Undefined",
                    ["The requested task/metric could not be evaluated."],
                    "Repair the target, features, or split so one baseline can be evaluated.",
                    {},
                )
            )
            continue
        best_model = str(by_model.idxmax())
        best = real[real["model"].eq(best_model)][primary].dropna()
        perm = control[control["model"].eq(best_model)][primary].dropna()
        metric = float(best.median())
        dispersion = float(best.std()) if len(best) > 1 else 0.0
        model_control = control[control["model"].eq(best_model)]
        if "permutation_draw" in model_control and model_control["permutation_draw"].notna().any():
            null_statistics = (
                model_control.groupby("permutation_draw")[primary].median().dropna().tolist()
            )
        else:
            null_statistics = perm.tolist()
        permutation_summary = empirical_permutation_summary(metric, null_statistics)
        permutation = permutation_summary["median"]
        delta = metric - permutation if permutation is not None else None
        permutation_p_value = permutation_summary["p_value"]
        numbers = {
            "primary_metric": primary,
            "median": metric,
            "dispersion": dispersion,
            "best_model": best_model,
            "permutation_median": permutation,
            "permutation_delta": delta,
            "permutation_q95": permutation_summary["q95"],
            "permutation_p_value": permutation_p_value,
            "permutation_draws": permutation_summary["draws"],
        }
        low_metric = metric < case.thresholds.limited_metric
        high_dispersion = dispersion > case.thresholds.maximum_dispersion
        metric_evidence = f"Best baseline median {primary}={metric:.3f} ({best_model})."
        evidence_for = [] if low_metric else [metric_evidence]
        evidence_against: list[str] = []
        if low_metric:
            evidence_against.append(metric_evidence)
        unmet: list[str] = []
        adequate_delta = delta is not None and delta >= case.thresholds.minimum_permutation_delta
        adequate_permutation_p = (
            permutation_p_value is not None
            and permutation_p_value <= case.thresholds.maximum_permutation_p_value
        )
        if delta is None:
            evidence_against.append("Delta over grouped permutation control is unavailable.")
        else:
            (evidence_for if adequate_delta else evidence_against).append(
                f"Delta over grouped permutation control={delta:.3f}."
            )
        if permutation_p_value is None:
            evidence_against.append("Empirical grouped-permutation p-value is unavailable.")
        else:
            (evidence_for if adequate_permutation_p else evidence_against).append(
                f"Empirical grouped-permutation p={permutation_p_value:.3f} "
                f"from {permutation_summary['draws']} draws."
            )
        if scenario.strategy not in {"random", "random_pair"} and random_median is not None:
            evidence_against.append(f"Random-split median across probes={random_median:.3f}.")
        weak_permutation = not (adequate_delta and adequate_permutation_p)
        if high_dispersion:
            evidence_against.append(
                f"Split dispersion {dispersion:.3f} exceeds the configured limit."
            )
        if weak_permutation:
            unmet.append("Permutation separation is insufficient.")
        if low_metric:
            unmet.append("The controlled baseline is below the configured useful metric.")
        if high_dispersion:
            unmet.append("Performance is unstable across the evaluated splits.")
        leakage_contradiction = (
            scenario.strategy not in {"random", "random_pair"}
            and random_median is not None
            and random_median >= case.thresholds.supported_metric
            and metric < case.thresholds.limited_metric
        )
        if leakage_contradiction:
            status = "CONTRADICTED"
            unmet.append(
                "Performance does not transfer from row-random to entity-separated validation."
            )
            next_step = (
                "Collect or reserve outcomes for genuinely unseen entities and validate once."
            )
        elif weak_permutation or low_metric:
            status = "INSUFFICIENT_EVIDENCE"
            if case.holdout.enabled:
                next_step = (
                    "Audit alignment between development data and the external target and "
                    "measurement boundary, including intended deployment, then reserve a new "
                    "untouched confirmation set; do not adapt to or rerun this holdout."
                )
            elif weak_permutation:
                next_step = (
                    "Localize the weak target or split boundary with existing baselines before "
                    "adding data or model capacity."
                )
            else:
                next_step = (
                    "Add independent units along this scenario, then repeat the fixed validation."
                )
        elif metric >= case.thresholds.supported_metric and not high_dispersion:
            status = "SUPPORTED"
            next_step = "Confirm once on a prospectively reserved or pseudo-sealed external set."
        else:
            status = "SUPPORTED_WITH_LIMITS"
            unmet.append("Evidence is moderate or sensitive to the sampled split.")
            next_step = "Increase independent entity coverage at the weakest split boundary."
        verdict = _verdict(
            scenario.name,
            status,
            evidence_for,
            evidence_against,
            f"Across-split standard deviation={dispersion:.3f}.",
            unmet,
            next_step,
            numbers,
        )
        _apply_random_split_limit(verdict, case, scenario, scenario_overlap)
        _apply_audit_limits(
            verdict,
            audit_summary,
            scenario_overlap,
        )
        _apply_test_class_limit(verdict, case, scenario_overlap)
        rows.append(verdict)
    if case.decision.kind == "top_k_per_group":
        k = str(min(case.decision.k))
        top = ranking.get("top_k", {}).get(k, {})
        overlap = top.get("average_pairwise_jaccard")
        if overlap is None:
            status = "NOT_ASSESSABLE"
            evidence_against = ["No comparable grouped rankings were available across runs."]
        elif overlap < case.thresholds.unstable_top_k:
            status = "INSUFFICIENT_EVIDENCE"
            evidence_against = [f"Average pairwise top-{k} Jaccard={overlap:.3f}."]
        elif overlap < case.thresholds.stable_top_k:
            status = "SUPPORTED_WITH_LIMITS"
            evidence_against = [f"Average pairwise top-{k} Jaccard={overlap:.3f} is only moderate."]
        else:
            status = "SUPPORTED"
            evidence_against = []
        verdict = _verdict(
            f"top-{k} selection within {case.decision.group_entity}",
            status,
            [] if overlap is None else [f"Observed top-{k} overlap={overlap:.3f}."],
            evidence_against,
            "Development-run membership variation; no final holdout exposure.",
            [] if status == "SUPPORTED" else ["Candidate membership changes across runs."],
            "Measure the borderline candidates again, then evaluate the locked ranking once.",
            {"average_pairwise_jaccard": overlap},
        )
        combined_overlap = {
            key: max(float(summary.get(key, 0)) for summary in scenario_overlap_summaries)
            for key in {
                "exact_duplicate_overlap",
                "pair_overlap",
                "protected_entity_overlap_count",
                "protected_entity_overlap_fraction",
            }
        }
        _apply_audit_limits(verdict, audit_summary, combined_overlap)
        rows.append(verdict)
    if audits:
        rows.append(_measurement_verdict(audits.get("measurement", {})))
        rows.extend(_missing_metadata_verdicts(audits.get("missing_high_value_metadata", [])))
    return rows


def _audit_conflicts(audits: dict[str, Any]) -> dict[str, int]:
    independence = audits.get("independence", {})
    entities = independence.get("entities", {})
    pair = independence.get("pair_structure", {})
    if pair.get("columns"):
        target_conflicts = int(pair.get("conflicting_label_pairs", 0))
    else:
        target_conflicts = sum(
            int(result.get("conflicting_target_entities", 0))
            for result in entities.values()
            if isinstance(result, dict)
        )
    representation_conflicts = sum(
        int(result.get("inconsistent_representation_entities", 0))
        for result in entities.values()
        if isinstance(result, dict)
    )
    return {
        "conflicting_target_entities": target_conflicts,
        "inconsistent_representation_entities": representation_conflicts,
    }


def _overlap_summary(
    overlap_results: dict[str, Any], case: CaseSpec, scenario: ScenarioSpec
) -> dict[str, Any]:
    protected_columns = {
        value
        for value in (
            scenario.group_column
            if scenario.strategy in {"group", "scaffold", "supplied"}
            else None,
            scenario.left_column if scenario.strategy in {"cold_left", "double_cold"} else None,
            scenario.right_column if scenario.strategy in {"cold_right", "double_cold"} else None,
        )
        if value is not None
    }
    protected_entities = {
        name
        for name, entity in case.entities.items()
        if entity.id_column in protected_columns
        or (scenario.strategy == "scaffold" and entity.representation_column in protected_columns)
    }
    bootstrap_entities = {
        name
        for name, entity in case.entities.items()
        if entity.id_column == case.evaluation.bootstrap_unit
    }
    summary: dict[str, Any] = {
        "exact_duplicate_overlap": 0,
        "pair_overlap": 0,
        "protected_entity_overlap_count": 0,
        "protected_entity_overlap_fraction": 0.0,
        "bootstrap_unit_overlap_count": 0,
        "bootstrap_unit_overlap_fraction": 0.0,
        "test_target_counts": {},
        "test_target_count_unit": None,
    }
    for key, result in overlap_results.items():
        if key.rsplit(":", 1)[0] != scenario.name:
            continue
        summary["exact_duplicate_overlap"] = max(
            int(summary["exact_duplicate_overlap"]),
            int(result.get("exact_duplicate_overlap", 0)),
        )
        summary["pair_overlap"] = max(
            int(summary["pair_overlap"]), int(result.get("pair_overlap") or 0)
        )
        target_counts = result.get("test_target_counts", {})
        if target_counts:
            minimum_counts = summary["test_target_counts"]
            labels = set(minimum_counts) | {str(label) for label in target_counts}
            if minimum_counts:
                summary["test_target_counts"] = {
                    label: min(
                        int(minimum_counts.get(label, 0)),
                        int(target_counts.get(label, 0)),
                    )
                    for label in labels
                }
            else:
                summary["test_target_counts"] = {
                    str(label): int(count) for label, count in target_counts.items()
                }
            summary["test_target_count_unit"] = result.get("test_target_count_unit")
        for entity_name in protected_entities:
            entity_overlap = result.get("entity_overlap", {}).get(entity_name, {})
            summary["protected_entity_overlap_count"] = max(
                int(summary["protected_entity_overlap_count"]),
                int(entity_overlap.get("count", 0)),
            )
            summary["protected_entity_overlap_fraction"] = max(
                float(summary["protected_entity_overlap_fraction"]),
                float(entity_overlap.get("test_fraction", 0.0)),
            )
        for entity_name in bootstrap_entities:
            entity_overlap = result.get("entity_overlap", {}).get(entity_name, {})
            summary["bootstrap_unit_overlap_count"] = max(
                int(summary["bootstrap_unit_overlap_count"]),
                int(entity_overlap.get("count", 0)),
            )
            summary["bootstrap_unit_overlap_fraction"] = max(
                float(summary["bootstrap_unit_overlap_fraction"]),
                float(entity_overlap.get("test_fraction", 0.0)),
            )
    return summary


def _apply_random_split_limit(
    verdict: dict[str, Any],
    case: CaseSpec,
    scenario: ScenarioSpec,
    overlap: dict[str, Any],
) -> None:
    if scenario.strategy not in {"random", "random_pair"}:
        return
    count = int(overlap.get("bootstrap_unit_overlap_count", 0))
    if not count:
        return
    fraction = float(overlap.get("bootstrap_unit_overlap_fraction", 0.0))
    unit = case.evaluation.bootstrap_unit or "independent unit"
    verdict["evidence_against"].append(
        f"The random-split diagnostic repeats {count} {unit} values across train and test "
        f"(maximum test fraction {fraction:.3%})."
    )
    verdict["numbers"]["bootstrap_unit_overlap_count"] = count
    verdict["numbers"]["bootstrap_unit_overlap_fraction"] = fraction
    verdict["unmet_assumptions"].append(
        "A random split does not assess generalization to unseen independent units."
    )
    if verdict["status"] == "SUPPORTED":
        verdict["status"] = "SUPPORTED_WITH_LIMITS"
    if verdict["status"] == "SUPPORTED_WITH_LIMITS":
        verdict["cheapest_next_evidence"] = (
            f"Evaluate the existing fixed baselines with {unit} held out across train and test."
        )


def _apply_audit_limits(
    verdict: dict[str, Any],
    conflicts: dict[str, int],
    overlap: dict[str, float | int],
) -> None:
    target_conflicts = conflicts.get("conflicting_target_entities", 0)
    representation_conflicts = conflicts.get("inconsistent_representation_entities", 0)
    if target_conflicts:
        verdict["evidence_against"].append(
            f"{target_conflicts} identical prediction units have conflicting targets."
        )
        verdict["numbers"]["conflicting_target_entities"] = target_conflicts
    if representation_conflicts:
        verdict["evidence_against"].append(
            f"{representation_conflicts} entity identifiers map to inconsistent representations."
        )
        verdict["numbers"]["inconsistent_representation_entities"] = representation_conflicts
    exact_overlap = int(overlap.get("exact_duplicate_overlap", 0))
    pair_overlap = int(overlap.get("pair_overlap", 0))
    protected_overlap = int(overlap.get("protected_entity_overlap_count", 0))
    if exact_overlap:
        verdict["evidence_against"].append(
            f"Up to {exact_overlap} exact records overlap train and test."
        )
        verdict["numbers"]["exact_duplicate_overlap"] = exact_overlap
    if pair_overlap:
        verdict["evidence_against"].append(
            f"Up to {pair_overlap} entity pairs overlap train and test."
        )
        verdict["numbers"]["pair_overlap"] = pair_overlap
    if protected_overlap:
        fraction = float(overlap.get("protected_entity_overlap_fraction", 0.0))
        entity_label = "entity" if protected_overlap == 1 else "entities"
        verdict["evidence_against"].append(
            f"The declared held-out entity boundary has {protected_overlap} overlapping "
            f"{entity_label} "
            f"(maximum test fraction {fraction:.3%})."
        )
        verdict["numbers"]["protected_entity_overlap_count"] = protected_overlap
        verdict["numbers"]["protected_entity_overlap_fraction"] = fraction
    if not any(
        [target_conflicts, representation_conflicts, exact_overlap, pair_overlap, protected_overlap]
    ):
        return
    if verdict["status"] == "SUPPORTED":
        verdict["status"] = "SUPPORTED_WITH_LIMITS"
    verdict["unmet_assumptions"].append(
        "Entity identity, target consistency, and split isolation must be resolved."
    )
    verdict["cheapest_next_evidence"] = (
        "Resolve conflicting entity records and regenerate overlap-free split manifests."
    )


def _apply_test_class_limit(
    verdict: dict[str, Any], case: CaseSpec, overlap: dict[str, Any]
) -> None:
    required = case.thresholds.minimum_test_class_count
    if case.task.kind != "binary_classification" or required is None:
        return
    counts = {str(key): int(value) for key, value in overlap["test_target_counts"].items()}
    if not counts:
        return
    observed = min(counts.values()) if len(counts) >= 2 else 0
    count_unit = str(overlap.get("test_target_count_unit") or "independent units")
    verdict["numbers"]["test_class_counts"] = counts
    verdict["numbers"]["test_class_count_unit"] = count_unit
    verdict["numbers"]["minimum_test_class_count_required"] = required
    verdict["uncertainty"] += f" Minimum holdout class count={observed}."
    if observed >= required:
        verdict["evidence_supporting"].append(
            f"Both holdout classes meet the configured minimum of {required}."
        )
        return
    verdict["evidence_against"].append(
        f"The minority holdout class has {observed} independent {count_unit} values; "
        f"the configured minimum is {required}."
    )
    if verdict["status"] not in {"NOT_ASSESSABLE", "CONTRADICTED"}:
        verdict["status"] = "INSUFFICIENT_EVIDENCE"
    verdict["unmet_assumptions"].append(
        "The external set must contain enough independent examples from both classes."
    )
    verdict["cheapest_next_evidence"] = (
        f"Add at least {required - observed} independent minority-class holdout examples "
        "under the locked protocol."
    )


def _measurement_verdict(measurement: dict[str, Any]) -> dict[str, Any]:
    if measurement.get("status") == "ASSESSED":
        return _verdict(
            "measurement reliability",
            "SUPPORTED_WITH_LIMITS",
            ["Replicate-aware dispersion proxies were computed."],
            [str(measurement.get("noise_warning", "No proven noise ceiling is available."))],
            "Empirical replicate proxies are not a proven noise ceiling.",
            ["Confirm measurement reliability under the intended assay protocol."],
            "Repeat a representative subset under the intended assay protocol.",
            {key: value for key, value in measurement.items() if key != "status"},
        )
    reason = str(measurement.get("reason", "Replicate evidence was not supplied."))
    return _verdict(
        "measurement reliability",
        "NOT_ASSESSABLE",
        [],
        [reason],
        "Within- and between-replicate variation cannot be separated.",
        ["Technical or biological replicate identifiers are missing."],
        "Add replicate identifiers or repeat a representative subset before confirmation.",
        {},
    )


def _missing_metadata_verdicts(items: list[str]) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        key, _, reason = item.partition(":")
        if key == "replicate identifiers":
            continue
        claim = {
            "batch_id": "batch confounding",
            "time_column": "temporal generalization",
        }.get(key, f"metadata: {key}")
        rows.append(
            _verdict(
                claim,
                "NOT_ASSESSABLE",
                [],
                [reason.strip() or f"Missing {key}."],
                "Required metadata was not supplied.",
                [f"{key} is required for this assessment."],
                f"Add {key} metadata and rerun the deterministic audit.",
                {"missing_metadata": key},
            )
        )
    return rows


def _verdict(
    claim: str,
    status: Status,
    evidence_for: list[str],
    evidence_against: list[str],
    uncertainty: str,
    assumptions: list[str],
    next_step: str,
    numbers: dict[str, Any],
) -> dict[str, Any]:
    return {
        "claim_or_scenario": claim,
        "status": status,
        "evidence_supporting": evidence_for,
        "evidence_against": evidence_against,
        "uncertainty": uncertainty,
        "unmet_assumptions": assumptions,
        "cheapest_next_evidence": next_step,
        "numbers": {
            key: None if isinstance(value, float) and not np.isfinite(value) else value
            for key, value in numbers.items()
        },
    }
